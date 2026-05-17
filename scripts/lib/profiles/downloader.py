"""v0.8.0 Pull-Emit-Derived `[E]` — STEP E2: the HF download stage.

CONTRACT-3 (the brief's locked E2 spec). This module owns ONLY the download
stage; it does NOT wire into `run_pull()` (E4), does NOT boot or emit capture
artifacts (E3), does NOT write docs (E5). It returns a structured
`DownloadResult` — the §6 capture-point-2 payload shape — which E3 will later
emit as an artifact; E2 just returns the struct.

Public API (stable for E3/E4)
-----------------------------

    from scripts.lib.profiles import downloader

    res = downloader.download_model(einput, *, fetcher=None)
    #   res: DownloadResult
    #     .ok           -> bool
    #     .files        -> list[str]   (the fetched DOWNLOAD_SET)
    #     .bytes        -> int         (Σ size of staged files)
    #     .sha_verified -> bool        (every *.safetensors etag-verified)
    #     .failure      -> None | "no-etag" | "sha-mismatch" | "gated-401"
    #                          | "disk" | "hf-cli-missing"
    #     .detail       -> str  (actionable message for "hf-cli-missing")
    #     .local_dir    -> str         (the CONTRACT-2 host --model dir, or
    #                                   the .incomplete path on failure)

CONTRACT-3 invariants enforced here
-----------------------------------
* Fetch EXACTLY `deriver.download_set(api)` — `hf download <slug>
  --local-dir <staging> --include <pat>` with ONE `--include` per
  `download_set` entry (exact filenames are valid globs). The include set
  IS the literal shared `download_set` list `[C2a]` sized. A test asserts
  fetched-set == sized-set (nothing extra).
* SHA: every `*.safetensors` — HF HEAD -> `x-linked-etag` -> SHA256(file)
  == etag. **UNLIKE `setup.sh:434/437` (prints SKIP, does NOT count a
  failure on missing etag), a missing/empty `x-linked-etag` is a HARD E2
  failure** (`failure="no-etag"`). Never silently trust an unverifiable
  multi-GB weight. SHA mismatch -> `failure="sha-mismatch"`. Metadata
  files: presence + size only.
* Atomic staging: download into
  `<hf_home>/club3090/pulls/<slug-sanitized>/.incomplete/`; on full success
  + all SHA verified, atomically move into
  `<hf_home>/club3090/pulls/<slug-sanitized>/` (the CONTRACT-2 host
  `--model` dir E1 emits the mount for). On ANY failure delete the
  `.incomplete` tree — no corrupt/partial residue (the `aria2c`
  corruption-incident lesson).
* The real fetcher shells out to the `hf` CLI (the SAME established
  pattern as `setup.sh:404-423` — prefer `hf`, legacy fallback
  `huggingface-cli`, else a structured actionable failure). It is NOT the
  `huggingface_hub` Python library: on this stack `huggingface_hub` is not
  installed in the bare `python3` that `scripts/pull.sh` runs under (it
  exists only as a `uv tool` exposing the `hf` CLI). An on-rig E5 run
  caught the prior lib-import as `ModuleNotFoundError("No module named
  'huggingface_hub'")`. The fetcher is injectable so tests are hermetic (a
  recorded-fixture fetcher / mocked subprocess; NO live multi-GB network
  in CI).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import deriver as D

_HF_RESOLVE = "https://huggingface.co"
_NET_TIMEOUT = 60  # seconds (HEAD for etag is cheap; weight bytes via hub)
_SHA_CHUNK = 8 * 1024 * 1024  # 8 MiB read window for the local SHA256


# ---------------------------------------------------------------------------
# Structured result — this IS the §6 capture-point-2 payload shape.
# E3 emits it as an artifact; E2 only returns the struct (NO artifact write).
# ---------------------------------------------------------------------------
@dataclass
class DownloadResult:
    ok: bool
    files: list[str] = field(default_factory=list)
    bytes: int = 0
    sha_verified: bool = False
    # None | "no-etag" | "sha-mismatch" | "gated-401" | "disk"
    #      | "hf-cli-missing"
    failure: Optional[str] = None
    local_dir: str = ""
    # Actionable detail (populated for "hf-cli-missing"; the canonical
    # install hint mirrored from setup.sh:418-421). Optional / additive —
    # existing failure paths leave it "".
    detail: str = ""


# The canonical actionable message when NEITHER `hf` nor `huggingface-cli`
# resolves — mirrors setup.sh:418-421 verbatim in intent.
_HF_CLI_MISSING_MSG = (
    "neither 'hf' nor 'huggingface-cli' found. Install with: "
    "pip install 'huggingface-hub[hf_transfer]'  or:  "
    "uv tool install --with hf_transfer huggingface-hub"
)


# ---------------------------------------------------------------------------
# slug sanitization — the CONTRACT-2 on-disk layout key.
# `<hf_home>/club3090/pulls/<slug-sanitized>/` is the single source referenced
# by --model, the volume mount, cleanup, and every capture artifact.
# ---------------------------------------------------------------------------
def sanitize_slug(slug: str) -> str:
    """`org/Model-Name` -> a filesystem-safe single path component:
    lowercase, every non-`[a-z0-9._-]` -> `-`, collapse repeats, strip
    leading/trailing separators. Mirrors the CONTRACT-2 served-model-name
    sanitation so the on-disk dir and the compose --model agree."""
    s = slug.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-._") or "model"


def pull_dir(hf_home: Path, slug: str) -> Path:
    """The CONTRACT-2 host `--model` dir for `slug`."""
    return Path(hf_home) / "club3090" / "pulls" / sanitize_slug(slug)


def incomplete_dir(hf_home: Path, slug: str) -> Path:
    """The atomic-staging `.incomplete` tree (deleted on ANY failure)."""
    return pull_dir(hf_home, slug) / ".incomplete"


# ---------------------------------------------------------------------------
# Fetcher abstraction (injectable; default = the `hf` CLI subprocess).
#
# A fetcher must provide:
#   .snapshot(repo_id, local_dir, allow_patterns)  -> list[str]
#       fetch EXACTLY allow_patterns into local_dir; return the relative
#       filenames actually written.
#   .head_etag(repo_id, filename)                  -> str | None
#       HF HEAD -> the `x-linked-etag` (the LFS SHA256) or None/"" if absent.
# Tests inject a recorded-fixture fetcher (no network / no subprocess). The
# real fetcher is below; it shells out to the `hf` CLI (NOT the
# huggingface_hub Python library — that import is unavailable in the bare
# python3 `scripts/pull.sh` runs under; see module docstring + E5).
# ---------------------------------------------------------------------------
class _MissingHfCli(RuntimeError):
    """NEITHER `hf` nor `huggingface-cli` resolvable. download_model maps
    this to the structured DownloadResult(failure="hf-cli-missing") with the
    canonical actionable detail — it is NEVER allowed to escape as a bare
    exception (the on-rig E5 ModuleNotFoundError regression)."""


def _resolve_hf_cli() -> list[str]:
    """Resolve the HF download CLI, mirroring setup.sh:411-423:
    prefer `hf`; else legacy `huggingface-cli`; else raise `_MissingHfCli`
    (download_model turns that into a structured failure — never a bare
    ModuleNotFoundError)."""
    hf = shutil.which("hf")
    if hf:
        return [hf]
    legacy = shutil.which("huggingface-cli")
    if legacy:
        return [legacy]
    raise _MissingHfCli(_HF_CLI_MISSING_MSG)


class HubFetcher:
    """Real fetcher. `snapshot` shells out to the `hf` CLI
    (`hf download <slug> --local-dir <staging> --include <pat> ...`) — the
    SAME established tool/pattern as `setup.sh:404-423`, NOT the
    `huggingface_hub` Python library (unavailable under the bare python3
    `pull.sh` runs; on-rig E5 caught the prior lib-import as
    ModuleNotFoundError). One `--include` per `download_set` entry so the
    fetched set is EXACTLY the allowlist (CONTRACT-3 / Codex-r7 High-1).
    `head_etag` reads HF's `x-linked-etag` (the canonical LFS SHA256) via a
    cheap HEAD — the SAME pattern as `setup.sh:434`; the no-etag DECISION
    (hard fail vs skip) lives in `download_model`, NOT here."""

    timeout = _NET_TIMEOUT

    def snapshot(
        self, repo_id: str, local_dir: str, allow_patterns: list[str]
    ) -> list[str]:
        # Resolve `hf` -> `huggingface-cli` -> structured missing (the
        # established setup.sh pattern; lib-import would ModuleNotFoundError
        # under the bare python3 pull.sh runs — the E5 regression).
        base = _resolve_hf_cli()
        argv = [*base, "download", repo_id, "--local-dir", str(local_dir)]
        for pat in allow_patterns:
            # Exact filenames are valid globs; one --include per
            # download_set entry => fetched set == download_set, nothing
            # extra (CONTRACT-3 exact-set guarantee / Codex-r7 High-1).
            argv += ["--include", pat]
        env = dict(os.environ)
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        proc = subprocess.run(
            argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            blob = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
            # Map non-zero exit to the EXISTING structured failure tokens,
            # exactly as the fixture fetcher would have raised.
            if (
                "401" in blob
                or "403" in blob
                or "gated" in blob
                or "authenticat" in blob
                or "access to model" in blob
                or "awaiting a review" in blob
            ):
                raise GatedError(proc.stderr or "gated-401")
            if (
                "enospc" in blob
                or "no space left" in blob
                or "disk quota" in blob
                or "oserror" in blob
            ):
                raise DiskError(proc.stderr or "disk")
            # Anything else still must NOT escape as a bare exception —
            # surface it as a disk-class staging failure (cleaned + struct).
            raise DiskError(
                proc.stderr or proc.stdout or "hf download failed"
            )
        out: list[str] = []
        root = Path(local_dir)
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(root)))
        return out

    def head_etag(self, repo_id: str, filename: str) -> Optional[str]:
        url = f"{_HF_RESOLVE}/{repo_id}/resolve/main/{filename}"
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                etag = resp.headers.get("x-linked-etag") or resp.headers.get(
                    "X-Linked-Etag"
                )
        except urllib.error.HTTPError as exc:  # 401/403/404 -> no usable etag
            if exc.code in (401, 403):
                return "__gated-401__"
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
        if not etag:
            return None
        return etag.strip().strip('"')


# A recorded HEAD/snapshot is what tests inject; HubFetcher is the default.
Fetcher = HubFetcher


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_SHA_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _rmtree(p: Path) -> None:
    try:
        if p.exists():
            shutil.rmtree(p)
    except OSError:  # pragma: no cover - defensive
        pass


def set_probe_fetcher(einput, fetcher: Optional[Any] = None):
    """Standardized wiring for E1's deferred dtype header-probe.

    E1's `generate_compose._resolve_compute_dtype()` step (2) calls the
    deriver's existing `probe_safetensors_dtype()` ONLY when a usable
    fetcher is present at `einput.diagnostics["fetcher"]` — in E1 that key
    was never populated, so the probe path was dead (resolution fail-closed
    at step (3)). E2 standardizes HOW it is populated here: this helper sets
    `einput.diagnostics["fetcher"]` to the canonical bounded-header-probe
    fetcher (`deriver.default_probe_fetcher()` by default; a fixture fetcher
    in tests). It does NOT alter E1's resolution ORDER or any other E1
    semantics — it only supplies the fetcher object E1 already looks for, so
    a quantized repo lacking `torch_dtype` can now resolve `--dtype` via the
    probe instead of fail-closing. E4 calls this inside `run_pull()` (NOT
    wired here — that is E4); E2 only defines + tests the standardizer.

    Returns `einput` (mutated in place) for call-chaining convenience.
    """
    if fetcher is None:
        fetcher = D.default_probe_fetcher()
    if einput.diagnostics is None:  # pragma: no cover - EInput defaults {}
        einput.diagnostics = {}
    einput.diagnostics["fetcher"] = fetcher
    return einput


def _selected_files_api(einput) -> dict:
    """The raw HF siblings API the deriver stashed on the profile (CONTRACT-3
    additive surface). E2 derives the allowlist from this — the SAME
    `download_set()` `[C2a]` sized."""
    prof = getattr(einput.der, "profile", None) or {}
    return prof.get("_hf_api") or {}


# ---------------------------------------------------------------------------
# Fetcher-side structured signals (a fixture/real fetcher raises these to
# tell the stage WHY a fetch failed; download_model maps them to the
# CONTRACT-3 failure tokens + always cleans .incomplete).
# ---------------------------------------------------------------------------
class GatedError(RuntimeError):
    """HF gated 401/403 surfaced by the fetcher during snapshot/HEAD."""


class DiskError(RuntimeError):
    """Out-of-space / staging IO failure surfaced by the fetcher."""


# ---------------------------------------------------------------------------
# THE download stage.
# ---------------------------------------------------------------------------
def download_model(einput, *, fetcher: Optional[Any] = None) -> DownloadResult:
    """Fetch EXACTLY `deriver.download_set(api)`, etag-SHA-verify every
    `*.safetensors`, atomically stage into the CONTRACT-2 host `--model`
    dir. Returns the structured `DownloadResult` (== §6 capture-point-2
    payload). Pure w.r.t. side effects beyond the staging dir; E4 wires it,
    E3 emits the artifact.

    CONTRACT-3 failure semantics (each deletes the `.incomplete` tree — no
    corrupt residue):
      * missing/empty `x-linked-etag`  -> failure="no-etag"  (HARD fail —
        UNLIKE setup.sh which would SKIP; we never trust an unverifiable
        multi-GB weight)
      * SHA(file) != etag              -> failure="sha-mismatch"
      * HF gated 401/403 on fetch/HEAD -> failure="gated-401"
      * staging-disk failure           -> failure="disk"
      * neither `hf` nor `huggingface-cli` resolvable
                                       -> failure="hf-cli-missing"
        (STRUCTURED, with actionable .detail — NEVER a bare
        ModuleNotFoundError; the on-rig E5 regression this closes)
    """
    if fetcher is None:
        fetcher = HubFetcher()

    slug = einput.slug
    api = _selected_files_api(einput)
    # The ONE shared allowlist — identical object [C2a] sized + E3 will smoke.
    allow = D.download_set(api)

    hf_home = Path(einput.hf_home)
    final_dir = pull_dir(hf_home, slug)
    staging = final_dir / ".incomplete"

    # Fresh staging tree (never reuse a prior partial — aria2c lesson).
    _rmtree(staging)
    try:
        staging.mkdir(parents=True, exist_ok=True)
    except OSError:
        return DownloadResult(
            ok=False,
            failure="disk",
            local_dir=str(staging),
            files=[],
        )

    # --- fetch EXACTLY the shared download_set ----------------------------
    try:
        written = fetcher.snapshot(
            repo_id=slug,
            local_dir=str(staging),
            allow_patterns=list(allow),
        )
    except _MissingHfCli as exc:
        # NEITHER `hf` nor `huggingface-cli` resolvable. STRUCTURED failure
        # with the canonical actionable detail — NEVER a bare/​swallowed
        # ModuleNotFoundError (the on-rig E5 regression this fix closes).
        _rmtree(staging)
        return DownloadResult(
            ok=False, failure="hf-cli-missing", local_dir=str(staging),
            files=[], detail=str(exc) or _HF_CLI_MISSING_MSG,
        )
    except GatedError:
        _rmtree(staging)
        return DownloadResult(
            ok=False, failure="gated-401", local_dir=str(staging), files=[]
        )
    except DiskError:
        _rmtree(staging)
        return DownloadResult(
            ok=False, failure="disk", local_dir=str(staging), files=[]
        )

    written_set = sorted(set(written))

    # --- SHA: every *.safetensors via HF x-linked-etag --------------------
    sha_ok = True
    safetensors = [n for n in written_set if n.endswith(".safetensors")]
    for name in safetensors:
        try:
            etag = fetcher.head_etag(slug, name)
        except GatedError:
            _rmtree(staging)
            return DownloadResult(
                ok=False, failure="gated-401", local_dir=str(staging),
                files=written_set,
            )
        if etag == "__gated-401__":
            _rmtree(staging)
            return DownloadResult(
                ok=False, failure="gated-401", local_dir=str(staging),
                files=written_set,
            )
        # CONTRACT-3 / Codex-r6 High-1: missing/empty etag is a HARD fail.
        # This is the DELIBERATE opposite of setup.sh:437 ("SKIP (no etag)").
        if not etag:
            _rmtree(staging)
            return DownloadResult(
                ok=False, failure="no-etag", local_dir=str(staging),
                files=written_set,
            )
        actual = _sha256_file(staging / name)
        if actual.lower() != str(etag).lower():
            _rmtree(staging)
            return DownloadResult(
                ok=False, failure="sha-mismatch", local_dir=str(staging),
                files=written_set,
            )
    sha_ok = True  # every *.safetensors verified (or there were none)

    # --- bytes (Σ staged file sizes; metadata = presence + size only) -----
    total_bytes = 0
    for name in written_set:
        fp = staging / name
        try:
            total_bytes += fp.stat().st_size
        except OSError:
            pass

    # --- atomic move: .incomplete -> final (ONLY on full success) ---------
    try:
        # The final dir currently is the PARENT of .incomplete; move the
        # verified tree up and drop .incomplete. Stage into a sibling temp,
        # then rename onto final_dir atomically.
        tmp_final = final_dir.parent / (sanitize_slug(slug) + ".staged")
        _rmtree(tmp_final)
        # move every verified file out of .incomplete into tmp_final
        tmp_final.mkdir(parents=True, exist_ok=True)
        for name in written_set:
            src = staging / name
            dst = tmp_final / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        _rmtree(staging)          # drop the now-empty .incomplete tree
        _rmtree(final_dir)        # remove the (now only-.incomplete) dir
        os.replace(str(tmp_final), str(final_dir))  # atomic rename onto final
    except OSError:
        _rmtree(staging)
        _rmtree(final_dir.parent / (sanitize_slug(slug) + ".staged"))
        return DownloadResult(
            ok=False, failure="disk", local_dir=str(final_dir),
            files=written_set,
        )

    return DownloadResult(
        ok=True,
        files=written_set,
        bytes=total_bytes,
        sha_verified=sha_ok,
        failure=None,
        local_dir=str(final_dir),
    )
