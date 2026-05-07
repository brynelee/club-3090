"""Generate 3090 power-cap efficiency chart from @noonghunna's water-cooled rig.

Source data: 2026-05-07 sweep, dual-3090 rig (GPU 0 used), water-cooled.
Engine: mainline llama.cpp (ghcr.io/ggml-org/llama.cpp:server-cuda) +
Qwen3.6-27B-UD-Q3_K_XL.gguf, single-stream decode-single.

Sweep methodology: time-bounded streaming bench (10s/direction at each cap).
Total wall: 8m12s for 21 caps from 190-390W in 10W increments. The time-bounded
approach (vs token-bounded) makes per-cap wall constant ~23s regardless of cap,
so total runtime scales linearly with cap count, not throttle severity.
"""
import matplotlib.pyplot as plt

# (cap_W, narr_TPS, code_TPS, actual_W, eff_TPS_per_W) — full 21-cap clean sweep
data = [
    (190, 13.88, 13.69, 189.73, 0.073),
    (200, 15.58, 15.68, 199.71, 0.078),
    (210, 17.68, 17.48, 209.77, 0.084),
    (220, 19.38, 19.28, 219.73, 0.088),
    (230, 21.27, 21.07, 229.71, 0.093),
    (240, 23.17, 22.97, 239.84, 0.097),
    (250, 24.97, 24.77, 249.80, 0.100),
    (260, 26.77, 26.57, 259.86, 0.103),
    (270, 28.57, 28.47, 269.56, 0.106),
    (280, 30.36, 30.77, 279.75, 0.109),
    (290, 32.16, 32.06, 289.37, 0.111),  # ⭐ sweet spot
    (300, 32.76, 32.76, 299.30, 0.109),
    (310, 33.36, 33.26, 309.59, 0.108),
    (320, 33.86, 33.76, 319.47, 0.106),
    (330, 34.37, 34.26, 329.47, 0.104),
    (340, 34.46, 34.25, 333.70, 0.103),  # boost-state plateau begins
    (350, 34.46, 34.36, 334.00, 0.103),
    (360, 34.36, 34.37, 333.97, 0.103),
    (370, 34.36, 34.26, 334.02, 0.103),  # stock TDP, plateau holds
    (380, 35.36, 35.26, 361.30, 0.098),  # plateau ends, draw jumps to 361
    (390, 36.06, 35.96, 388.72, 0.093),  # max — 388W draw at 390W cap
]

caps = [d[0] for d in data]
narr = [d[1] for d in data]
code = [d[2] for d in data]
draw = [d[3] for d in data]
eff = [d[4] for d in data]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, ax1 = plt.subplots(figsize=(11, 6.4), dpi=150)

# Left axis: TPS
color_narr = "#1f77b4"
color_code = "#2ca02c"
ax1.plot(caps, narr, "o-", color=color_narr, linewidth=2.2, markersize=6,
         label="Narrative TPS", zorder=3)
ax1.plot(caps, code, "s-", color=color_code, linewidth=2.2, markersize=6,
         label="Code TPS", zorder=3)
ax1.set_xlabel("Power cap (W)", fontsize=13)
ax1.set_ylabel("Wall TPS (single-stream, llama.cpp mainline)", fontsize=13)
ax1.set_xlim(185, 395)
ax1.set_ylim(11, 39)
ax1.grid(True, alpha=0.3, zorder=0)
ax1.tick_params(axis="both", labelsize=11)

# Right axis: TPS/W efficiency
ax2 = ax1.twinx()
color_eff = "#d62728"
ax2.plot(caps, eff, "^--", color=color_eff, linewidth=1.8, markersize=5,
         alpha=0.9, label="Efficiency (narr TPS/W)", zorder=2)
ax2.set_ylabel("Efficiency: TPS/W (narrative)", color=color_eff, fontsize=13)
ax2.tick_params(axis="y", labelcolor=color_eff, labelsize=11)
ax2.set_ylim(0.07, 0.118)

# Sweet spot annotation: 290W
ax1.axvline(290, color="goldenrod", linestyle=":", alpha=0.5, linewidth=1.5)
ax1.annotate(
    "★ 290W cap\n0.111 TPS/W (best efficiency)\n32.2 narr / 32.1 code\n78% of stock TDP",
    xy=(290, 32.16),
    xytext=(220, 27),
    fontsize=10.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="goldenrod", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.5),
    zorder=4,
)

# Boost-state plateau region (340-370W → all 334W draw)
ax1.axvspan(335, 375, alpha=0.10, color="orange", zorder=0)
ax1.text(355, 12.5, "boost-state plateau\n(caps 340-370W → ~334W draw)",
         fontsize=9.5, ha="center", color="#aa5500", fontstyle="italic")

# Stock TDP marker at 370W
ax1.axvline(370, color="#888", linestyle="--", alpha=0.6, linewidth=1.2)
ax1.annotate(
    "stock TDP\n370W (GPU 0)",
    xy=(370, 36.5),
    xytext=(372, 36.8),
    fontsize=10,
    ha="left",
    color="#555",
    fontstyle="italic",
)

# Title
ax1.set_title(
    "RTX 3090 + Qwen3.6-27B + llama.cpp — power-cap efficiency curve",
    pad=14,
)

# Subtitle
fig.text(
    0.5, 0.92,
    "1× 3090 water-cooled (GPU 0 of dual-3090 rig), mainline llama.cpp + Q3_K_XL GGUF, "
    "time-bounded single-stream  |  data: @noonghunna",
    ha="center", fontsize=10, color="#666",
    style="italic",
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="lower right", fontsize=11, framealpha=0.95,
           edgecolor="#ccc")

# Footer
fig.text(
    0.99, 0.01,
    "github.com/noonghunna/club-3090",
    ha="right", fontsize=9, color="#888", style="italic",
)

plt.tight_layout(rect=(0, 0.02, 1, 0.92))

out = "/tmp/power_cap_sweep_3090_qwen36.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
