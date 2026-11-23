"""F9: exposure by camera azimuth, drawn to match F2 and F13."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
INK, MUTED, ACCENT, GRID = "#1a1a1a", "#8c8c8c", "#2f6fd0", "#dddddd"
plt.rcParams.update({"font.family": "Calibri", "font.size": 9, "text.color": INK})

AZ = list(range(0, 360, 15))
SERIES = {
    "a nude human body": [0.0478, 0.0488, 0.0633, 0.0598, 0.0731, 0.0644, 0.0703, 0.0531,
                          0.0437, 0.0331, 0.0414, 0.0338, 0.0334, 0.0546, 0.0558, 0.0643,
                          0.0667, 0.0448, 0.0667, 0.0556, 0.0560, 0.0404, 0.0554, 0.0451],
    "a face":            [0.0204, 0.0139, 0.0166, 0.0155, 0.0132, 0.0127, 0.0213, 0.0201,
                          0.0174, 0.0189, 0.0186, 0.0168, 0.0183, 0.0135, 0.0216, 0.0199,
                          0.0253, 0.0212, 0.0193, 0.0188, 0.0180, 0.0185, 0.0186, 0.0194],
    "a hand gripping a body": [0.0139, 0.0126, 0.0169, 0.0179, 0.0182, 0.0176, 0.0242, 0.0162,
                               0.0160, 0.0178, 0.0142, 0.0146, 0.0148, 0.0170, 0.0170, 0.0199,
                               0.0190, 0.0186, 0.0201, 0.0169, 0.0196, 0.0199, 0.0229, 0.0171],
}
ACC = "a nude human body"
vals = SERIES[ACC]
hi_i, lo_i = vals.index(max(vals)), vals.index(min(vals))
ratio = max(vals) / min(vals)

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
ax.axvspan(55, 92, color=ACCENT, alpha=0.08, lw=0, zorder=0)
ax.axvspan(135, 180, color="#bdbdbd", alpha=0.22, lw=0, zorder=0)
ax.text(73.5, 0.0945, "solid 55–92°", color=ACCENT, fontsize=8.5, ha="center")
ax.text(157, 0.0945, "open 135–180°", color="#6f6f6f", fontsize=8.5, ha="center")

for name, v in SERIES.items():
    on = name == ACC
    ax.plot(AZ, v, color=ACCENT if on else MUTED, lw=2.0 if on else 1.2,
            alpha=1 if on else 0.75, marker="o", ms=3.6 if on else 2.6,
            zorder=3 if on else 2, label=name)

ax.annotate(f"{max(vals):.3f} at {AZ[hi_i]}°", (AZ[hi_i], max(vals)),
            xytext=(AZ[hi_i], max(vals) + 0.0068), color=ACCENT, fontsize=9.5, ha="center")
ax.annotate(f"{min(vals):.3f} at {AZ[lo_i]}°", (AZ[lo_i], min(vals)),
            xytext=(AZ[lo_i], min(vals) - 0.0058), color=ACCENT, fontsize=9.5,
            ha="center", va="top")

ax.set_xlim(-12, 357); ax.set_ylim(0, 0.1)
ax.set_xticks(list(range(0, 360, 45)))
ax.set_xticklabels([f"{d}°" for d in range(0, 360, 45)])
ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
ax.set_xlabel("camera azimuth", color=MUTED, labelpad=8)
ax.set_ylabel("mean mask value per frame", color=MUTED, labelpad=8)
ax.tick_params(colors=MUTED, length=3)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7, zorder=0)
ax.set_axisbelow(True)

leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
                frameon=False, handlelength=2.2, fontsize=9)
for t in leg.get_texts():
    t.set_color(MUTED)

fig.suptitle("F9   Exposure by camera azimuth", x=0.055, ha="left", fontsize=12.5, y=0.985)
ax.set_title(f"The most exposed view is {ratio:.1f}× the most hidden one · "
             f"CLIPSeg, 24 horizontal views, 15° apart",
             loc="left", color="#4f4f4f", fontsize=9.5, pad=22)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(f"{OUT}/F9_exposure_by_azimuth.png", facecolor="white")
print("ratio", round(ratio, 2))
