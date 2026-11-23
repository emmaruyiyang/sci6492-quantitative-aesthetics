"""F2: the camera rig, drawn to scale. The camera stands off at
r / sin(fov/2) = 2.92 r, so its 40 degree cone is exactly tangent to the
model's bounding sphere."""
import math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc

OUT = sys.argv[1]
INK, MUTED, ACCENT, GRID = "#1a1a1a", "#8c8c8c", "#2f6fd0", "#dddddd"
plt.rcParams.update({"font.family": "Calibri", "font.size": 9, "text.color": INK})

R = 36.0                                   # bounding sphere, as drawn
FOV = 40.0
D = R / math.sin(math.radians(FOV / 2))    # 2.92 R

def pol(deg, r, cx=0.0, cy=0.0):
    a = math.radians(deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)

fig, (plan, sec) = plt.subplots(1, 2, figsize=(11.4, 5.9), dpi=200)
for ax in (plan, sec):
    ax.set_aspect("equal"); ax.axis("off")

# ---- plan
plan.add_patch(Circle((0, 0), R, facecolor="#f2f2f2", edgecolor=MUTED, lw=0.9))
plan.add_patch(Circle((0, 0), D, facecolor="none", edgecolor=GRID, lw=0.9, ls=(0, (4, 4))))
plan.text(0, 0, "bounding\nsphere", ha="center", va="center", color=MUTED, fontsize=8.5)
for k in range(24):
    x, y = pol(k * 15, D)
    plan.plot([x * 0.985, x * 0.42], [y * 0.985, y * 0.42], color=GRID, lw=0.55, zorder=1)
    plan.plot([x], [y], marker="o", ms=4.6,
              color=ACCENT if k in (4, 9) else INK, zorder=3)
plan.annotate("60°  most exposed", xy=pol(60, D), xytext=(66, 142),
              color=ACCENT, fontsize=9.5, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8, shrinkA=0, shrinkB=5))
plan.annotate("135°  most hidden", xy=pol(135, D), xytext=(-172, 142),
              color=ACCENT, fontsize=9.5, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8, shrinkA=0, shrinkB=5))
plan.annotate("", xy=(D + 26, 0), xytext=(D + 4, 0),
              arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
plan.text(D + 30, 0, "0°", va="center", color=MUTED, fontsize=9.5)
plan.text(-172, -138, "plan: 24 cameras, 15° apart, every one aimed at the centre",
          fontsize=10, ha="left", color=INK)
plan.set_xlim(-175, 175); plan.set_ylim(-150, 158)

# ---- section (same scale and limits as the plan)
CZ = 0.0
sec.add_patch(Circle((0, CZ), R, facecolor="none", edgecolor=GRID, lw=0.9, ls=(0, (4, 4))))
sec.plot([-7, 7, 7, -7, -7], [-R + 2, -R + 2, R - 2, R - 2, -R + 2], color=MUTED, lw=1.0)
sec.text(-13, 0, "model", ha="right", va="center", color=MUTED, fontsize=9)

cx, cy = pol(10, D, 0, CZ)
axis = math.degrees(math.atan2(CZ - cy, 0 - cx))
for s_ in (1, -1):
    ang = math.radians(axis + s_ * FOV / 2)
    sec.plot([cx, cx + (D * 1.25) * math.cos(ang)], [cy, cy + (D * 1.25) * math.sin(ang)],
             color=ACCENT, lw=1.0, zorder=2)
sec.plot([cx, 0], [cy, CZ], color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
sec.plot([cx, cx - 66], [cy, cy], color=GRID, lw=0.8, ls=(0, (2, 3)), zorder=1)
sec.add_patch(Arc((cx, cy), 92, 92, theta1=180, theta2=190, color=MUTED, lw=0.9))
sec.annotate("elevation 10\u00b0", xy=(cx - 46, cy - 4), xytext=(cx - 112, cy + 54),
             color=MUTED, fontsize=9, ha="center", va="bottom",
             arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=2, shrinkB=2))
sec.plot([cx], [cy], marker="o", ms=6, color=INK, zorder=4)
sec.text(cx + 7, cy + 2, "camera", color=ACCENT, fontsize=9.5, va="bottom", ha="left")
sec.annotate("fov 40\u00b0", xy=(cx - 125, cy - 72.2), xytext=(cx - 125, cy - 58),
             color=ACCENT, fontsize=9.5, ha="center", va="bottom",
             arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8, shrinkA=1, shrinkB=1))
sec.annotate("", xy=(0, -R - 34), xytext=(cx, -R - 34),
             arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
sec.text(cx / 2, -R - 40, "r / sin(fov / 2)  =  2.92 r", ha="center", va="top",
         color=MUTED, fontsize=9)
sec.text(-172, -138, "section: the cone is tangent to the bounding sphere, so the camera\n"
                     "stands off at 2.92 r \u2014 never inside, never behind the model",
         fontsize=10, ha="left", va="top", color=INK, linespacing=1.5)
sec.set_xlim(-175, 175); sec.set_ylim(-150, 158)

fig.suptitle("F2   The 24-camera orbit: every frame becomes one image to segment",
             x=0.03, ha="left", fontsize=12.5, y=0.97)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{OUT}/F2_camera_rig.png", facecolor="white", bbox_inches="tight", pad_inches=0.12)
print("D / R =", round(D / R, 2))
