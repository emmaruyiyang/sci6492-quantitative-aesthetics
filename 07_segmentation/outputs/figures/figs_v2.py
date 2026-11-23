"""F2 (the camera rig) and F13 (fin width over the exposure curve), laid out
so no label touches another."""
import math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Arc

OUT = sys.argv[1]
EXPOSURE = [0.0478, 0.0488, 0.0633, 0.0598, 0.0731, 0.0644, 0.0703, 0.0531,
            0.0437, 0.0331, 0.0414, 0.0338, 0.0334, 0.0546, 0.0558, 0.0643,
            0.0667, 0.0448, 0.0667, 0.0556, 0.0560, 0.0404, 0.0554, 0.0451]
LO, HI = min(EXPOSURE), max(EXPOSURE)
INK, MUTED, ACCENT, GRID = "#1a1a1a", "#8c8c8c", "#2f6fd0", "#dddddd"
plt.rcParams.update({"font.family": "Calibri", "font.size": 9, "text.color": INK})

def exposure_at(d):
    x = (d % 360) / 15.0
    i = int(math.floor(x)); f = x - i
    return EXPOSURE[i % 24] * (1 - f) + EXPOSURE[(i + 1) % 24] * f

def openness(d):
    return (exposure_at(d) - LO) / (HI - LO)

def pol(deg, r):
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)

# ------------------------------------------------------------------ F2
fig, (plan, sec) = plt.subplots(1, 2, figsize=(11, 5.4), dpi=200)
for ax in (plan, sec):
    ax.set_aspect("equal"); ax.axis("off")

plan.add_patch(Circle((0, 0), 36, facecolor="#f2f2f2", edgecolor=MUTED, lw=0.8))
plan.add_patch(Circle((0, 0), 76, facecolor="none", edgecolor=GRID, lw=0.9, ls=(0, (4, 4))))
plan.text(0, 0, "model", ha="center", va="center", color=MUTED, fontsize=9)
for k in range(24):
    x, y = pol(k * 15, 76)
    ix, iy = pol(k * 15, 40)
    plan.plot([x, ix], [y, iy], color=GRID, lw=0.6, zorder=1)
    plan.plot([x], [y], marker="o", ms=5,
              color=ACCENT if k in (4, 9) else INK, zorder=3)
plan.annotate("60°  most exposed", xy=pol(60, 76), xytext=(96, 112),
              color=ACCENT, fontsize=9.5, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8,
                              shrinkA=0, shrinkB=6))
plan.annotate("135°  most hidden", xy=pol(135, 76), xytext=(-138, 112),
              color=ACCENT, fontsize=9.5, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8,
                              shrinkA=0, shrinkB=6))
plan.annotate("", xy=(100, 0), xytext=(80, 0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
plan.text(104, 0, "0°", va="center", color=MUTED, fontsize=9)
plan.text(-138, -108, "plan: 24 cameras, 15° apart, all aimed at the centre",
          fontsize=10, ha="left", color=INK)
plan.set_xlim(-140, 140); plan.set_ylim(-120, 128)

CZ = 58.0
sec.plot([-120, 120], [0, 0], color=GRID, lw=0.9)
sec.add_patch(Circle((0, CZ), 76, facecolor="none", edgecolor=GRID, lw=0.9, ls=(0, (4, 4))))
sec.plot([-14, 14, 14, -14, -14], [0, 0, 116, 116, 0], color=MUTED, lw=1.0)
sec.text(0, CZ, "model", ha="center", va="center", color=MUTED, fontsize=9)
cx, cy = pol(10, 76); cy += CZ
sec.plot([0, cx], [CZ, cy], color=MUTED, lw=0.9)
sec.plot([cx, cx - 46], [cy, cy], color=GRID, lw=0.8, ls=(0, (2, 3)))
sec.add_patch(Arc((cx, cy), 56, 56, angle=0, theta1=180, theta2=190, color=MUTED, lw=0.9))
sec.text(cx - 32, cy + 5, "10°", color=MUTED, fontsize=9, ha="center", va="bottom")
for s in (1, -1):
    ex, ey = pol(10 + s * 20, 76)
    sec.plot([cx, cx - (cx - 0) * 1.0 + (0 - cx) * 0.0], [cy, cy], alpha=0)   # keep limits stable
    sec.plot([cx, cx - 76 * math.cos(math.radians(10)) - s * 0], [cy, cy], alpha=0)
far = 150
for s in (1, -1):
    ang = math.radians(180 + 10 + s * 20)
    sec.plot([cx, cx + far * math.cos(ang)], [cy, cy + far * math.sin(ang)],
             color=ACCENT, lw=1.0)
sec.plot([cx], [cy], marker="o", ms=6, color=INK, zorder=4)
sec.text(cx + 6, cy, "camera\nfov 40°", color=ACCENT, fontsize=9.5, va="center", ha="left")
sec.annotate("", xy=(0, CZ - 86), xytext=(cx, CZ - 86),
             arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
sec.text(cx / 2, CZ - 94, "distance = r / sin(fov / 2)", ha="center", va="top",
         color=MUTED, fontsize=9)
sec.text(-120, -108, "section: the camera always sits on the bounding sphere,\n"
                     "so no camera can ever be inside the model",
         fontsize=10, ha="left", va="top", color=INK)
sec.set_xlim(-125, 155); sec.set_ylim(-120, 128)

fig.suptitle("F2   The 24-camera orbit: every frame becomes one image to segment",
             x=0.035, ha="left", fontsize=12.5, y=0.97)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{OUT}/F2_camera_rig.png", facecolor="white")
plt.close(fig)

# ----------------------------------------------------------------- F13
fig = plt.figure(figsize=(8.6, 9.4), dpi=200)
ax = fig.add_subplot(111); ax.set_aspect("equal"); ax.axis("off")
R_FIN, C0, C1 = 46.0, 62.0, 96.0

ax.add_patch(Wedge((0, 0), C1 + 14, 55, 92, width=10, facecolor=ACCENT, alpha=0.20, lw=0))
ax.add_patch(Wedge((0, 0), C1 + 14, 135, 180, width=10, facecolor="#cccccc", alpha=0.75, lw=0))
ax.add_patch(Circle((0, 0), 36, facecolor="#f2f2f2", edgecolor=MUTED, lw=0.8))
ax.text(0, 0, "Ratto delle\nSabine", ha="center", va="center", color=MUTED, fontsize=9)

STEP = 2.5
for k in range(int(360 / STEP)):
    deg = k * STEP
    t = openness(deg)
    if t < 0.04:
        continue
    half = t * (STEP / 2) * 0.98
    ax.add_patch(Wedge((0, 0), R_FIN + 2.4, deg - half, deg + half, width=4.8,
                       facecolor=INK, lw=0, zorder=4))
for r in (C0, C1):
    ax.add_patch(Circle((0, 0), r, facecolor="none", edgecolor=GRID, lw=0.7, ls=(0, (3, 3))))

pts = [pol(d, C0 + (C1 - C0) * openness(d)) for d in range(0, 361)]
ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ACCENT, lw=1.8, zorder=5)
for d in range(0, 360, 15):
    ax.plot(*pol(d, C0 + (C1 - C0) * openness(d)), marker="o", ms=3.4, color=ACCENT, zorder=6)

for d, ha, va in ((0, "left", "center"), (90, "center", "bottom"),
                  (180, "right", "center"), (270, "center", "top")):
    ax.text(*pol(d, 126), f"{d}°", ha=ha, va=va, color=MUTED, fontsize=9.5)

ax.annotate("0.073 at 60°\nfins touch — solid", xy=pol(60, C1), xytext=(128, 86),
            color=ACCENT, fontsize=9.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8, shrinkA=0, shrinkB=4))
ax.annotate("0.033 at 135°\nfins vanish — open", xy=pol(135, C0), xytext=(-186, 86),
            color=INK, fontsize=9.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=4))
ax.text(*pol(73.5, C1 + 26), "solid\n55–92°", color=ACCENT, fontsize=9,
        ha="center", va="center")
ax.text(*pol(157, C1 + 26), "open\n135–180°", color="#6f6f6f", fontsize=9,
        ha="center", va="center")

ax.text(-186, 182, "F13   The screen is the curve", fontsize=13.5, ha="left", va="top")
ax.text(-186, 166, "Fin width at each azimuth = the mean CLIPSeg mask value measured there.\n"
                   "135 fins of a possible 144. No arc was chosen by hand.",
        fontsize=9.5, ha="left", va="top", color="#4f4f4f", linespacing=1.5)
ax.text(-186, -168, "blue: mean mask value for “a nude human body” over 24 views"
                    "      black: the fins, on a cylinder of radius 46",
        fontsize=9, ha="left", va="center", color=MUTED)
ax.set_xlim(-190, 190); ax.set_ylim(-178, 190)
fig.savefig(f"{OUT}/F13_plan_over_curve.png", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("ok")
