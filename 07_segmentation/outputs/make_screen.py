"""The occluding screen: vertical fins on a cylinder around the sculpture,
each fin as wide as that azimuth is exposed. Where the sweep found the most
"a nude human body" the fins touch and the screen is solid; where it found
the least they vanish and the sculpture is open to view."""
import math, sys
import rhino3dm

SRC = sys.argv[1]
DST = sys.argv[2]

# mean of the "a nude human body" mask per frame, azimuth 0..345 every 15
EXPOSURE = [0.0478, 0.0488, 0.0633, 0.0598, 0.0731, 0.0644, 0.0703, 0.0531,
            0.0437, 0.0331, 0.0414, 0.0338, 0.0334, 0.0546, 0.0558, 0.0643,
            0.0667, 0.0448, 0.0667, 0.0556, 0.0560, 0.0404, 0.0554, 0.0451]
LO, HI = min(EXPOSURE), max(EXPOSURE)

CX, CY = 28.14, 22.38          # the model's vertical axis
RADIUS = 46.0                  # the screen stands 10 units clear of the model
Z0, Z1 = -2.0, 120.0
STEP = 2.5                     # degrees between fins
THICK = 0.9
SEGMENTS = 3                   # angular segments across one fin


def exposure_at(deg: float) -> float:
    """The measured curve, read cyclically between its 15-degree samples."""
    x = (deg % 360) / 15.0
    i = int(math.floor(x))
    f = x - i
    return EXPOSURE[i % 24] * (1 - f) + EXPOSURE[(i + 1) % 24] * f


def openness(deg: float) -> float:
    """0 where the sculpture is least exposed, 1 where it is most."""
    return (exposure_at(deg) - LO) / (HI - LO)


def add_fin(mesh: rhino3dm.Mesh, a0: float, a1: float) -> None:
    ri, ro = RADIUS - THICK / 2, RADIUS + THICK / 2
    base = len(mesh.Vertices)
    for s in range(SEGMENTS + 1):
        a = math.radians(a0 + (a1 - a0) * s / SEGMENTS)
        c, si = math.cos(a), math.sin(a)
        for r in (ri, ro):
            for z in (Z0, Z1):
                mesh.Vertices.Add(CX + r * c, CY + r * si, z)
    # 4 vertices per angular step: inner-bottom, inner-top, outer-bottom, outer-top
    for s in range(SEGMENTS):
        p, q = base + 4 * s, base + 4 * (s + 1)
        mesh.Faces.AddFace(p + 0, q + 0, q + 1, p + 1)       # inner
        mesh.Faces.AddFace(p + 2, p + 3, q + 3, q + 2)       # outer
        mesh.Faces.AddFace(p + 1, q + 1, q + 3, p + 3)       # top
        mesh.Faces.AddFace(p + 0, p + 2, q + 2, q + 0)       # bottom
    for p, flip in ((base, False), (base + 4 * SEGMENTS, True)):
        a, b, c, d = p + 0, p + 1, p + 3, p + 2
        mesh.Faces.AddFace(*((d, c, b, a) if flip else (a, b, c, d)))


screen = rhino3dm.Mesh()
fins = 0
for k in range(int(360 / STEP)):
    deg = k * STEP
    t = openness(deg)
    if t < 0.04:                       # fully open: no fin at all
        continue
    half = t * (STEP / 2) * 0.98       # t = 1 -> neighbouring fins touch
    add_fin(screen, deg - half, deg + half)
    fins += 1
screen.Normals.ComputeNormals()
screen.Compact()

src = rhino3dm.File3dm.Read(SRC)
out = rhino3dm.File3dm()
out.Settings.ModelUnitSystem = src.Settings.ModelUnitSystem

sculpture = rhino3dm.Layer()
sculpture.Name = "sculpture"
sculpture.Color = (170, 170, 170, 255)
li_s = out.Layers.Add(sculpture)
shield = rhino3dm.Layer()
shield.Name = "screen"
shield.Color = (40, 40, 40, 255)
li_c = out.Layers.Add(shield)

a_s = rhino3dm.ObjectAttributes()
a_s.LayerIndex = li_s
out.Objects.AddMesh(src.Objects[0].Geometry, a_s)
a_c = rhino3dm.ObjectAttributes()
a_c.LayerIndex = li_c
out.Objects.AddMesh(screen, a_c)
out.Write(DST, 7)

solid = [k * STEP for k in range(int(360 / STEP)) if openness(k * STEP) > 0.85]
open_ = [k * STEP for k in range(int(360 / STEP)) if openness(k * STEP) < 0.04]
print(f"fins: {fins} of {int(360/STEP)}   screen vertices: {len(screen.Vertices)}")
print(f"solid (>85% closed): {solid[0]:.0f}-{solid[-1]:.0f} deg")
print(f"fully open: {open_[0]:.0f}-{open_[-1]:.0f} deg" if open_ else "fully open: none")
print("wrote", DST)
