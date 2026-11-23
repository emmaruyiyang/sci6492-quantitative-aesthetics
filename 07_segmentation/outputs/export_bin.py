import json, struct, numpy as np, rhino3dm as r3
from pathlib import Path

SRC = "07_segmentation/outputs/sabine_with_screen.3dm"
OUT = Path("/private/tmp/claude-501/-Users-ruyiyang-Documents-courseware-2026-Fall-SCI6492-Quantitative-Aesthetics-sci6492-code/9a1f0159-20e9-42ac-b865-3ac624dd6256/scratchpad/render")

f = r3.File3dm.Read(SRC)
names = [l.Name for l in f.Layers]
manifest = {}

for o in f.Objects:
    g = o.Geometry
    if not isinstance(g, r3.Mesh):
        continue
    name = names[o.Attributes.LayerIndex]
    nv = len(g.Vertices)
    pos = np.empty((nv, 3), np.float32)
    for i in range(nv):
        p = g.Vertices[i]
        pos[i] = (p.X, p.Y, p.Z)

    tris = []
    for i in range(len(g.Faces)):
        fa = g.Faces[i]
        if len(fa) == 4 and fa[2] != fa[3]:
            tris.append((fa[0], fa[1], fa[2]))
            tris.append((fa[0], fa[2], fa[3]))
        else:
            tris.append((fa[0], fa[1], fa[2]))
    idx = np.asarray(tris, np.uint32)

    # smooth-ish vertex normals, area weighted
    nrm = np.zeros((nv, 3), np.float32)
    a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(nrm, idx[:, k], fn)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = (nrm / np.where(ln == 0, 1, ln)).astype(np.float32)

    (OUT / f"{name}_pos.bin").write_bytes(pos.tobytes())
    (OUT / f"{name}_nrm.bin").write_bytes(nrm.tobytes())
    (OUT / f"{name}_idx.bin").write_bytes(idx.tobytes())
    manifest[name] = {"verts": nv, "tris": int(len(idx))}
    print(name, nv, len(idx))

bb = pos  # last one; compute over sculpture separately below
f2 = r3.File3dm.Read(SRC)
allp = []
for o in f2.Objects:
    g = o.Geometry
    if isinstance(g, r3.Mesh) and names[o.Attributes.LayerIndex] == "sculpture":
        bbx = g.GetBoundingBox()
        manifest["bbox"] = {"min": [bbx.Min.X, bbx.Min.Y, bbx.Min.Z],
                            "max": [bbx.Max.X, bbx.Max.Y, bbx.Max.Z]}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(manifest["bbox"])
