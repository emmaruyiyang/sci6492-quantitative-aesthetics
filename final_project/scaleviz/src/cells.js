// P4 - the cell sprite atlas.
//
// Up to P3 a cell was procedural: a membrane, some cytoplasm, a nucleus,
// drawn by the fragment shader. From here a cell is a PHOTOGRAPH - one cell
// cut out of one of the 500 micrographs in sample/topography/cell by
// tools/extract_cells.py, which also decides where its edge is (the alpha
// channel) and evens the exposure out across the set.
//
// One atlas, one texture bind. The variety you see at depth comes from
// 100 tiles x 8 free orientations x per-point size and brightness jitter.

import * as THREE from 'three';
import { asset } from './assets.js';

let ATLAS = null;
let NORMALS = null;
let META = null;

export async function loadCellAtlas(base = '/') {
  const meta = await fetch(asset(base, 'cells/cells.json')).then((r) => r.json());
  const tex = await new THREE.TextureLoader().loadAsync(asset(base, 'cells/cell_atlas.png'));
  // NOT tagged sRGB. These are raw ShaderMaterials with no
  // <colorspace_fragment>, so nothing re-encodes on the way out: tagging the
  // texture sRGB decodes it and never encodes it back, and the whole cell
  // level comes out a gamma too dark. Every other colour in the piece is
  // authored in output space, so this one is too.
  tex.colorSpace = THREE.NoColorSpace;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  ATLAS = tex;
  META = meta;

  // The matching normal map, baked from the 500 depth images next to the
  // photographs. This one is DATA, not colour: no colour space, and it must
  // not be stretched or graded on the way in.
  if (meta.normals) {
    const nrm = await new THREE.TextureLoader().loadAsync(asset(base, 'cells/' + meta.normals));
    nrm.colorSpace = THREE.NoColorSpace;
    nrm.minFilter = THREE.LinearMipmapLinearFilter;
    nrm.magFilter = THREE.LinearFilter;
    nrm.generateMipmaps = true;
    nrm.wrapS = nrm.wrapT = THREE.ClampToEdgeWrapping;
    nrm.needsUpdate = true;
    NORMALS = nrm;
  }
  return { tex, meta, normals: NORMALS };
}

export function cellAtlas() { return ATLAS; }
export function cellNormals() { return NORMALS; }
export function cellMeta() { return META; }
// Content has to know the tile count to pick one, but must not import three.
export function cellTileCount() { return META ? META.count : 1; }
export function cellTileMeans() { return META ? META.means : null; }

// Mean luminance of the atlas, alpha-weighted (the per-tile means already
// are). The river draws cell silhouettes in WATER colour, not in the
// photograph's own colour: shape and internal structure come from the
// photograph, hue from the river. Dividing by this mean is what keeps that
// substitution exposure-neutral.
let MEAN_LUM = null;
export function cellMeanLum() {
  if (MEAN_LUM != null) return MEAN_LUM;
  const M = META && META.means;
  if (!M || !M.length) return (MEAN_LUM = 1);
  let s = 0;
  for (const m of M) s += 0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2];
  return (MEAN_LUM = s / M.length);
}
