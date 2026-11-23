// Loads the density fields produced by tools/extract_fields.py and turns
// them into samplers. Nothing here is procedural: every distribution in the
// piece is read out of the two reference photographs.

let F = null;

function cdfFrom(map) {
  // Cumulative distribution over the flattened map, for O(log n) sampling.
  const cdf = new Float32Array(map.length);
  let acc = 0;
  for (let i = 0; i < map.length; i++) {
    acc += map[i] / 255;
    cdf[i] = acc;
  }
  return { cdf, total: acc };
}

// Returns [u, v] in [0,1], cell-jittered so samples are not stuck on a lattice.
export function sampleField(field, rnd) {
  const { cdf, total, w, h } = field;
  const target = rnd() * total;
  let lo = 0, hi = cdf.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cdf[mid] < target) lo = mid + 1; else hi = mid;
  }
  const x = lo % w, y = (lo / w) | 0;
  return [(x + rnd()) / w, (y + rnd()) / h];
}

// Read the diffuse density back at a point. The drainage network uses it as
// the potential that rivers flow toward.
export function densityAt(u, v) {
  const d = F.galaxy.diffuse;
  const x = Math.min(d.w - 1, Math.max(0, (u * d.w) | 0));
  const y = Math.min(d.h - 1, Math.max(0, (v * d.h) | 0));
  return d.data[y * d.w + x] / 255;
}

// Inverse of galaxyLocal: chunk-local coords back to field coords.
export function galaxyUV(x, y) {
  const a = F.galaxy.aspect;
  return [x / (2 * a) + 0.5, 0.5 - y / 2];
}

export function sampleColor(u, v) {
  const c = F.galaxy.color;
  const x = Math.min(c.w - 1, Math.max(0, (u * c.w) | 0));
  const y = Math.min(c.h - 1, Math.max(0, (v * c.h) | 0));
  const i = (y * c.w + x) * 3;
  return [c.data[i] / 255, c.data[i + 1] / 255, c.data[i + 2] / 255];
}

async function bin(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('missing field ' + url + ' (run tools/extract_fields.py)');
  return new Uint8Array(await r.arrayBuffer());
}

export async function loadFields(base = '/') {
  const meta = await (await fetch(base + 'fields/fields.json')).json();
  const g = meta.galaxy, r = meta.river;

  const [diffuse, stars, color, mask, dist, spine] = await Promise.all([
    bin(base + g.diffuse.file),
    bin(base + g.stars.file),
    bin(base + g.color.file),
    bin(base + r.mask.file),
    bin(base + r.dist.file),
    (await fetch(base + r.spine.file)).json(),
  ]);

  // The galaxy reference is 1:2, so the long axis spans the chunk's local
  // [-1,1] and the short axis keeps its true proportion.
  const gAspect = g.w / g.h;

  F = {
    galaxy: {
      w: g.w, h: g.h, aspect: gAspect,
      diffuse: { ...cdfFrom(diffuse), data: diffuse, w: g.w, h: g.h },
      stars:   { ...cdfFrom(stars),   data: stars,   w: g.w, h: g.h },
      color:   { data: color, w: g.color.w, h: g.color.h },
    },
    river: {
      w: r.w, h: r.h,
      mask, dist,
      spine,
      sinuosity: r.sinuosity,
      curvRms: r.curvRms,
      meanHalfWidth: r.meanHalfWidth,
    },
    meta,
  };
  return F;
}

export function fields() {
  if (!F) throw new Error('fields not loaded');
  return F;
}

// Map a field sample in [0,1]^2 to the galaxy chunk's local coords,
// preserving the photograph's aspect. v is flipped: image y runs down.
export function galaxyLocal(u, v) {
  const a = F.galaxy.aspect;
  return [(u - 0.5) * 2 * a, -(v - 0.5) * 2];
}
