// The 3-level hierarchy. Every distribution here is driven by a field
// extracted from one of the two reference photographs (see tools/extract_fields.py):
//
//   galaxy  point density  = low-pass luminance of galaxy_ref.jpg
//           star density   = high-pass luminance of the same
//           point colour   = blurred colour of the same
//   river   channel shape  = the curvature signature of the centreline
//                            traced out of river_ref.jpg
//           channel width  = that centreline's half-width profile
//           water colour   = colour sampled along the real channel
//   cell    packing here; the cell ITSELF is a photograph, cut out of
//           sample/topography/cell by tools/extract_cells.py (see cells.js)
//
// CHUNK CONTRACT
//   key / origin(absolute float64) / unit / build() -> {pos,size,color,aux}
//   pos in local [-1,1]; size in local units; aux is per-mark (see cascade.js)

import { hashStr, mulberry32, gaussian, makeNoise1D } from './rng.js';
import { fields, sampleField, sampleColor, galaxyLocal, densityAt, galaxyUV } from './fields.js';
import { cellTileCount, cellTileMeans } from './cells.js';

// P7 - the step between cell and river.
//
// 100x per level made the hierarchy REAL but not VISIBLE. At that ratio one
// river edge is ~70 cells long and ~2,000-8,000 cells in area, so a legible
// stretch of drainage network holds about 700,000 of them: far past anything
// that can be drawn, which is why the two levels were never on screen at the
// same time and the piece read as three separate pictures.
//
// 10^1.5 instead. An edge is now ~22 cells long, a node owns ~300 cell
// chunks instead of ~3,000, and the view that holds 60 river nodes holds
// about 18,000 cell chunks - under the impostor cap, so at that zoom every
// cell in the picture is a real one.
//
// The river -> galaxy step stays at 100x. That relationship was already
// visible without drawing it: spacing follows the galaxy's density with a
// measured exponent of 0.93 and the colour chain correlates +0.88, so it
// composes to the eye already. Only the cell -> river step needed closing.
//
// Cost: the whole range goes from 4.45 decades to 3.95.
const STEP = Math.pow(10, 1.5);        // 31.6228
export const LEVELS = [
  { index: 0, name: 'cell',   unit: 1 },
  { index: 1, name: 'river',  unit: STEP },
  { index: 2, name: 'galaxy', unit: STEP * 100 },
];

// How far a chunk's content reaches, in local units. A river chunk owns half
// of each incident edge, so it spills past the [-1,1] box its node sits in;
// culling, impostor sprite size and the bake camera all have to know that.
export const LEVEL_EXTENT = [1.14, 1.52, 1.0];

export const GALAXY_UNIT = LEVELS[2].unit;
export const RIVER_UNIT  = LEVELS[1].unit;
export const CELL_UNIT   = LEVELS[0].unit;

const N_GALAXY_GLOW   = 78000;
const N_GALAXY_STARS  = 22000;
const N_RIVER_CHUNKS  = 1600;

// ADAPTIVE spacing. A fixed minimum distance makes the galaxy decide only
// WHERE rivers are, never HOW MANY: once a region is packed to the limit,
// extra brightness buys nothing. Measured, that compressed a 250x range in
// galaxy density into a 19x range in river density - areal density went as
// density^0.36 where a composed hierarchy wants density^1.0.
//
// Poisson-disk radius r ~ d^(-1/2) gives points per area ~ 1/r^2 ~ d, which
// is the exponent we want. The clamps are what the next scale down can
// actually use: below R_MIN the 290-wide chunks stack into a sheet again,
// above R_MAX nothing is close enough to connect to.
// In RIVER UNITS, not absolute: these are statements about how the level
// packs, and they must not change meaning when the step between scales does.
const RIVER_R_MIN = 0.70 * RIVER_UNIT;
const RIVER_R_MAX = 2.90 * RIVER_UNIT;   // <= EDGE_MAX, or the sparsest nodes cannot link
const RIVER_R_K   = 1.68 * Math.sqrt(0.10) * RIVER_UNIT;
const RIVER_R_CELL = 1.00 * RIVER_UNIT;  // hash cell; >= R_MIN
function riverSpacing(d) {
  const r = RIVER_R_K / Math.sqrt(Math.max(d, 1e-4));
  return Math.min(RIVER_R_MAX, Math.max(RIVER_R_MIN, r));
}
const N_CELL_POINTS   = 2400;

/* ------------------------------------------------------------------ */
/* L2 — galaxy, sampled from the photograph's luminance                */
/* ------------------------------------------------------------------ */

function buildGalaxy() {
  const F = fields().galaxy;
  const rnd = mulberry32(hashStr('galaxy'));
  const n = N_GALAXY_GLOW + N_GALAXY_STARS;
  const pos = new Float32Array(n * 3);
  const size = new Float32Array(n);
  const color = new Float32Array(n * 3);
  const aux = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const isStar = i >= N_GALAXY_GLOW;
    const [u, v] = sampleField(isStar ? F.stars : F.diffuse, rnd);
    const [x, y] = galaxyLocal(u, v);
    const c = sampleColor(u, v);

    pos[i * 3] = x;
    pos[i * 3 + 1] = y;
    // thin disc: the band is flat, the scatter only gives P3 something to
    // parallax against later
    pos[i * 3 + 2] = gaussian(rnd) * 0.012;

    if (isStar) {
      const b = 0.55 + 0.45 * rnd();
      size[i] = 0.0016 + 0.0026 * rnd() * rnd();
      // stars read cooler than the dust they sit in
      color[i * 3]     = Math.min(1, c[0] * 0.55 + 0.52) * b;
      color[i * 3 + 1] = Math.min(1, c[1] * 0.55 + 0.56) * b;
      color[i * 3 + 2] = Math.min(1, c[2] * 0.55 + 0.68) * b;
      aux[i] = 1;
    } else {
      // large, overlapping, faint: the band has to read as light, not dots
      size[i] = 0.018 + 0.046 * Math.pow(rnd(), 1.6);
      color[i * 3]     = c[0];
      color[i * 3 + 1] = c[1];
      color[i * 3 + 2] = c[2];
      aux[i] = 0;
    }
  }
  return { pos, size, color, aux };
}

// River chunks sit where the galaxy's light is: the same density field that
// draws the band also decides where the next scale down exists at all.
let riverOrigins = null;
let riverRadius = null;
let riverRecycled = 0;

// How many times to take the nodes that landed somewhere they cannot connect
// and spend them again. Each pass strictly shrinks the isolated set, so this
// is a cap, not a target.
const RECYCLE_PASSES = 8;

export function getRiverOrigins() {
  if (riverOrigins) return riverOrigins;
  const F = fields().galaxy;
  const rnd = mulberry32(hashStr('river-origins'));

  // Density decides where rivers are likely; blue-noise rejection decides
  // that they do not land on top of each other. Plain density sampling gave
  // a median spacing of 78 absolute units for chunks 200 wide, so every
  // chunk was stacked 2-3 deep and the level rendered as a solid sheet.
  //
  // The rejection radius is now per candidate (see riverSpacing), so the
  // packing itself carries the galaxy's density instead of flattening it.
  const key = (gx, gy) => gx * 73856093 ^ gy * 19349663;
  const SPACE_RINGS = Math.ceil(RIVER_R_MAX / RIVER_R_CELL);
  let grid = new Map();
  let acc = [];                      // entries are [x, y, z, r]

  function put(p) {
    const k = key(Math.floor(p[0] / RIVER_R_CELL), Math.floor(p[1] / RIVER_R_CELL));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(p);
  }
  function rebuildGrid() {
    grid = new Map();
    for (const p of acc) put(p);
  }

  function spacingOK(x, y, r) {
    const gx = Math.floor(x / RIVER_R_CELL), gy = Math.floor(y / RIVER_R_CELL);
    for (let dx = -SPACE_RINGS; dx <= SPACE_RINGS; dx++) {
      for (let dy = -SPACE_RINGS; dy <= SPACE_RINGS; dy++) {
        const bucket = grid.get(key(gx + dx, gy + dy));
        if (!bucket) continue;
        for (const p of bucket) {
          // the larger of the two radii wins, so a sparse node is not
          // crowded by a dense one that happens to be sampled later
          const m = Math.max(r, p[3]);
          if ((p[0] - x) ** 2 + (p[1] - y) ** 2 < m * m) return false;
        }
      }
    }
    return true;
  }

  // A node ends up rendering nothing exactly when no other node lies within
  // EDGE_MAX of it: with a neighbour in range it either flows to one that is
  // brighter, or a dimmer one flows into it, or - being a one-node basin -
  // depression filling links it across components. So connectivity can be
  // tested here, during sampling, without building the graph first.
  const LINK_RINGS = Math.ceil(EDGE_MAX / RIVER_R_CELL);
  function hasNeighbour(x, y, self) {
    const gx = Math.floor(x / RIVER_R_CELL), gy = Math.floor(y / RIVER_R_CELL);
    for (let dx = -LINK_RINGS; dx <= LINK_RINGS; dx++) {
      for (let dy = -LINK_RINGS; dy <= LINK_RINGS; dy++) {
        const bucket = grid.get(key(gx + dx, gy + dy));
        if (!bucket) continue;
        for (const p of bucket) {
          if (p === self) continue;
          if ((p[0] - x) ** 2 + (p[1] - y) ** 2 <= EDGE_MAX2) return true;
        }
      }
    }
    return false;
  }

  function sample(budget, requireLink) {
    for (let attempt = 0; attempt < budget && acc.length < N_RIVER_CHUNKS; attempt++) {
      const [u, v] = sampleField(F.diffuse, rnd);
      const [lx, ly] = galaxyLocal(u, v);
      const x = lx * GALAXY_UNIT, y = ly * GALAXY_UNIT;
      const r = riverSpacing(densityAt(u, v));
      if (!spacingOK(x, y, r)) continue;
      // On a recycle pass a node must join the network it is being spent on.
      // Without this the sampler keeps putting them back in the same empty
      // country the CDF sent them to the first time, and the passes just
      // churn. With it, the freed budget goes where the spacing rule still
      // has room - which, the rule being proportional to density, is the
      // bright regions.
      if (requireLink && acc.length && !hasNeighbour(x, y, null)) continue;
      const p = [x, y, gaussian(rnd) * 0.012 * GALAXY_UNIT, r];
      acc.push(p);
      put(p);
    }
  }

  sample(240000, false);
  for (let pass = 0; pass < RECYCLE_PASSES; pass++) {
    const keep = acc.filter((p) => hasNeighbour(p[0], p[1], p));
    const lost = acc.length - keep.length;
    if (!lost) break;
    riverRecycled += lost;
    acc = keep;
    rebuildGrid();
    sample(120000, true);
  }

  const out = new Float64Array(acc.length * 3);
  const rad = new Float32Array(acc.length);
  for (let i = 0; i < acc.length; i++) {
    out[i * 3] = acc[i][0]; out[i * 3 + 1] = acc[i][1]; out[i * 3 + 2] = acc[i][2];
    rad[i] = acc[i][3];
  }
  riverRadius = rad;
  riverOrigins = out;
  return out;
}

export function riverRecycleCount() { getRiverOrigins(); return riverRecycled; }

// Each node's own Poisson radius, i.e. how far apart rivers are HERE.
export function getRiverRadius() { getRiverOrigins(); return riverRadius; }

/* ------------------------------------------------------------------ */
/* The warm-cool axis: one number per place, shared by both levels      */
/* ------------------------------------------------------------------ */

// The galaxy photograph decides WHERE the next scale down exists, but until
// now it had nothing to say about what that scale LOOKS like: the river took
// its colour from the river photograph and the tissue picked a hue at
// random, so the hue correlation between a place in the galaxy and the cells
// at that place measured -0.075. Every scale was independently plausible and
// none of them was made of the one above.
//
// warmth(node) is the one scalar that joins them: where the node sits on the
// galaxy's own warm-cool axis, percentile-stretched so the photograph's
// narrow spread (about 3 radians of hue) fills the range the levels below
// can actually use. Cells choose their tissue with it; rivers rotate their
// water toward it.
let warmSorted = null;
let warmRaw = null;
function warmthOf(x, y) {
  const [u, v] = galaxyUV(x / GALAXY_UNIT, y / GALAXY_UNIT);
  const c = sampleColor(u, v);
  return (c[0] - c[2]) / Math.max(c[0] + c[2], 1e-3);
}
function warmthTable() {
  if (warmSorted) return warmSorted;
  const ro = getRiverOrigins();
  const n = ro.length / 3;
  warmRaw = new Float32Array(n);
  for (let i = 0; i < n; i++) warmRaw[i] = warmthOf(ro[i * 3], ro[i * 3 + 1]);
  warmSorted = Float32Array.from(warmRaw).sort();
  return warmSorted;
}

// 0 = the coolest place in the galaxy, 1 = the warmest. Percentile rank, so
// the output is uniform however the photograph's colours happen to sit.
export function nodeWarmth(node) {
  const srt = warmthTable();
  const n = srt.length;
  if (node >= n || node < 0) {
    // the impostor bake variants: spread evenly, so no real tissue is far
    // from the variant standing in for it
    return (((node % 101) + 0.5) / 101);
  }
  const w = warmRaw[node];
  let lo = 0, hi = n - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (srt[mid] < w) lo = mid + 1; else hi = mid; }
  return lo / Math.max(1, n - 1);
}

// Luminance-preserving hue rotation (the feColorMatrix one).
export function hueRotateRGB(c, a) {
  const co = Math.cos(a), si = Math.sin(a);
  const r = c[0], g = c[1], b = c[2];
  return [
    (0.213 + co * 0.787 - si * 0.213) * r + (0.715 - co * 0.715 - si * 0.715) * g + (0.072 - co * 0.072 + si * 0.928) * b,
    (0.213 - co * 0.213 + si * 0.143) * r + (0.715 + co * 0.285 + si * 0.140) * g + (0.072 - co * 0.072 - si * 0.283) * b,
    (0.213 - co * 0.213 - si * 0.787) * r + (0.715 - co * 0.715 + si * 0.715) * g + (0.072 + co * 0.928 + si * 0.072) * b,
  ];
}

export function riverCount() { return getRiverOrigins().length / 3; }

/* ------------------------------------------------------------------ */
/* L1 — river, a drainage network spanning the whole galaxy             */
/* ------------------------------------------------------------------ */
//
// P2. Before this, every river chunk grew its own private meander and
// neighbouring chunks knew nothing about each other, so the level was 1,600
// unrelated squiggles rather than a network.
//
// Now the river origins are the NODES of one global drainage graph. Each node
// flows to the nearest neighbour with higher density -- the same galaxy field
// that placed them -- which gives a forest of trees converging on the bright
// parts of the band, the way real catchments converge on an outlet. Strictly
// increasing potential along every link means the graph cannot contain a
// cycle.
//
// The chunking problem: an edge spans two nodes, so it belongs to neither
// chunk. The fix is HALF-EDGES. Both endpoints generate the identical edge
// geometry from a seed derived from the sorted node pair, and each draws only
// the half nearer to itself. Every edge is therefore drawn exactly once, the
// two halves meet at the midpoint to the last bit, and a chunk still builds
// independently from nothing but its own id.

const EDGE_MAX = 3.00 * RIVER_UNIT;   // longer links are left unconnected
const EDGE_MAX2 = EDGE_MAX * EDGE_MAX;
const EDGE_STEPS = 128;
// 0.064 was calibrated against a river whose marks were abstract streaks
// scaled to the channel: at that width the median channel is 2.7 cells
// across, which shows cells but does not show MANY cells making one river.
// Widened until the median channel is ~4 cells and the widest trunk ~26,
// which is as far as the cell level's per-frame budget goes.
const WIDTH_K = 0.125 * RIVER_UNIT;   // half-width of a headwater
// Hydraulic scaling with drainage area. Real channels run roughly 0.3-0.5;
// 0.35 was the first pick and it is still inside that range, but once the
// mark became a cell the exponent stopped being only a hydraulics claim and
// started setting how MANY cells wide the median channel is. At 0.35 the
// headwaters were 2.6 cells across - a string of beads, not a river - and
// widening the whole law to fix that inflated the trunks until the dense
// core merged into one mass. 0.30 lifts the bottom and leaves the top where
// it was.
const WIDTH_EXP = 0.30;
// How many times over each bit of channel gets painted. Point COUNT alone is
// the wrong control: a mark's size scales with the channel's width too, so
// counting by area makes coverage grow as width^2 and wide trunks clip the
// framebuffer to white while their impostors stay dim.
// Measured, not derived: the edge taper and size jitter put the real coverage
// at roughly half this number, which is where the channel reads as water.
const TARGET_COVER = 22;
const MARK_FRAC = 0.20;      // mark radius, at the reference half-width
const MARK_ELONG = 2.6;      // must match ELONG in cascade.js

// A mark is the TEXTURE OF WATER, not a scaled copy of the channel. Tying
// its radius straight to the half-width (MARK_FRAC * hw) meant a trunk 13x
// wider than a headwater got marks 13x bigger, and past about hw = 20 they
// overlap into a smooth fog with no channel left inside it. That is what the
// widest nodes in the galaxy's core actually looked like: not many channels
// merging, which is what I assumed, but single channels dissolving.
//
// Sub-linear instead. A big river's surface is not 13x coarser than a
// creek's; if anything the two look alike close up. The point budget follows
// automatically, because it is computed from the mark area (see buildRiver),
// so coverage stays at TARGET_COVER either way.
const MARK_HW_REF = 0.08 * RIVER_UNIT;   // roughly the median half-width
const MARK_EXP = 0.45;

// P7 - the river is drawn out of cells.
//
// Three levels that never share the screen are three pictures, not a
// cascade. Measured: the cell level is faded out by u=2.0, where only 3
// river nodes are in view; the drainage network does not read as a network
// until u~2.8, where the view holds 703,269 cell chunks against an impostor
// cap of 72,000. The gap is 1.2 decades wide and no budget closes it.
//
// So the river's own mark stops being an abstract streak and becomes a cell.
// Nothing about the level's structure changes - same chunks, same draw call,
// same enumeration - but every mid-zoom frame now shows what the water is
// made of.
export const RIVER_CELLS = true;
const TAU = Math.PI * 2;

// A cell is a cell. The sub-linear law below is right for a streak, which is
// a texture of water and may coarsen with the channel; it is wrong for an
// object with a real size. At RIVER_CELLS the mark is therefore CONSTANT in
// absolute units, and set to one cell chunk's own extent (LEVEL_EXTENT[0]),
// so a mark here and a chunk one level down are the same thing at the same
// size.
const CELL_MARK_R = 1.10;
// Measured off public/cells/cell_atlas.png, not guessed. A baked cell spans
// 0.907 of its tile's width and inks 0.412 of its area; the shader samples
// RCELL_FIT wider so a free rotation cannot clip it (see cascade.js), which
// shrinks both figures by that factor. Getting this wrong is what made the
// first attempt look like dust: the sprite was sized as if the cell filled
// it, so the cell came out 2.2x too small and the channel 2.2x too thin.
const CELL_FIT = 1.281;
const CELL_SCREEN_FRAC = 0.907 / CELL_FIT;        // 0.708
const CELL_INK_FRAC = 0.412 / (CELL_FIT * CELL_FIT);   // 0.251
// local sprite side that puts a cell of radius CELL_MARK_R on screen
const CELL_SPRITE = 2 * CELL_MARK_R / CELL_SCREEN_FRAC;
const CELL_INK_AREA = CELL_INK_FRAC * CELL_SPRITE * CELL_SPRITE;
// Matter, not light: eleven layers of source-over composite to a flat sheet,
// so the channel is painted about twice over instead of eleven times.
const TARGET_COVER_CELL = 2.2;
// Cells follow the current, but not to the degree: a perfectly combed field
// reads as corduroy, not as water.
const CELL_MARK_JITTER = 0.9;   // radians, full width
const RIVER_POINT_CAP = 20000;

export function markRadius(hw) {
  if (RIVER_CELLS) return CELL_MARK_R;
  return MARK_FRAC * MARK_HW_REF * Math.pow(Math.max(hw, 0.5) / MARK_HW_REF, MARK_EXP);
}

let riverGraph = null;

export function getRiverGraph() {
  if (riverGraph) return riverGraph;
  const ro = getRiverOrigins();
  const n = ro.length / 3;

  // Potential = the galaxy's own luminance, smoothed. Sampling it raw leaves
  // a local maximum every few nodes, which fragments the network into
  // hundreds of two-node basins.
  const pot = new Float32Array(n);
  const TAPS = [[0, 0], [0.006, 0], [-0.006, 0], [0, 0.006], [0, -0.006],
                [0.012, 0], [-0.012, 0], [0, 0.012], [0, -0.012]];
  for (let i = 0; i < n; i++) {
    const [u, v] = galaxyUV(ro[i * 3] / GALAXY_UNIT, ro[i * 3 + 1] / GALAXY_UNIT);
    let acc = 0;
    for (const [du, dv] of TAPS) acc += densityAt(u + du, v + dv);
    pot[i] = acc / TAPS.length;
  }

  // spatial hash so the neighbour search is not O(n^2)
  const cell = EDGE_MAX;
  const grid = new Map();
  const gkey = (gx, gy) => gx * 73856093 ^ gy * 19349663;
  for (let i = 0; i < n; i++) {
    const k = gkey(Math.floor(ro[i * 3] / cell), Math.floor(ro[i * 3 + 1] / cell));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  }

  // EDGE_MAX stays a hard cap, and so does the spacing. A node's half-edge
  // is what LEVEL_EXTENT[1] promises its chunk reaches, and culling, the
  // impostor sprite size and the bake camera all believe that promise - so
  // letting sparse nodes reach further to stay connected would silently
  // break three other things. Nodes with no neighbour inside EDGE_MAX stay
  // inactive instead, which is its own honest answer: out there, no river.
  const RINGS = 1;

  // parent = nearest neighbour that is brighter. Ties break on index so the
  // graph is identical on every run.
  const parent = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const x = ro[i * 3], y = ro[i * 3 + 1];
    const gx = Math.floor(x / cell), gy = Math.floor(y / cell);
    let best = -1, bestD2 = Infinity;
    for (let dx = -RINGS; dx <= RINGS; dx++) {
      for (let dy = -RINGS; dy <= RINGS; dy++) {
        const b = grid.get(gkey(gx + dx, gy + dy));
        if (!b) continue;
        for (const j of b) {
          if (j === i) continue;
          if (pot[j] < pot[i] || (pot[j] === pot[i] && j <= i)) continue;
          const d2 = (ro[j * 3] - x) ** 2 + (ro[j * 3 + 1] - y) ** 2;
          if (d2 < bestD2 && d2 <= EDGE_MAX2) { bestD2 = d2; best = j; }
        }
      }
    }
    parent[i] = best;
  }

  // Depression filling. "Flow to the brightest neighbour" leaves one basin
  // per local maximum -- 355 of them, and 142 nodes with no link at all,
  // which rendered as empty chunks. Join each remaining basin to the nearest
  // node in a DIFFERENT basin. Linking only across components is what makes
  // a cycle impossible, so the potential rule can be dropped here.
  const uf = new Int32Array(n);
  for (let i = 0; i < n; i++) uf[i] = i;
  const find = (x) => { while (uf[x] !== x) x = uf[x] = uf[uf[x]]; return x; };
  const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) uf[ra] = rb; };
  for (let i = 0; i < n; i++) if (parent[i] >= 0) union(i, parent[i]);

  const roots = [];
  for (let i = 0; i < n; i++) if (parent[i] < 0) roots.push(i);
  roots.sort((a, b) => pot[a] - pot[b]);
  let merged = 0;
  for (const r of roots) {
    const x = ro[r * 3], y = ro[r * 3 + 1];
    const gx = Math.floor(x / cell), gy = Math.floor(y / cell);
    let best = -1, bestD2 = Infinity;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const b = grid.get(gkey(gx + dx, gy + dy));
        if (!b) continue;
        for (const j of b) {
          if (j === r || find(j) === find(r)) continue;
          const d2 = (ro[j * 3] - x) ** 2 + (ro[j * 3 + 1] - y) ** 2;
          if (d2 < bestD2 && d2 <= EDGE_MAX2) { bestD2 = d2; best = j; }
        }
      }
    }
    if (best >= 0) { parent[r] = best; union(r, best); merged++; }
  }

  const children = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) if (parent[i] >= 0) children[parent[i]].push(i);

  // Drainage area by post-order traversal. The merge step can link a basin
  // to a dimmer node, so potential is no longer a valid processing order.
  const area = new Float32Array(n).fill(1);
  const stack = [];
  for (let i = 0; i < n; i++) if (parent[i] < 0) stack.push(i);
  const post = [];
  while (stack.length) {
    const v = stack.pop();
    post.push(v);
    for (const c of children[v]) stack.push(c);
  }
  for (let k = post.length - 1; k >= 0; k--) {
    const v = post[k];
    if (parent[v] >= 0) area[parent[v]] += area[v];
  }

  const hw = new Float32Array(n);
  for (let i = 0; i < n; i++) hw[i] = WIDTH_K * Math.pow(area[i], WIDTH_EXP);

  let nRoots = 0, maxArea = 0, isolated = 0;
  for (let i = 0; i < n; i++) {
    if (parent[i] < 0) nRoots++;
    if (parent[i] < 0 && children[i].length === 0) isolated++;
    if (area[i] > maxArea) maxArea = area[i];
  }

  // A node with no link has nothing to draw. 122 of them survive the merge
  // because no other node lies within EDGE_MAX of them at all; they are
  // marked inactive rather than rendered as empty chunks.
  const active = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    active[i] = (parent[i] >= 0 || children[i].length > 0) ? 1 : 0;
  }

  riverGraph = { n, pos: ro, pot, parent, children, area, hw, active,
                 roots: nRoots, maxArea, merged, isolated };
  return riverGraph;
}

// Absolute polyline of the channel between nodes a and b. Deterministic in
// the SORTED pair, so both endpoints compute the identical curve and their
// halves join exactly.
const edgeCache = new Map();
export function edgePath(a, b) {
  const lo = Math.min(a, b), hi = Math.max(a, b);
  const key = lo + ':' + hi;
  let got = edgeCache.get(key);
  if (got) return got;

  const g = getRiverGraph();
  const sp = fields().river.spine;
  const rnd = mulberry32(hashStr('edge:' + key));
  const noise = makeNoise1D(rnd, 4);
  const mirror = rnd() < 0.5 ? -1 : 1;
  const gain = 0.8 + rnd() * 0.4;
  const N = sp.n;
  const start = (rnd() * N) | 0;

  // meander driven by the reference river's own curvature
  let heading = 0, x = 0, y = 0;
  const px = new Float64Array(EDGE_STEPS), py = new Float64Array(EDGE_STEPS);
  for (let i = 0; i < EDGE_STEPS; i++) {
    heading += sp.curvature[(start + i) % N] * mirror * gain + noise(i / EDGE_STEPS * 3.1) * 0.012;
    x += Math.cos(heading); y += Math.sin(heading);
    px[i] = x; py[i] = y;
  }

  // Pin the ends to the two nodes with a similarity transform. Sinuosity is
  // a ratio, so it survives untouched: the channel between two nodes 200
  // apart comes out about 1.69 x 200 long, like the traced river.
  const cx = px[EDGE_STEPS - 1] - px[0], cy = py[EDGE_STEPS - 1] - py[0];
  const chord = Math.hypot(cx, cy) || 1;
  const ax = g.pos[lo * 3], ay = g.pos[lo * 3 + 1], az = g.pos[lo * 3 + 2];
  const bx = g.pos[hi * 3], by = g.pos[hi * 3 + 1], bz = g.pos[hi * 3 + 2];
  const tx = bx - ax, ty = by - ay;
  const target = Math.hypot(tx, ty) || 1;
  const sc = target / chord;
  const rot = Math.atan2(ty, tx) - Math.atan2(cy, cx);
  const cr = Math.cos(rot) * sc, sr = Math.sin(rot) * sc;

  const out = new Float64Array(EDGE_STEPS * 3);
  const head = new Float32Array(EDGE_STEPS);
  for (let i = 0; i < EDGE_STEPS; i++) {
    const rx = px[i] - px[0], ry = py[i] - py[0];
    out[i * 3]     = ax + rx * cr - ry * sr;
    out[i * 3 + 1] = ay + rx * sr + ry * cr;
    out[i * 3 + 2] = az + (bz - az) * (i / (EDGE_STEPS - 1));
  }
  for (let i = 0; i < EDGE_STEPS; i++) {
    const j = Math.min(EDGE_STEPS - 1, i + 1), k = Math.max(0, i - 1);
    head[i] = Math.atan2(out[j * 3 + 1] - out[k * 3 + 1], out[j * 3] - out[k * 3]);
  }

  got = { lo, hi, pts: out, head, len: target };
  edgeCache.set(key, got);
  if (edgeCache.size > 4000) edgeCache.delete(edgeCache.keys().next().value);
  return got;
}

// The half-edges a node is responsible for: its link to its parent and the
// links from each of its children, each covered from the node to the midpoint.
export function nodeHalfEdges(i) {
  const g = getRiverGraph();
  const out = [];
  if (g.parent[i] >= 0) out.push(g.parent[i]);
  for (const c of g.children[i]) out.push(c);
  return out;
}

// How far the water may be turned off the river photograph's own colour,
// toward the galaxy's colour here. The channel keeps the photograph's
// shading and its along-channel variation; only the hue moves.
const RIVER_HUE_SWING = 1.15;   // radians, at the ends of the warm-cool axis

function buildRiver(i) {
  const g = getRiverGraph();
  const sp = fields().river.spine;
  // One rotation for the whole chunk: the water here belongs to this part of
  // the galaxy, so a warm arm runs warm rivers and a cool one runs cool.
  // Negative: feColorMatrix rotates R->G->B, and the water starts at teal,
  // so a POSITIVE rotation walks it toward blue. Warm has to come down the
  // hue circle, not up it. Measured with the sign the wrong way round, the
  // river's warmth correlated -0.90 with the galaxy's.
  const hueShift = -(nodeWarmth(i) - 0.5) * 2 * RIVER_HUE_SWING;
  const spColor = sp.color.map((c) => hueRotateRGB(c, hueShift));
  const pal = RIVER_CELLS ? cellPalette(i) : null;
  const rnd = mulberry32(hashStr('river-fill:' + i));
  const others = nodeHalfEdges(i);

  const ox = g.pos[i * 3], oy = g.pos[i * 3 + 1], oz = g.pos[i * 3 + 2];
  const half = (EDGE_STEPS >> 1);

  // Budget points for constant COVERAGE, not constant area density.
  const segs = [];
  let total = 0;
  for (const j of others) {
    const e = edgePath(i, j);
    const wA = g.hw[i], wB = g.hw[j];
    const hwMean = (wA + (wA + wB) * 0.5) * 0.5;
    const channelArea = (e.len * 0.5) * 2 * hwMean;
    const markArea = RIVER_CELLS ? CELL_INK_AREA
                   : Math.PI * markRadius(hwMean) ** 2 * MARK_ELONG;
    const cover = RIVER_CELLS ? TARGET_COVER_CELL : TARGET_COVER;
    const quota = Math.max(60, Math.min(RIVER_CELLS ? 9000 : 4000,
      Math.round(cover * channelArea / Math.max(markArea, 1e-6))));
    segs.push({ e, fromLo: e.lo === i, j, quota });
    total += quota;
  }
  total = Math.min(total, RIVER_CELLS ? RIVER_POINT_CAP : 9000);

  const pos = new Float32Array(total * 3);
  const size = new Float32Array(total);
  const color = new Float32Array(total * 3);
  const aux = new Float32Array(total);

  let w = 0;
  for (let si = 0; si < segs.length && w < total; si++) {
    const { e, fromLo, j } = segs[si];
    const quota = Math.min(segs[si].quota, total - w);
    const wA = g.hw[i], wB = g.hw[j];

    for (let q = 0; q < quota && w < total; q++, w++) {
      // t runs 0 (this node) -> 1 (midpoint of the edge)
      const t = rnd();
      const idx = fromLo ? t * half : (EDGE_STEPS - 1) - t * half;
      const i0 = Math.max(0, Math.min(EDGE_STEPS - 2, Math.floor(idx)));
      const f = idx - i0;
      const cx = e.pts[i0 * 3] + (e.pts[(i0 + 1) * 3] - e.pts[i0 * 3]) * f;
      const cy = e.pts[i0 * 3 + 1] + (e.pts[(i0 + 1) * 3 + 1] - e.pts[i0 * 3 + 1]) * f;
      const cz = e.pts[i0 * 3 + 2];
      const hd = e.head[i0];

      // width grows from this node toward the midpoint, toward the mean of
      // the two endpoint widths -- a confluence is wider than its branches
      const hwA = wA, hwM = (wA + wB) * 0.5;
      const hwHere = hwA + (hwM - hwA) * t;

      const nx = -Math.sin(hd), ny = Math.cos(hd);
      const s2 = rnd() * 2 - 1;
      const off = Math.sign(s2) * Math.pow(Math.abs(s2), 0.72) * hwHere;

      // store LOCAL to this chunk
      pos[w * 3]     = (cx + nx * off - ox) / RIVER_UNIT;
      pos[w * 3 + 1] = (cy + ny * off - oy) / RIVER_UNIT;
      pos[w * 3 + 2] = (cz - oz) / RIVER_UNIT + gaussian(rnd) * 0.003;

      const ci = Math.min(sp.n - 1, Math.floor(t * sp.n));
      const edge = Math.abs(off) / Math.max(hwHere, 1e-5);
      const shade = 0.72 + 0.45 * (1 - edge);
      color[w * 3]     = spColor[ci][0] * shade;
      color[w * 3 + 1] = spColor[ci][1] * shade;
      color[w * 3 + 2] = spColor[ci][2] * shade;

      size[w] = ((RIVER_CELLS ? CELL_SPRITE : markRadius(hwHere)) / RIVER_UNIT) *
                (RIVER_CELLS ? 1 : (1 - 0.55 * edge)) *
                (RIVER_CELLS ? 0.72 + 0.56 * rnd() : 0.55 + 0.9 * rnd()) + 0.0015;
      if (RIVER_CELLS) {
        // tile from THIS node's tissue palette, so the river and the cells
        // growing on it are made of the same photographs
        const tile = pal[(rnd() * pal.length) | 0];
        let fr = ((hd + (rnd() - 0.5) * CELL_MARK_JITTER) % TAU) / TAU;
        fr -= Math.floor(fr);
        aux[w] = tile + Math.min(0.99999, fr);
      } else {
        aux[w] = hd;
      }
    }
  }

  return {
    pos: pos.subarray(0, w * 3), size: size.subarray(0, w),
    color: color.subarray(0, w * 3), aux: aux.subarray(0, w),
  };
}

// Cell chunk origins, strung along the half-edges this node owns and scattered
// across the channel, in ABSOLUTE coords.
const cellOriginCache = new Map();
export function getCellOrigins(nodeId) {
  let got = cellOriginCache.get(nodeId);
  if (got) return got;
  const g = getRiverGraph();
  const rnd = mulberry32(hashStr('cell-origins:' + nodeId));
  const others = nodeHalfEdges(nodeId);
  const half = (EDGE_STEPS >> 1);

  const list = [];
  // Station spacing. A cell chunk is 2 units wide, but its soft edge makes
  // the effective width nearer 1.4, so 1.7 left visible stripes.
  // 0.62 was calibrated in P2 against the old procedural marks. P4's
  // photographed cells are 1.45x wider, so the lattice can be looser for
  // the same coverage - and it has to be: at 0.62 the chunks stacked 13 deep
  // and the deepest frame cost 14.3 ms of a 16.7 ms budget.
  const SP = 0.82;
  for (const j of others) {
    const e = edgePath(nodeId, j);
    const fromLo = (e.lo === nodeId);
    const wA = g.hw[nodeId], wM = (g.hw[nodeId] + g.hw[j]) * 0.5;
    const hwMean = (wA + wM) * 0.5;

    // A JITTERED LATTICE, not random points. Scattering stations at random
    // needs ~3.6x oversampling before the Poisson holes close; a lattice
    // covers the channel evenly at the same count, which is what stops the
    // deep view looking moth-eaten.
    let along = Math.max(2, Math.round((e.len * 0.5) / SP));
    let across = Math.max(1, Math.round(2 * hwMean / SP));
    while (along * across > 14000 && along > 2) { along = Math.max(2, (along * 0.8) | 0); across = Math.max(1, (across * 0.8) | 0); }

    for (let ka = 0; ka < along; ka++) {
      const t = (ka + 0.5 + (rnd() - 0.5) * 1.0) / along;
      const idx = fromLo ? t * half : (EDGE_STEPS - 1) - t * half;
      // Interpolate along the polyline. Snapping to the nearest vertex put a
      // floor of one segment length (~2 units) under the station spacing,
      // which is wider than a cell chunk, so the channel came out striped.
      const i0 = Math.max(0, Math.min(EDGE_STEPS - 2, Math.floor(idx)));
      const ff = idx - i0;
      const bx = e.pts[i0 * 3] + (e.pts[(i0 + 1) * 3] - e.pts[i0 * 3]) * ff;
      const by = e.pts[i0 * 3 + 1] + (e.pts[(i0 + 1) * 3 + 1] - e.pts[i0 * 3 + 1]) * ff;
      const bz = e.pts[i0 * 3 + 2];
      const hd = e.head[i0];
      const hwHere = wA + (wM - wA) * t;
      const nx = -Math.sin(hd), ny = Math.cos(hd);
      for (let kl = 0; kl < across; kl++) {
        const lat = ((kl + 0.5 + (rnd() - 0.5) * 1.0) / across - 0.5) * 2 * hwHere * 0.92;
        list.push([
          bx + nx * lat,
          by + ny * lat,
          bz + gaussian(rnd) * 0.4,
        ]);
      }
    }
  }

  const origins = new Float64Array(list.length * 3);
  for (let k = 0; k < list.length; k++) {
    origins[k * 3] = list[k][0]; origins[k * 3 + 1] = list[k][1]; origins[k * 3 + 2] = list[k][2];
  }
  got = { origins, count: list.length };
  cellOriginCache.set(nodeId, got);
  // each entry is now up to ~1200 stations, so the cache has to be smaller
  if (cellOriginCache.size > 48) cellOriginCache.delete(cellOriginCache.keys().next().value);
  return got;
}

/* ------------------------------------------------------------------ */
/* L0 — cell                                                           */
/* ------------------------------------------------------------------ */

// `node` is the river node the chunk hangs off. Every cell chunk in one
// river draws from the same handful of photographs, so a field of cells
// reads as ONE TISSUE rather than as a hundred unrelated organisms shuffled
// together. Picking a tile uniformly out of the atlas looked like confetti.
const PALETTE = 6;

// The tiles one river's tissue is made of. Kept separate from buildCell so
// the impostor path can ask the same question without building a chunk.
// The atlas runs the whole spectrum - algae green, haematoxylin purple,
// fluorescence magenta. Six tiles drawn uniformly out of that average to
// grey, so EVERY tissue averaged to the same grey and the whole cell level
// read as colourless haze wherever chunks overlapped. A tissue is one kind
// of cell, so a palette is a window on the hue circle, not a lottery.
let hueOrder = null;
function tilesByHue() {
  if (hueOrder) return hueOrder;
  const M = cellTileMeans();
  const n = M ? M.length : 1;
  const idx = [];
  for (let i = 0; i < n; i++) idx.push(i);
  let hue = new Array(n).fill(0);
  if (M) {
    hue = M.map(([r, g, b]) =>
      Math.atan2(Math.sqrt(3) * (g - b), 2 * r - g - b));
    idx.sort((a, b) => hue[a] - hue[b]);
  }
  hueOrder = { order: idx, hue };
  return hueOrder;
}

const HUE_WINDOW = 15;        // tiles, out of 100, that count as "same tissue"
const HUE_JITTER = 6;         // how far a tissue may drift off the galaxy's hue
const paletteCache = new Map();
export function cellPalette(node) {
  let p = paletteCache.get(node);
  if (p) return p;
  const { order, hue } = tilesByHue();
  const TILES = order.length;
  const prnd = mulberry32(hashStr('tissue:' + node));

  // WHICH window is no longer a lottery either: it is where this place sits
  // on the galaxy's warm-cool axis. Warm -> the red end of the atlas, cool ->
  // the cyan end. The jitter keeps neighbouring rivers from being identical
  // without letting them wander off the galaxy's colour.
  const target = (1 - nodeWarmth(node)) * Math.PI;
  let base = 0, bd = Infinity;
  for (let k = 0; k < TILES; k++) {
    let d = Math.abs(hue[order[k]] - target);
    if (d > Math.PI) d = 2 * Math.PI - d;
    if (d < bd) { bd = d; base = k; }
  }
  const jitter = ((prnd() - 0.5) * 2 * HUE_JITTER) | 0;
  const start = ((base - (HUE_WINDOW >> 1) + jitter) % TILES + TILES) % TILES;
  const seen = new Set();
  p = [];
  for (let guard = 0; p.length < Math.min(PALETTE, TILES) && guard < 200; guard++) {
    const t = order[(start + ((prnd() * HUE_WINDOW) | 0)) % TILES];
    if (seen.has(t)) continue;
    seen.add(t);
    p.push(t);
  }
  paletteCache.set(node, p);
  if (paletteCache.size > 4096) paletteCache.delete(paletteCache.keys().next().value);
  return p;
}

// Mean colour of a tissue, and of the whole atlas. An impostor is one of 12
// baked variants, so on its own it cannot be the colour of the particular
// tissue it stands in for - every chunk past the full-detail threshold would
// go the same average beige, and a node's colour would pop into existence as
// you zoomed in. The sprite carries paletteMean(node)/paletteMean(variant)
// as a tint instead, which costs three floats and makes the two paths agree.
let atlasMean = null;
export function paletteMean(node) {
  const M = cellTileMeans();
  if (!M) return [1, 1, 1];
  if (!atlasMean) {
    atlasMean = [0, 0, 0];
    for (const m of M) { atlasMean[0] += m[0]; atlasMean[1] += m[1]; atlasMean[2] += m[2]; }
    for (let k = 0; k < 3; k++) atlasMean[k] /= M.length;
  }
  const pal = cellPalette(node);
  const out = [0, 0, 0];
  for (const t of pal) { out[0] += M[t][0]; out[1] += M[t][1]; out[2] += M[t][2]; }
  // 85% of the cells come from the palette, 15% from anywhere in the atlas
  for (let k = 0; k < 3; k++) out[k] = 0.85 * (out[k] / pal.length) + 0.15 * atlasMean[k];
  return out;
}

// Out of the real node range on purpose, so a bake variant never borrows a
// real tissue's colour; nodeWarmth spreads these evenly along the axis.
const VARIANT_NODE = 100000;
// Memoised: this is called once per impostor per frame, tens of thousands of
// times, and handing the walk a fresh array each time made the collector the
// most expensive thing in the frame the last time it was measured.
const tintCache = new Map();
export function cellImpostorTint(node, v) {
  const k = node * 64 + v;
  let t = tintCache.get(k);
  if (t) return t;
  const a = paletteMean(node), b = paletteMean(VARIANT_NODE + v);
  t = [
    Math.min(2.6, Math.max(0.35, a[0] / Math.max(b[0], 1e-3))),
    Math.min(2.6, Math.max(0.35, a[1] / Math.max(b[1], 1e-3))),
    Math.min(2.6, Math.max(0.35, a[2] / Math.max(b[2], 1e-3))),
  ];
  tintCache.set(k, t);
  return t;
}

function buildCell(node, id) {
  const rnd = mulberry32(hashStr('cell:' + node + ':' + id));
  const n = N_CELL_POINTS;
  const pos = new Float32Array(n * 3);
  const size = new Float32Array(n);
  const color = new Float32Array(n * 3);
  const aux = new Float32Array(n);
  const TILES = Math.max(1, cellTileCount());
  const palette = cellPalette(node);

  const G = 44;
  const grid = new Float32Array(G * G).fill(-1);
  let i = 0, guard = 0;
  while (i < n && guard < n * 40) {
    guard++;
    const x = rnd() * 2 - 1;
    const y = rnd() * 2 - 1;
    // soft-edged square, not a disc: adjacent chunks have to blend into one
    // continuous tissue rather than read as separate pills
    const edge = 1 - Math.max(Math.abs(x), Math.abs(y));
    if (rnd() > Math.min(1, edge * 3.2)) continue;
    const gx = Math.min(G - 1, Math.floor((x + 1) * 0.5 * G));
    const gy = Math.min(G - 1, Math.floor((y + 1) * 0.5 * G));
    const gi = gy * G + gx;
    const big = rnd() < 0.42;
    if (big && grid[gi] > 0) continue;
    // Bigger than the procedural marks were. Those were additive glows that
    // filled the gaps between them for free; a photographed cell has an edge,
    // so the tissue only closes up if the discs actually overlap.
    const r = big ? 0.050 + 0.072 * rnd() : 0.014 + 0.023 * rnd();
    if (big) grid[gi] = r;

    pos[i * 3] = x;
    pos[i * 3 + 1] = y;
    pos[i * 3 + 2] = gaussian(rnd) * 0.03;
    size[i] = r;

    // The photograph carries the colour now, so `tint` is only exposure:
    // a little brightness scatter and a trace of warm/cool, enough that a
    // field of the same tile does not read as a stamped pattern.
    const b = 0.74 + 0.42 * rnd();
    const warm = (rnd() - 0.5) * 0.10;
    color[i * 3]     = b * (1 + warm);
    color[i * 3 + 1] = b;
    color[i * 3 + 2] = b * (1 - warm);

    // aux packs WHICH photograph and HOW it is turned: tile * 8 + orient,
    // exact in float32 well past the 100 tiles the atlas holds.
    // mostly this tissue's own cells, with the odd stranger so a chunk is
    // never just four stamps
    const tile = rnd() < 0.85
      ? palette[(rnd() * PALETTE) | 0]
      : Math.min(TILES - 1, (rnd() * TILES) | 0);
    const orient = (rnd() * 8) | 0;
    aux[i] = tile * 8 + orient;
    i++;
  }
  return {
    pos: pos.subarray(0, i * 3),
    size: size.subarray(0, i),
    color: color.subarray(0, i * 3),
    aux: aux.subarray(0, i),
  };
}

/* ------------------------------------------------------------------ */
/* Chunk queries                                                       */
/* ------------------------------------------------------------------ */

// Allocation-free iteration. The impostor path visits thousands of chunks per
// frame; handing it an object per chunk made the garbage collector the most
// expensive thing in the frame. a/b are integer ids: river index, cell index.
export function forEachChunk(level, F, halfX, halfY, cb) {
  if (level === 2) { cb(0, 0, 0, -1, -1); return; }

  const ro = getRiverOrigins();
  const nR = ro.length / 3;

  const g = getRiverGraph();

  if (level === 1) {
    const pad = RIVER_UNIT * LEVEL_EXTENT[1];
    for (let i = 0; i < nR; i++) {
      if (!g.active[i]) continue;
      const x = ro[i * 3], y = ro[i * 3 + 1];
      if (Math.abs(x - F[0]) > halfX + pad) continue;
      if (Math.abs(y - F[1]) > halfY + pad) continue;
      cb(x, y, ro[i * 3 + 2], i, -1);
    }
    return;
  }

  const rPad = RIVER_UNIT * LEVEL_EXTENT[1];
  const cPad = CELL_UNIT;
  for (let i = 0; i < nR; i++) {
    if (!g.active[i]) continue;
    const x = ro[i * 3], y = ro[i * 3 + 1];
    if (Math.abs(x - F[0]) > halfX + rPad) continue;
    if (Math.abs(y - F[1]) > halfY + rPad) continue;
    const { origins, count } = getCellOrigins(i);
    for (let j = 0; j < count; j++) {
      const cx = origins[j * 3], cy = origins[j * 3 + 1];
      if (Math.abs(cx - F[0]) > halfX + cPad) continue;
      if (Math.abs(cy - F[1]) > halfY + cPad) continue;
      cb(cx, cy, origins[j * 3 + 2], i, j);
    }
  }
}

export function chunkKey(level, a, b) {
  return level === 2 ? 'g' : level === 1 ? 'r' + a : 'c' + a + ':' + b;
}

export function buildChunk(level, a, b) {
  if (level === 2) return buildGalaxy();
  if (level === 1) return buildRiver(a);
  return buildCell(a, b);
}

// Representative chunks for the impostor atlas. Fixed seeds, so the bake is
// deterministic and the same picture every run.
// Active nodes ordered by drainage area, and each node's place in that
// order. A river chunk's brightness follows its channel area, so this is the
// axis the impostor variants have to span and the axis a chunk has to be
// matched on.
let areaRank = null;
function riverAreaRank() {
  if (areaRank) return areaRank;
  const g = getRiverGraph();
  const order = [];
  for (let i = 0; i < g.n; i++) if (g.active[i]) order.push(i);
  order.sort((a, b) => g.area[a] - g.area[b] || a - b);
  const rank = new Float32Array(g.n).fill(-1);
  for (let k = 0; k < order.length; k++) rank[order[k]] = k / Math.max(1, order.length - 1);
  areaRank = { order, rank };
  return areaRank;
}

// Which baked variant stands in for this river chunk. Picking it by hash,
// as the cell level does, is wrong here: cell chunks are interchangeable,
// but a headwater and a trunk differ by a factor of six in width, and a
// trunk drawn as a headwater is most of the deficit that showed up as
// impostor frames measuring 22-52% dimmer than the chunks they replaced.
// What a river chunk actually puts on screen, in local units^2 of cell that
// is NOT hidden behind another cell.
//
// An impostor is one of 12 baked chunks, chosen by drainage area. That was
// close enough while the mark was an additive streak whose size tracked the
// channel. It is not close enough now: with a cell of fixed size, the whole
// of a channel's weight is carried by HOW MANY cells and HOW WIDE they are
// spread, and drainage area is power-law, so the top band holds nodes that
// differ by an order of magnitude. Measured, the stand-ins came out 1.80x
// brighter than the chunks they replaced at 771 px - visibly fatter
// channels, not a subtle grade.
//
// Point count alone is the wrong correction: a narrow channel piles its
// cells on top of each other and most of them never reach the frame. Area
// times (1 - e^-coverage) is the part that does.
const visCache = new Map();
export function riverVis(i) {
  let v = visCache.get(i);
  if (v !== undefined) return v;
  const g = getRiverGraph();
  const sinu = fields().river.sinuosity || 1.0;
  let total = 0, pts = 0;
  for (const j of nodeHalfEdges(i)) {
    const dx = g.pos[j * 3] - g.pos[i * 3], dy = g.pos[j * 3 + 1] - g.pos[i * 3 + 1];
    const len = Math.hypot(dx, dy) * sinu * 0.5;          // this node's half
    const hwMean = (g.hw[i] + (g.hw[i] + g.hw[j]) * 0.5) * 0.5;
    const area = Math.max(len * 2 * hwMean, 1e-6);
    const quota = Math.max(60, Math.min(RIVER_CELLS ? 9000 : 4000,
      Math.round((RIVER_CELLS ? TARGET_COVER_CELL : TARGET_COVER) * area /
                 CELL_INK_AREA)));
    pts += quota;
    total += area * (1 - Math.exp(-quota * CELL_INK_AREA / area));
  }
  // the per-chunk cap bites on the widest trunks, and it bites the real
  // chunk and its stand-in differently
  if (pts > RIVER_POINT_CAP) total *= RIVER_POINT_CAP / pts;
  total /= RIVER_UNIT * RIVER_UNIT;                        // -> local units^2
  visCache.set(i, total);
  return total;
}

// The node each baked variant was built from (see variantData).
export function riverVariantNode(v, count) {
  const { order } = riverAreaRank();
  return order[Math.min(order.length - 1,
                        Math.floor((v + 0.5) / count * order.length))];
}

const GAIN_CAL = 1.10;
const gainCache = new Map();
export function riverImpostorGain(node, v, count) {
  const k = node * 64 + v;
  let g = gainCache.get(k);
  if (g !== undefined) return g;
  // riverVis is an analytic estimate of what the rasteriser does, so the
  // ratio of two of them is close but not exact. Measured against forced
  // impostor frames it runs about 10% low across the sizes the hand-off
  // actually uses; this is that correction, not a free parameter.
  const a = riverVis(node), b = riverVis(riverVariantNode(v, count));
  g = Math.min(2.5, Math.max(0.25, GAIN_CAL * a / Math.max(b, 1e-9)));
  gainCache.set(k, g);
  if (gainCache.size > 8192) gainCache.delete(gainCache.keys().next().value);
  return g;
}

export function riverVariant(node, count) {
  const { rank } = riverAreaRank();
  const r = rank[node];
  if (r < 0) return 0;
  return Math.min(count - 1, Math.floor(r * count));
}

// Representative chunks for the impostor atlas. Fixed seeds, so the bake is
// deterministic and the same picture every run.
export function variantData(level, v, count) {
  if (level === 1) {
    // Span the drainage-area distribution, and sample each variant at the
    // MIDDLE of the band riverVariant() will assign to it, so the stand-in
    // is the median chunk of its band rather than its edge.
    const { order } = riverAreaRank();
    const k = Math.min(order.length - 1,
                       Math.floor((v + 0.5) / count * order.length));
    return buildRiver(order[k]);
  }
  return buildCell(VARIANT_NODE + v, 0);
}

export function probeAnchor() {
  const g = getRiverGraph();
  let i = 0;
  while (i < g.n && !g.active[i]) i++;
  const { origins, count } = getCellOrigins(i);
  if (!count) return [g.pos[i * 3], g.pos[i * 3 + 1], g.pos[i * 3 + 2]];
  return [origins[0], origins[1], origins[2]];
}
