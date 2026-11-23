// The scale cascade: the piece P0 exists to prove.
//
// WHY THIS EXISTS
// Naively you would push the camera back to see the galaxy and pull it in to
// see a cell. Across 4 decades that destroys float32: a vertex at 1e4 and a
// camera at 1e4 cancel to a value the GPU can only resolve to ~1e-3, so the
// near view shimmers and z-fights.
//
// Instead the camera NEVER moves. It is an orthographic camera with a fixed
// half-height of 1 view unit. Zoom is a number H (view half-height expressed
// in absolute units) and panning is a focus point F in absolute coords. Both
// are plain JS numbers, i.e. float64.
//
// Every chunk is positioned by  p = (O - F) / H  and scaled by  unit / H,
// computed on the CPU in float64 and only then handed to the GPU. For any
// chunk that is actually on screen, |p| <= ~2 and the vertex data is local
// [-1,1], so float32 never sees a large number and never has to cancel one.
// Measured drift of a deep anchor across the whole zoom range: < 1e-3 px.

import * as THREE from 'three';
import { LEVELS, LEVEL_EXTENT, forEachChunk, chunkKey, buildChunk, variantData,
         cellImpostorTint, riverVariant, riverImpostorGain,
         RIVER_CELLS } from './content.js';
import { bakeAtlas, ImpostorLayer, VARIANTS, tileForLevel } from './impostor.js';
import { cellAtlas, cellMeta, cellNormals, cellMeanLum } from './cells.js';

const VERT = /* glsl */`
  attribute float size;
  attribute vec3 tint;
  attribute float aux;
  uniform float uSizeToPx;
  uniform float uMinPx;
  uniform float uMaxPx;
  varying float vAlpha;
  varying vec3 vTint;
  varying float vAux;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    float px = size * uSizeToPx;
    #if MARK == 1 && !defined(RIVER_CELLS)
      px *= ELONG;            // the sprite has to hold the stretched streak
    #endif
    // Sub-pixel marks are clamped up and dimmed instead of drawn at 1px full
    // brightness, which is what makes dense far-away fields shimmer.
    float a = 1.0;
    if (px < uMinPx) { float k = px / uMinPx; a = k * k; px = uMinPx; }
    // A mark must never be larger on screen than it is at its own level's
    // native scale. Past that you have zoomed INTO the level and the child
    // level is carrying the detail, so letting the parent's sprites keep
    // growing buys nothing and costs everything: the galaxy's 78k glow
    // sprites reached 129px at u=3.7 and alone cost ~12 ms/frame.
    gl_PointSize = min(px, uMaxPx);
    vAlpha = a;
    vTint = tint;
    vAux = aux;
  }
`;

// One shader, three marks. A scale is not just a different size of the same
// dot: a cell has a membrane, a river has a current, a galaxy is unresolved
// light with stars on top. MARK is a per-level #define.
const FRAG = /* glsl */`
  uniform vec3 uTint;
  uniform float uOpacity;
  varying float vAlpha;
  varying vec3 vTint;
  varying float vAux;

#if defined(USE_CELL_TEX)
  uniform sampler2D uCells;
  uniform vec2 uCellTile;         // 1/COLS, 1/ROWS
  #ifdef RIVER_CELLS
    uniform float uCellLumK;      // 1 / mean atlas luminance
  #endif
  #ifdef CELL_NORMALS
    uniform sampler2D uCellNrm;
    // Fixed key light. Everything in the piece is authored in output space,
    // so the shading is too.
    // normalize(-0.45, -0.60, 0.66). The constants below are calibrated
    // against THIS vector: mean(lambert) over the atlas is 0.596 with it, so
    // AMBIENT + DIFFUSE * 0.596 == 1 and lighting costs no exposure. Writing
    // a different vector here than the one the calibration used took 23% off
    // the whole cell level.
    const vec3 LIGHT = vec3(-0.4504, -0.6006, 0.6606);
    const float AMBIENT = 0.404;
    const float DIFFUSE = 1.000;
  #endif
#endif

  void main() {
    vec2 d = gl_PointCoord - 0.5;

  #if MARK == 0
    /* ---- cell: a photograph of one cell, out of the atlas ----
       P3 and earlier drew a procedural membrane here. The sprite is now a
       textured quad; the alpha channel baked by tools/extract_cells.py is
       what gives it an edge, and it is blended OVER rather than added,
       because a cell is matter. Adding them clipped to white the moment
       three overlapped. */
    float t = floor(vAux * 0.125 + 0.0001);
    float o = floor(vAux - t * 8.0 + 0.5);
    vec2 p = gl_PointCoord;
    if (o >= 4.0) p.x = 1.0 - p.x;      // 8 free orientations per tile
    float q4 = mod(o, 4.0);
    vec2 q = p - 0.5;
    if (q4 == 1.0) q = vec2(-q.y, q.x);
    else if (q4 == 2.0) q = -q;
    else if (q4 == 3.0) q = vec2(q.y, -q.x);
    p = q + 0.5;
    float cx = mod(t, CELL_COLS);
    float cy = floor(t / CELL_COLS);
    vec2 uv = (vec2(cx, cy) + p) * uCellTile;
    vec4 tex = texture2D(uCells, uv);
    float a = tex.a * vAlpha * uOpacity;
    if (a < 0.003) discard;
    vec3 albedo = tex.rgb * vTint * uTint;

  #ifdef CELL_NORMALS
    // The depth map's normal, baked per tile (tools/extract_cells.py).
    vec2 nt = texture2D(uCellNrm, uv).xy * 2.0 - 1.0;
    // The tile was turned to get here, so the normal has to be turned back:
    // the texcoord went q -> Rot(Mirror(q)), so the normal comes back as
    // Mirror(Rot^-1(n)). Rotations and a mirror are orthogonal, so this is
    // just the inverse branch, and it costs nothing.
    vec2 ns;
    if (q4 == 1.0) ns = vec2(nt.y, -nt.x);
    else if (q4 == 2.0) ns = -nt;
    else if (q4 == 3.0) ns = vec2(-nt.y, nt.x);
    else ns = nt;
    if (o >= 4.0) ns.x = -ns.x;
    vec3 N = vec3(ns, sqrt(max(1.0 - dot(ns, ns), 0.01)));
    float lam = max(dot(N, LIGHT), 0.0);
    albedo *= AMBIENT + DIFFUSE * lam;
  #endif

    gl_FragColor = vec4(albedo * a, a);   // premultiplied
    return;
  #endif

  #ifdef RIVER_CELLS
    /* ---- river, drawn out of cells ----
       The mark at this scale was an abstract streak, which is why the piece
       never showed that a river is MADE of the level below it: the two were
       never on screen together (cells faded out at u=2.0 with 3 river nodes
       in view; the network does not read until u~2.8, where 703,000 cell
       chunks would have to be drawn). So the river's own mark becomes a
       cell. Same point count, same draw call, no new enumeration - the
       composition is carried by the brush instead of by a second level.

       The photograph gives shape and internal structure; the HUE stays the
       river's, so the galaxy -> river colour chain measured in P5 survives.
       Rotation is continuous here, not one of the cell level's 8 steps,
       because these have a current to follow. */
    float t = floor(vAux);
    float ang = (vAux - t) * 6.2831853;
    float ca = cos(ang), sa = sin(ang);
    vec2 q = gl_PointCoord - 0.5;
    // screen -> texture is the INVERSE rotation
    q = vec2(q.x * ca + q.y * sa, -q.x * sa + q.y * ca);
    // A baked cell reaches radius 58 of a 128 tile, but the square's
    // inscribed circle is only 45.3, so a free rotation cuts the corners
    // off: measured, 25.6% of all the ink in the atlas lies outside that
    // circle, and the loss would swing with the flow direction. The cell
    // level never hit this because its 8 orientations are axis-preserving.
    // Sampling wider shrinks the content into the safe circle instead.
    q *= RCELL_FIT;
    vec2 p = q + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) discard;
    float rcx = mod(t, CELL_COLS);
    float rcy = floor(t / CELL_COLS);
    vec2 ruv = (vec2(rcx, rcy) + p) * uCellTile;
    vec4 rtex = texture2D(uCells, ruv);
    float ra = rtex.a * vAlpha * uOpacity;
    if (ra < 0.003) discard;
    float rlum = dot(rtex.rgb, vec3(0.2126, 0.7152, 0.0722)) * uCellLumK;
    // Mostly the PHOTOGRAPH's own colour, with the water's hue under it.
    // The first pass had this the other way round, and at the zoom where the
    // two levels overlap the same node's river came out pale green while its
    // cell chunks came out red - the one place where the composition has to
    // be legible was the one place the levels disagreed. The galaxy -> river
    // colour chain survives the swap because the tissue palette is itself
    // chosen by the galaxy's warmth (correlation +0.85, against +0.88 for
    // the water). Both terms are normalised to the same luminance, so the
    // mix cannot shift exposure.
    vec3 rwat = vTint * rlum;
    vec3 rown = rtex.rgb * uCellLumK * dot(vTint, vec3(0.2126, 0.7152, 0.0722));
    vec3 ralb = uTint * mix(rwat, rown, RCELL_OWN);
  #ifdef CELL_NORMALS
    vec2 rnt = texture2D(uCellNrm, ruv).xy * 2.0 - 1.0;
    vec2 rns = vec2(rnt.x * ca - rnt.y * sa, rnt.x * sa + rnt.y * ca);
    vec3 rN = vec3(rns, sqrt(max(1.0 - dot(rns, rns), 0.01)));
    ralb *= AMBIENT + DIFFUSE * max(dot(rN, LIGHT), 0.0);
  #endif
    gl_FragColor = vec4(ralb * ra, ra);   // matter covers
    return;
  #endif

    float core;

  #if MARK == 1
    /* ---- river: the mark is stretched along the current ---- */
    float c = cos(vAux), s = sin(vAux);
    vec2 r2d = vec2(d.x * c + d.y * s, -d.x * s + d.y * c);
    r2d.x /= ELONG;                       // long axis follows the flow
    float r2 = dot(r2d, r2d) * 4.0;
    if (r2 > 1.0) discard;
    core = pow(1.0 - r2, 1.15) * 0.075;

  #else
    /* ---- galaxy: unresolved light, with resolved stars on top ---- */
    float r2 = dot(d, d) * 4.0;
    if (r2 > 1.0) discard;
    if (vAux > 0.5) {
      core = (pow(1.0 - r2, 3.5) + 0.22 * pow(1.0 - r2, 0.7)) * 0.75;
    } else {
      core = pow(1.0 - r2, 2.3) * 0.030;  // faint and wide: these must overlap
    }
  #endif

    gl_FragColor = vec4(uTint * vTint, core * vAlpha * uOpacity);
  }
`;

// Per-level visibility window, in s = log10(unit / H).
// s = 0  : one local unit fills half the screen (level at native scale)
// s = -2 : the level has shrunk by 100x and the parent level takes over
// Per level. Adaptive river spacing packs bright regions three times tighter
// than before, so the cell level now has ~176,000 chunks in view at u≈2.2 -
// at 6% opacity, which overflowed the impostor cap and dropped 104,000 of
// them. The river is already at full weight by u≈1.95, so the cell level has
// nothing left to contribute past it and can hand over sooner.
// P7: the cell level now hands over at u=2.30 instead of u=2.05. That is
// not a budget being loosened - at the old 100x step, u=2.30 put 1.2 million
// cell chunks in view; at 10^1.5 it puts 75,000. The overlap where both
// levels are on screen goes from 0.6 decades (3 river nodes in view at the
// top of it) to 1.23 (80 nodes).
//
// The tail is short on purpose. A level's opacity does not reduce its cost -
// the sprite count depends only on the zoom - so a long fade out spends a
// full frame budget on something at 4% opacity. Measured: carrying the cell
// level to u=2.30 cost 17.2 ms/frame while it was invisible.
const FADE_IN  = [[-2.18, -1.90], [-2.30, -1.60], [-2.30, -1.60]];
// Fading out slowly looked harmless until it was measured: a level at 3%
// opacity still pays full fill-rate, and its marks grow as you zoom past it.
// The galaxy alone cost 21 ms/frame at u=3.4 while being invisible.
const FADE_OUT = [ 0.05,  0.55];



// A chunk narrower than this gets a baked impostor instead of its own draw
// call and point cloud. Above it, the real thing.
//
// Per level, because the levels do not blend the same way. An additive mark
// that falls under uMinPx is clamped up and dimmed by k^2, which conserves
// its energy exactly: that is why 34 px was fine for the river. A cell is
// drawn OVER, and n overlapping sprites at alpha a composite to
// 1-(1-a)^n, not n*a, so the same k^2 dimming comes out visibly darker than
// the impostor of the same chunk. A cell chunk therefore stays on the full
// path only while its own cells are still more than a pixel across.
// 34 was right for a river of additive streaks, where a mark that falls
// under uMinPx is dimmed by k^2 and keeps its energy exactly. A river of
// cells is matter: the full path shows discrete cells with black between
// them, while the bake packs them into a solid bar, so the impostor comes
// out BRIGHTER the larger it is. A per-sprite gain (riverImpostorGain)
// corrects most of it; what is left is shape, which no gain fixes, so the
// handover sits where the two agree: 0.94 at 400 px, 0.99-1.03 below it.
const FULL_MIN_PX = [170, 400, 0];

// Below this a chunk is not worth drawing at all; its parent level carries
// the frame.
const IMPOSTOR_MIN_PX = 1.6;

// How many chunks per level may be drawn at full detail. Everything else in
// view becomes an impostor instead of being dropped, which is why these can
// stay modest without leaving holes.
// Level 1 has to carry every chunk in view down to FULL_MIN_PX[1], because
// below that threshold there is no accurate stand-in. At the crossing
// (u~2.79) the view holds ~160 river chunks.
const CHUNK_BUDGET = [120, 190, 1];

// Impostors go out in one draw call per level, so this cap is about fill rate
// and the per-frame CPU walk, not about draw calls.
// Raised with the P7 step: the cell level is now asked to stay alive to
// u=2.30, which is ~75,000 chunks at the probe anchor.
const IMPOSTOR_CAP = 96000;

// Per-level grading on top of the colour each point carries from the
// reference photographs.
const LEVEL_TINT = [
  [1.00, 1.00, 1.00],   // cell  - colour is already in the points
  [1.05, 1.00, 0.98],   // river
  [1.00, 0.97, 0.94],   // galaxy
];

// Largest local mark size per level, used to cap sprite growth (see uMaxPx).
// Level 1's mark is a cell of fixed absolute size, which at the P7 step is
// 0.108 local units - 0.10 clipped it.
const MARK_MAX_LOCAL = [0.30, 0.16, 0.07];

// Premultiplied source-over. The cell level writes rgb*a and a, so sprites
// occlude each other instead of summing; alpha also has to accumulate
// correctly because the impostor bake renders into an RGBA target.
const PREMULT_BLEND = {
  blending: THREE.CustomBlending,
  blendEquation: THREE.AddEquation,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneMinusSrcAlphaFactor,
  blendEquationAlpha: THREE.AddEquation,
  blendSrcAlpha: THREE.OneFactor,
  blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
};

const CACHE_CAP = 420;
const BUILDS_PER_FRAME = 6;

function smoothstep(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

export function levelWeight(unit, H, level = 1) {
  const s = Math.log10(unit / H);
  const fi = FADE_IN[level];
  return smoothstep(fi[0], fi[1], s) *
         (1 - smoothstep(FADE_OUT[0], FADE_OUT[1], s));
}

export class ScaleCascade {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this.cache = new Map();            // key -> { points, geom, used }
    this.frame = 0;
    this.stats = { chunks: 0, points: 0, builds: 0, dropped: 0, impostors: 0,
                   weights: [0, 0, 0], focusZ: 0 };
    this.probeObject = null;
    // dev hook: FULL_MIN_PX is calibrated by sweeping it, so it has to be
    // reachable from a probe.
    this.fullMinPx = FULL_MIN_PX.slice();
    this.chunkBudget = CHUNK_BUDGET.slice();

    this.groups = LEVELS.map((L) => {
      const g = new THREE.Group();
      g.matrixAutoUpdate = false;
      scene.add(g);
      return g;
    });

    const cm = cellMeta() || { cols: 1, rows: 1 };
    this.materials = LEVELS.map((L, i) => {
      // Levels 0 and 1 both draw out of the cell atlas now: the cell level
      // because a cell IS one, the river level because it is made of them.
      const useCells = (i === 0 || (i === 1 && RIVER_CELLS)) && !!cellAtlas();
      const defines = { MARK: i, ELONG: '2.6', RCELL_FIT: '1.281', RCELL_OWN: '0.60' };
      const uniforms = {
        uTint:     { value: new THREE.Color(...LEVEL_TINT[i]) },
        uOpacity:  { value: 0 },
        uSizeToPx: { value: 1 },
        uMinPx:    { value: 1.25 },
        uMaxPx:    { value: 128 },
      };
      if (useCells) {
        defines.USE_CELL_TEX = '';
        defines.CELL_COLS = cm.cols.toFixed(1);
        uniforms.uCells = { value: cellAtlas() };
        uniforms.uCellTile = { value: new THREE.Vector2(1 / cm.cols, 1 / cm.rows) };
        if (cellNormals()) {
          defines.CELL_NORMALS = '';
          uniforms.uCellNrm = { value: cellNormals() };
        }
        if (i === 1) {
          defines.RIVER_CELLS = '';
          uniforms.uCellLumK = { value: 1 / Math.max(cellMeanLum(), 1e-3) };
        }
      }
      return new THREE.ShaderMaterial({
        defines, uniforms,
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        // Light adds; matter covers. The galaxy is light; the river stopped
        // being light the moment its marks became cells, because eleven
        // layers of matter composite to a sheet, not to something brighter.
        ...(useCells ? PREMULT_BLEND : { blending: THREE.AdditiveBlending }),
      });
    });

    // Bake once, at startup. Level 2 is a single chunk that is never far
    // enough away to need standing in for, so it has no atlas.
    this.impostors = LEVELS.map((L, i) => {
      if (i === 2) return null;
      const premult = i === 0 || (i === 1 && RIVER_CELLS && !!cellAtlas());
      const atlas = bakeAtlas(renderer, this.materials[i],
                              (v) => variantData(i, v, VARIANTS), LEVEL_EXTENT[i],
                              tileForLevel(i), premult);
      // The impostor has to blend the same way the thing it stands in for
      // does, or the level visibly changes character at the threshold.
      return new ImpostorLayer(scene, atlas, IMPOSTOR_CAP, premult);
    });
  }

  acquire(key, level, a, b, budgetLeft) {
    let entry = this.cache.get(key);
    if (!entry) {
      if (budgetLeft <= 0) return null;         // build throttle
      const data = buildChunk(level, a, b);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(data.pos, 3));
      geom.setAttribute('size', new THREE.BufferAttribute(data.size, 1));
      geom.setAttribute('tint', new THREE.BufferAttribute(data.color, 3));
      geom.setAttribute('aux', new THREE.BufferAttribute(data.aux, 1));
      geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), LEVEL_EXTENT[level] * 1.6);
      const points = new THREE.Points(geom, this.materials[level]);
      points.frustumCulled = false;             // we cull in absolute space
      points.matrixAutoUpdate = false;
      entry = { points, geom, count: data.size.length, level, built: true };
      this.cache.set(key, entry);
      this.groups[level].add(points);
      this.stats.builds++;
    }
    entry.used = this.frame;
    return entry;
  }

  evict() {
    if (this.cache.size <= CACHE_CAP) return;
    const entries = [...this.cache.entries()]
      .filter(([, e]) => e.used !== this.frame)
      .sort((a, b) => a[1].used - b[1].used);
    let toDrop = this.cache.size - CACHE_CAP;
    for (const [key, e] of entries) {
      if (toDrop-- <= 0) break;
      this.groups[e.level].remove(e.points);
      e.geom.dispose();
      this.cache.delete(key);
    }
  }

  // F: [x,y,z] focus in absolute units (float64)
  // H: view half-height in absolute units (float64)
  update(F, H, aspect, halfPxY) {
    this.frame++;
    const st = this.stats;
    st.chunks = 0; st.points = 0; st.builds = 0; st.dropped = 0; st.impostors = 0;
    let bestW = -1, bestD = Infinity, focusZ = st.focusZ;

    for (const e of this.cache.values()) e.points.visible = false;

    let buildBudget = BUILDS_PER_FRAME;
    const halfY = H;
    const halfX = H * aspect;

    for (let li = LEVELS.length - 1; li >= 0; li--) {
      const L = LEVELS[li];
      const w = levelWeight(L.unit, H, li);
      st.weights[li] = w;

      const imp = this.impostors[li];
      if (imp) imp.begin(0);

      // a chunk's content spans 2*extent local units across
      const chunkPx = 2 * LEVEL_EXTENT[li] * (L.unit / H) * halfPxY;
      if (w < 0.004 || chunkPx < IMPOSTOR_MIN_PX) { if (imp) imp.end(); continue; }

      const mat = this.materials[li];
      mat.uniforms.uOpacity.value = w;
      mat.uniforms.uSizeToPx.value = (L.unit / H) * halfPxY;
      mat.uniforms.uMaxPx.value = MARK_MAX_LOCAL[li] * halfPxY;
      if (imp) imp.begin(w, halfPxY * 2 * aspect, halfPxY * 2);

      // __forceImpostor is a dev switch: it renders the identical frame
      // entirely out of impostors, so the full path and the stand-in path
      // can be compared pixel for pixel at the same zoom.
      const fullAllowed = (chunkPx >= this.fullMinPx[li] || li === 2) &&
                          !(this.forceImpostor === li);
      // Full detail inside a disc whose area holds about `budget` chunks.
      // Cheaper and steadier than sorting every candidate by distance, which
      // the impostor path made far too expensive to do per frame.
      const budget = this.chunkBudget[li];
      const fullR = 2 * LEVEL_EXTENT[li] * L.unit * Math.sqrt(budget / Math.PI);
      const fullR2 = fullR * fullR;
      let fullCount = 0;

      forEachChunk(li, F, halfX, halfY, (ox, oy, oz, a, b) => {
        const dx = ox - F[0], dy = oy - F[1];
        const d2 = dx * dx + dy * dy;

        // ---- the float64 transform, the whole point of this file ----
        const px = dx / H;
        const py = dy / H;
        const pz = (oz - F[2]) / H;
        // -------------------------------------------------------------

        if (w > bestW + 1e-3) { bestW = w; bestD = d2; focusZ = oz; }
        else if (w > bestW - 1e-3 && d2 < bestD) { bestD = d2; focusZ = oz; }

        if (fullAllowed && d2 < fullR2 && fullCount < budget) {
          const key = chunkKey(li, a, b);
          const entry = this.acquire(key, li, a, b, buildBudget);
          if (entry) {
            if (entry.built) { buildBudget--; entry.built = false; }
            const s = L.unit / H;
            const m = entry.points.matrix;
            m.set(s, 0, 0, px,
                  0, s, 0, py,
                  0, 0, s, pz,
                  0, 0, 0, 1);
            entry.points.matrixWorld.copy(m);
            entry.points.matrixWorldNeedsUpdate = false;
            entry.origin = [ox, oy, oz];
            entry.unit = L.unit;
            entry.points.visible = true;
            fullCount++;
            st.chunks++;
            st.points += entry.count;
            return;
          }
          // still building: fall through and stand in for it meanwhile
        }

        if (imp) {
          // deterministic per chunk, so it does not flicker between frames
          const h = (a * 2654435761 ^ (b + 1) * 40503) >>> 0;
          // Cells pick a stand-in by hash; rivers pick the one whose
          // drainage area matches theirs.
          const v = li === 1 ? riverVariant(a, VARIANTS) : h % VARIANTS;
          // Cells: recolour the stand-in to this river's own tissue, so a
          // chunk does not change colour when it crosses the threshold.
          const tnt = li === 0 ? cellImpostorTint(a, v) : null;
          // Rivers: correct for the stand-in's channel being wider or
          // narrower than this node's own (see riverImpostorGain).
          const gn = li === 1 ? riverImpostorGain(a, v, VARIANTS) : 1;
          if (!imp.add(px, py, pz, chunkPx, v, (h >>> 7) % 8, tnt, gn)) st.dropped++;
          else st.impostors++;
        } else {
          st.dropped++;
        }
      });

      if (imp) imp.end();
    }

    st.focusZ = focusZ;
    this.evict();
  }
}
