import * as THREE from 'three';
import { ScaleCascade, levelWeight } from './cascade.js';
import { LEVELS, probeAnchor } from './content.js';
import * as content from './content.js';
import * as fieldsMod from './fields.js';
import { loadFields } from './fields.js';
import { loadCellAtlas } from './cells.js';

// The density fields are the content: nothing can be generated until the
// fields extracted from the reference photographs are in memory.
const BASE = import.meta.env.BASE_URL || '/';
const fieldMeta = await loadFields(BASE);
// The cell atlas has to exist before the cascade is built: the material
// needs its tile grid as a #define and the impostor bake runs in the
// constructor.
const cellInfo = await loadCellAtlas(BASE);

/* ---------------------------------------------------------------- */
/* renderer                                                          */
/* ---------------------------------------------------------------- */

const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x05060a, 1);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
// Depth is never tested (additive points), so the clip volume only has to be
// generous enough that a chunk offset in z is never clipped away.
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1e5, 1e5);
camera.position.set(0, 0, 1);

const cascade = new ScaleCascade(scene, renderer);

/* ---------------------------------------------------------------- */
/* zoom state                                                        */
/* ---------------------------------------------------------------- */
// u = log10(H). H is the view half-height in absolute (cell) units.
// u = 0   -> half-height is one cell diameter
// u = 4.1 -> the whole galaxy is framed
const U_MIN = -0.3;
const U_MAX = 3.65;   // galaxy unit is 10^3.5 since P7 (see content.js)

const anchor = probeAnchor();
const state = {
  u: U_MAX,
  uT: U_MAX,
  F: [0, 0, 0],
  FT: [0, 0, 0],
};

function H() { return Math.pow(10, state.u); }

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h);
  const aspect = w / h;
  camera.left = -aspect; camera.right = aspect;
  camera.top = 1; camera.bottom = -1;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

/* ---------------------------------------------------------------- */
/* controls: logarithmic zoom toward the cursor + drag pan           */
/* ---------------------------------------------------------------- */
// Linear camera motion feels wrong across decades: nothing happens when
// far out, everything blurs when close in. Zoom is linear in u = log10(H),
// so the apparent rate of change is constant.

const el = renderer.domElement;

function ndc(ev) {
  return [
    (ev.clientX / window.innerWidth) * 2 - 1,
    -((ev.clientY / window.innerHeight) * 2 - 1),
  ];
}

el.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const [nx, ny] = ndc(ev);
  const aspect = window.innerWidth / window.innerHeight;
  const h0 = Math.pow(10, state.uT);
  // absolute point currently under the cursor
  const ax = state.FT[0] + nx * aspect * h0;
  const ay = state.FT[1] + ny * h0;

  const k = ev.deltaMode === 1 ? 0.04 : 0.0016;
  state.uT = Math.min(U_MAX, Math.max(U_MIN, state.uT + ev.deltaY * k));

  // keep that point pinned under the cursor
  const h1 = Math.pow(10, state.uT);
  state.FT[0] = ax - nx * aspect * h1;
  state.FT[1] = ay - ny * h1;
}, { passive: false });

let dragging = false, lastX = 0, lastY = 0;
el.addEventListener('pointerdown', (ev) => {
  dragging = true; lastX = ev.clientX; lastY = ev.clientY;
  el.setPointerCapture(ev.pointerId);
});
el.addEventListener('pointerup', (ev) => {
  dragging = false; el.releasePointerCapture(ev.pointerId);
});
el.addEventListener('pointermove', (ev) => {
  if (!dragging) return;
  const h = Math.pow(10, state.uT);
  const dx = (ev.clientX - lastX) / window.innerHeight * 2 * h;
  const dy = (ev.clientY - lastY) / window.innerHeight * 2 * h;
  state.FT[0] -= dx;
  state.FT[1] += dy;
  lastX = ev.clientX; lastY = ev.clientY;
});

window.addEventListener('keydown', (ev) => {
  if (ev.key === 'r') { state.uT = U_MAX; state.FT = [0, 0, 0]; }
  if (ev.key === 'd') { state.uT = U_MIN; state.FT = [anchor[0], anchor[1], anchor[2]]; }
  if (ev.key === ' ') { ev.preventDefault(); tour.active = !tour.active; }
});

/* ---------------------------------------------------------------- */
/* auto tour (space)                                                 */
/* ---------------------------------------------------------------- */
const tour = { active: false, t: 0 };

/* ---------------------------------------------------------------- */
/* HUD                                                               */
/* ---------------------------------------------------------------- */
const hud = document.getElementById('hud');
const sbFill = document.getElementById('sbFill');
const sbLabel = document.getElementById('sbLabel');
const sbBar = document.getElementById('scalebar');
const sbHit = document.getElementById('sbHit');
const sbThumb = document.getElementById('sbThumb');
const sbU = document.getElementById('sbU');
const sbTicks = document.getElementById('sbTicks');
let fps = 60, lastT = performance.now(), worstFps = 999, fpsSamples = [];

/* ---------------------------------------------------------------- */
/* the scale bar is the zoom control                                 */
/* ---------------------------------------------------------------- */
// The wheel is a local gesture: it moves you a little, from wherever you
// already are. Nothing in the piece let you address the range as a whole -
// to get from a cell to the galaxy you had to keep scrolling and hope. The
// bar was already drawing that range, so it may as well be grabbable.
//
// Dragging it changes uT only. F is left alone, so the slider zooms about
// the centre of the current view, the way the keyboard shortcuts do; the
// wheel keeps its own cursor-pinning behaviour.

const sbPct = (u) => ((u - U_MIN) / (U_MAX - U_MIN)) * 100;

let sbDrag = false;

function sbFromEvent(ev) {
  const r = sbHit.getBoundingClientRect();
  const t = Math.min(1, Math.max(0, (ev.clientX - r.left) / Math.max(r.width, 1)));
  return U_MIN + t * (U_MAX - U_MIN);
}

function sbSet(u) {
  state.uT = Math.min(U_MAX, Math.max(U_MIN, u));
  tour.active = false;      // a hand on the slider outranks the tour
}

sbHit.addEventListener('pointerdown', (ev) => {
  ev.preventDefault();
  sbDrag = true;
  sbBar.classList.add('dragging');
  sbHit.setPointerCapture(ev.pointerId);
  sbHit.focus();
  sbSet(sbFromEvent(ev));
});
sbHit.addEventListener('pointermove', (ev) => { if (sbDrag) sbSet(sbFromEvent(ev)); });
function sbEnd(ev) {
  if (!sbDrag) return;
  sbDrag = false;
  sbBar.classList.remove('dragging');
  try { sbHit.releasePointerCapture(ev.pointerId); } catch (e) { /* already released */ }
}
sbHit.addEventListener('pointerup', sbEnd);
sbHit.addEventListener('pointercancel', sbEnd);

// The wheel over the bar is the canvas gesture minus the cursor pinning -
// otherwise the bar would be a dead zone for the scroll wheel.
sbHit.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  sbSet(state.uT + ev.deltaY * (ev.deltaMode === 1 ? 0.04 : 0.0016));
}, { passive: false });

sbHit.addEventListener('keydown', (ev) => {
  const step = ev.shiftKey ? 0.25 : 0.05;
  if (ev.key === 'ArrowLeft'  || ev.key === 'ArrowDown') { sbSet(state.uT - step); ev.preventDefault(); }
  if (ev.key === 'ArrowRight' || ev.key === 'ArrowUp')   { sbSet(state.uT + step); ev.preventDefault(); }
  if (ev.key === 'Home') { sbSet(U_MIN); ev.preventDefault(); }
  if (ev.key === 'End')  { sbSet(U_MAX); ev.preventDefault(); }
});

// Ticks go where each level's cross-fade actually peaks. Evenly spaced
// labels would misstate the hierarchy: the cell-river step is 10^1.5 and
// the river-galaxy step is 10^2, so the gaps are not equal.
for (let i = 0; i < LEVELS.length; i++) {
  let top = 0;
  for (let u = U_MIN; u <= U_MAX + 1e-9; u += 0.01) {
    const w = levelWeight(LEVELS[i].unit, Math.pow(10, u), i);
    if (w > top) top = w;
  }
  let sum = 0, n = 0;
  for (let u = U_MIN; u <= U_MAX + 1e-9; u += 0.01) {
    const w = levelWeight(LEVELS[i].unit, Math.pow(10, u), i);
    if (w >= top - 1e-6) { sum += u; n++; }      // middle of the plateau
  }
  const at = n ? sum / n : U_MIN;
  const b = document.createElement('button');
  b.className = 'tick';
  b.textContent = LEVELS[i].name;
  b.style.left = Math.min(96, Math.max(4, sbPct(at))).toFixed(1) + '%';
  b.title = LEVELS[i].name + ' at u=' + at.toFixed(2);
  // Keep the keyboard on the slider, not on the tick that was just hit:
  // after jumping to a level you usually want to nudge from there.
  b.addEventListener('click', () => { sbSet(at); sbHit.focus(); });
  sbTicks.appendChild(b);
}

function updateHud(h) {
  const st = cascade.stats;
  const w = st.weights;
  const label = w[0] > w[1] && w[0] > w[2] ? 'cell' : (w[1] >= w[2] ? 'river' : 'galaxy');
  sbLabel.textContent = label;
  // While the slider is being dragged, follow the target rather than the
  // eased value: the thumb has to stay under the finger even though the
  // view is still catching up to it.
  const uShown = sbDrag ? state.uT : state.u;
  const pc = sbPct(uShown);
  sbFill.style.width = pc.toFixed(1) + '%';
  sbThumb.style.left = pc.toFixed(1) + '%';
  sbU.textContent = 'u=' + uShown.toFixed(2);
  sbHit.setAttribute('aria-valuenow', uShown.toFixed(2));
  sbHit.setAttribute('aria-valuetext', label + ' u=' + uShown.toFixed(2));
  hud.innerHTML =
    `<b>zoom</b>  u=${state.u.toFixed(2)}   H=${h.toPrecision(3)} cell-units\n` +
    `<b>fps</b>   ${fps.toFixed(0)}  (worst ${worstFps === 999 ? '-' : worstFps.toFixed(0)})\n` +
    `<b>points</b> ${st.points.toLocaleString()}   chunks ${st.chunks}` +
    `   impostors ${st.impostors.toLocaleString()}` +
    (st.dropped ? `  (+${st.dropped} dropped)` : '') + `\n` +
    `<b>draws</b>  ${renderer.info.render.calls}   cache ${cascade.cache.size}\n` +
    `<b>levels</b> ` + LEVELS.map((L, i) => `${L.name} ${w[i].toFixed(2)}`).join('  ') + `\n` +
    `<b>fields</b> galaxy ${fieldMeta.galaxy.w}x${fieldMeta.galaxy.h}  ` +
    `river sinuosity ${fieldMeta.river.sinuosity}\n` +
    `\nwheel zoom · drag pan · drag the scale bar · [space] tour · [d] dive · [r] reset`;
}

/* ---------------------------------------------------------------- */
/* loop                                                              */
/* ---------------------------------------------------------------- */

function frame(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  const inst = 1 / Math.max(dt, 1e-4);
  fps += (inst - fps) * 0.1;

  if (tour.active) {
    tour.t += dt * 0.09;
    const p = (Math.sin(tour.t * Math.PI * 2 - Math.PI / 2) * 0.5 + 0.5);
    state.uT = U_MIN + (U_MAX - U_MIN) * p;
    state.FT[0] += (anchor[0] * (1 - p) - state.FT[0]) * 0.02;
    state.FT[1] += (anchor[1] * (1 - p) - state.FT[1]) * 0.02;
  }

  // exponential ease, in log space for zoom
  const ease = 1 - Math.pow(0.0015, dt);
  state.u += (state.uT - state.u) * ease;
  state.F[0] += (state.FT[0] - state.F[0]) * ease;
  state.F[1] += (state.FT[1] - state.F[1]) * ease;
  state.F[2] += (state.FT[2] - state.F[2]) * ease;

  // follow the z of whatever sheet we are looking at, so pz stays near 0
  state.FT[2] += (cascade.stats.focusZ - state.FT[2]) * Math.min(1, dt * 4);

  const h = H();
  const aspect = window.innerWidth / window.innerHeight;
  const halfPxY = window.innerHeight * 0.5 * renderer.getPixelRatio();

  cascade.update(state.F, h, aspect, halfPxY);
  renderer.render(scene, camera);

  // The boot notice stays up until something has actually been drawn, so a
  // page that fails silently says so instead of showing its own background.
  if (window.__bootOK) { window.__bootOK(); window.__bootOK = null; }

  if (fpsSamples.length < 100000) fpsSamples.push(inst);
  if (now > 3000 && inst < worstFps) worstFps = inst;
  updateHud(h);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

/* ---------------------------------------------------------------- */
/* dev probes (used to verify P0, not part of the piece)             */
/* ---------------------------------------------------------------- */

// Where does the real (float32, GPU-bound) pipeline place each visible
// chunk, compared with an exact float64 computation? Any growth of this
// error as we zoom in is the precision failure P0 exists to rule out.
// We measure the chunk's local (1,1) corner, i.e. a real offset, not just
// the translation, so a degenerate scale would show up too.
window.__probe = function (u) {
  state.u = state.uT = u;
  state.F[0] = state.FT[0] = anchor[0];
  state.F[1] = state.FT[1] = anchor[1];
  state.F[2] = state.FT[2] = anchor[2];
  const h = Math.pow(10, u);
  const aspect = window.innerWidth / window.innerHeight;
  const halfPxY = window.innerHeight * 0.5 * renderer.getPixelRatio();
  cascade.update(state.F, h, aspect, halfPxY);

  const pxPerView = window.innerHeight * 0.5;
  let maxErr = 0, n = 0;
  for (const e of cascade.cache.values()) {
    if (!e.points.visible || !e.origin) continue;
    const m = e.points.matrixWorld.elements;   // float32
    const pipeX = m[12] + m[0];                // local (1,1,0) corner
    const pipeY = m[13] + m[5];
    const exactX = (e.origin[0] + e.unit - state.F[0]) / h;   // float64
    const exactY = (e.origin[1] + e.unit - state.F[1]) / h;
    const err = Math.hypot(pipeX - exactX, pipeY - exactY) * pxPerView;
    if (err > maxErr) maxErr = err;
    n++;
  }
  return {
    u: +u.toFixed(2),
    H: h,
    errPx: maxErr,
    checked: n,
    chunks: cascade.stats.chunks,
    points: cascade.stats.points,
    weights: cascade.stats.weights.map((w) => +w.toFixed(2)),
  };
};

window.__sweep = function () {
  const rows = [];
  for (let u = U_MIN; u <= U_MAX + 1e-9; u += 0.25) rows.push(window.__probe(u));
  const maxErr = Math.max(...rows.map((r) => r.errPx));
  const maxPts = Math.max(...rows.map((r) => r.points));
  return { maxErrPx: maxErr, maxPoints: maxPts, rows };
};

window.__bench = function (seconds = 6) {
  return new Promise((resolve) => {
    fpsSamples = [];
    worstFps = 999;
    tour.active = true;
    tour.t = 0;
    setTimeout(() => {
      tour.active = false;
      const s = fpsSamples.slice(30).sort((a, b) => a - b);
      resolve({
        n: s.length,
        median: s[Math.floor(s.length * 0.5)],
        p5: s[Math.floor(s.length * 0.05)],
        min: s[0],
      });
    }, seconds * 1000);
  });
};

window.__dbg = { cascade, state, renderer, scene, camera, LEVELS, anchor,
                 content, fields: fieldsMod };
