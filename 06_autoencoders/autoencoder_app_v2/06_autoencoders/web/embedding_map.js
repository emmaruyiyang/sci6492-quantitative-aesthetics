// Embedding Map front end.
//
// Every change of a setting asks the server for a new map (POST /api/run) and
// polls the job until it is done. The server returns, for every image of the
// dataset, a position in 0..1 x 0..1 and a cluster number. The images are
// placed with a CSS transform, and the .item transition in the stylesheet
// makes them glide from the previous map to the new one.

const $ = (id) => document.getElementById(id);

// Categorical colors in fixed order: cluster 0 (the biggest) always gets the
// first one. Clusters past the eighth and DBSCAN noise are drawn gray; the
// legend (click to highlight) and the hover panel still tell them apart.
const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const NOISE = "#9a9a98";
const colorOf = (label) => (label >= 0 && label < PALETTE.length ? PALETTE[label] : NOISE);

// Diverging colors of the barcode: negative blue, 0 light gray, positive red
const NEGATIVE = [42, 120, 214];
const MIDDLE = [236, 236, 234];
const POSITIVE = [227, 73, 72];
// Classifier softmax barcode: 0 light gray -> 1 dark blue (one hue)
const SCORE = [16, 60, 120];

// Images are identified by their index in the whole dataset (the server may
// use only a sample of it), so the same image keeps its element, and glides,
// when the sample grows or shrinks.
const state = {
  options: null,
  result: null,         // the last map from the server
  rows: new Map(),      // image index -> its row in the result's arrays
  coords: new Map(),    // image index -> [x, y], after alignment
  shown: new Map(),     // image index -> [x, y] where it is drawn (coords, or snapped to tiles)
  items: new Map(),     // image index -> element on the map
  dataset: null,        // dataset the elements belong to
  view: { scale: 1, tx: 0, ty: 0 },
  focus: null,          // highlighted cluster (legend click)
  hovered: null,        // image index of the image in the side panel
  run: 0,               // number of the latest run, older ones are ignored
  vectors: new Map(),   // image index -> vector of the hovered images (current result)
  shownVector: null,    // the vector in the side panel (redrawn when "raw values" changes)
  pick: { a: null, b: null },   // interpolation: image index of each end
  picking: null,        // "a" or "b" while waiting for a click on the map
  interp: null,         // the last interpolation from the server (+ key: what it was made for)
  interpRun: 0,         // number of the latest interpolation request
};

const labelOf = (i) => state.result.labels[state.rows.get(i)];
const isGrid = () => state.result && state.result.projection === "clusters";

const settings = () => ({
  model: $("model").value,
  dataset: $("dataset").value,
  max_images: Number($("max-images").value),
  projection: $("projection").value,
  perplexity: Number($("perplexity").value),
  neighbors: Number($("neighbors").value),
  prep: $("prep").value,
  clustering: $("clustering").value,
  cluster_on: $("cluster-on").value,
  k: Number($("k").value),
  eps: Number($("eps").value),
  min_samples: Number($("min-samples").value),
});

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${path}: ${response.status} ${await response.text()}`);
  return response.json();
}

const postJSON = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const query = (params) => new URLSearchParams(params).toString();
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function setProgress(fraction) {
  $("progress").hidden = fraction === null;
  $("progress-fill").style.width = `${Math.round((fraction || 0) * 100)}%`;
}

// ---------------------------------------------------------------------------
// Running a map
// ---------------------------------------------------------------------------

let runTimer = null;
function scheduleRun() {
  clearTimeout(runTimer);
  runTimer = setTimeout(startRun, 250);
}

async function startRun() {
  const s = settings();
  if (!s.model || !s.dataset) return;
  const run = ++state.run;
  setStatus("starting…");
  setProgress(0);
  try {
    const { job } = await postJSON("/api/run", s);
    while (run === state.run) {
      const status = await api(`/api/job/${job}`);
      if (run !== state.run) return;       // a newer run replaced this one
      if (status.error) throw new Error(status.error);
      if (status.finished) {
        if (status.result) showResult(status.result);
        return;
      }
      setStatus(status.stage);
      setProgress(status.progress);
      await sleep(250);
    }
  } catch (e) {
    if (run === state.run) {
      setStatus(e.message, true);
      setProgress(null);
    }
  }
}

function showResult(result) {
  // anchors (ids < 0) are the classes of a classifier: start over when the classes differ
  const classes = JSON.stringify(result.class_names);
  if (state.classes !== classes) {
    for (const [i, el] of state.items) {
      if (i < 0) { el.remove(); state.items.delete(i); }
    }
    state.classes = classes;
  }
  if (state.dataset !== result.dataset) {
    $("items").innerHTML = "";
    state.items.clear();
    state.coords.clear();
    state.dataset = result.dataset;
    state.view = { scale: 1, tx: 0, ty: 0 };
    state.hovered = null;
    setPick("a", null);
    setPick("b", null);
  }
  // A grid has its own fixed orientation: no aligning to it or from it
  let xy = result.xy;
  const grids = result.projection === "clusters" || isGrid();
  if ($("align").checked && !grids && state.coords.size) xy = align(state.coords, result.indices, xy);

  state.result = result;
  state.rows = new Map(result.indices.map((i, row) => [i, row]));
  state.coords = new Map(result.indices.map((i, row) => [i, xy[row]]));
  quantizeCoords();
  state.vectors.clear();
  if (state.focus !== null && !result.labels.includes(state.focus)) state.focus = null;

  syncItems(result);
  updateInterpolation(result);
  drawLegend();
  $("empty").hidden = true;
  setProgress(null);
  const clusters = new Set(result.labels.filter((l) => l >= 0)).size;
  const noise = result.labels.filter((l) => l < 0).length;
  const seconds = Object.entries(result.seconds).filter(([, v]) => v >= 0.05)
    .map(([k, v]) => `${k} ${v}s`).join(", ");
  const images = result.indices.filter((i) => i >= 0).length;     // without the class anchors
  const count = images < result.total ? `${images} of ${result.total}` : result.total;
  const prep = state.options.preps[result.prep] + (result.prep_is_default ? " (default)" : "");
  setStatus(`${count} images, ${result.dimensions} dimensions, ${prep}, ${clusters} clusters`
    + (noise ? `, ${noise} noise` : "") + (seconds ? ` (${seconds})` : ""));
  if (state.hovered !== null && state.rows.has(state.hovered)) showInfo(state.hovered);
}

// Rotate (and maybe mirror) the new map so it lines up with the previous one
// as well as possible, using the images that are on both. t-SNE, UMAP and PCA
// don't care which way is up, so without this the images would spin around
// for no reason. (2D orthogonal Procrustes: the best rotation angle has a
// closed form.)
function align(previous, indices, next) {
  const shared = indices.map((i, row) => [i, row]).filter(([i]) => previous.has(i));
  if (shared.length < 3) return next;
  const mean = (pts) => [0, 1].map((k) => pts.reduce((s, p) => s + p[k], 0) / pts.length);
  const [ax, ay] = mean(shared.map(([i]) => previous.get(i)));
  const [bx, by] = mean(shared.map(([, row]) => next[row]));
  const A = shared.map(([i]) => [previous.get(i)[0] - ax, previous.get(i)[1] - ay]);
  const B = shared.map(([, row]) => [next[row][0] - bx, next[row][1] - by]);
  const best = (mirror) => {
    let a = 0, b = 0;
    A.forEach(([x1, y1], k) => {
      const x2 = mirror * B[k][0], y2 = B[k][1];
      a += x1 * x2 + y1 * y2;
      b += y1 * x2 - x1 * y2;
    });
    return { mirror, angle: Math.atan2(b, a), score: Math.hypot(a, b) };
  };
  const plain = best(1), mirrored = best(-1);
  const { mirror, angle } = mirrored.score > plain.score ? mirrored : plain;
  const c = Math.cos(angle), s = Math.sin(angle);
  return toUnitSquare(next.map(([x, y]) => {
    const u = mirror * (x - bx), v = y - by;
    return [c * u - s * v, s * u + c * v];
  }));
}

// Same as to_unit_square on the server: centered in 0..1, proportions kept
function toUnitSquare(pts) {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const extent = Math.max(...pts.map(([x, y]) => Math.max(Math.abs(x - cx), Math.abs(y - cy)))) || 1;
  return pts.map(([x, y]) => [(x - cx) / (2 * extent) + 0.5, (y - cy) / (2 * extent) + 0.5]);
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

const MARGIN = 0.05;    // fraction of the map left empty around the images

// In the clusters grid every image fills its cell (with a small gap) and the
// size slider is ignored; elsewhere the slider sets it.
function itemSize() {
  const thumbnails = $("thumbnails").checked;
  if (isQuantized()) {
    // a tile is one cell of the grid, minus the gap: fixed in map units, so it zooms with the map
    const cell = (1 - 2 * MARGIN) * $("map").clientWidth * state.view.scale / Number($("subdivisions").value);
    return Math.max(1, cell * (1 - Number($("gap").value)) * (thumbnails ? 1 : 0.5));
  }
  if (isGrid()) {
    const cell = state.result.cell * (1 - 2 * MARGIN) * $("map").clientWidth * state.view.scale;
    return Math.max(2, cell * (thumbnails ? 0.92 : 0.5));
  }
  const size = Number($("item-size").value);
  return thumbnails ? size : Math.max(5, Math.round(size / 3));
}

// map coordinates (0..1) -> pixels, with zoom and pan
function toScreen([fx, fy]) {
  const side = $("map").clientWidth;
  const { scale, tx, ty } = state.view;
  return [(MARGIN + fx * (1 - 2 * MARGIN)) * side * scale + tx,
          (MARGIN + fy * (1 - 2 * MARGIN)) * side * scale + ty];
}

const position = (i) => toScreen(state.shown.get(i));

// ---------------------------------------------------------------------------
// Quantized tiles
// ---------------------------------------------------------------------------
//
// The map (0..1 along its longest side) is cut into `subdivisions` x
// `subdivisions` square cells and every image snaps to the center of its cell.
// Two images can't share a cell, they would hide each other: the images
// closest to the center of their cell keep it, the others move to the nearest
// free cell (searched in growing square rings around their own). With gap 0
// the tiles touch: a mosaic of the projection.

const isQuantized = () => $("quantize").checked && !isGrid();

function quantizeCoords() {
  if (!isQuantized()) {
    state.shown = state.coords;
    return;
  }
  const n = Number($("subdivisions").value);
  const cellOf = (v) => Math.min(n - 1, Math.max(0, Math.floor(v * n)));
  const center = (k) => (k + 0.5) / n;
  const wanted = [...state.coords].map(([i, [x, y]]) => {
    const cx = cellOf(x), cy = cellOf(y);
    return { i, x, y, cx, cy, d: Math.hypot(x - center(cx), y - center(cy)) };
  });
  wanted.sort((a, b) => a.d - b.d);
  const taken = new Set();
  state.shown = new Map();
  for (const w of wanted) {
    let best = null;
    // ring r: the cells whose larger offset from the wanted cell is r
    for (let r = 0; !best; r++) {
      for (let dx = -r; dx <= r; dx++) {
        for (let dy = -r; dy <= r; dy++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
          const cx = w.cx + dx, cy = w.cy + dy;
          if (taken.has(`${cx},${cy}`)) continue;
          const d = Math.hypot(w.x - center(cx), w.y - center(cy));
          if (!best || d < best.d) best = { cx, cy, d };
        }
      }
    }
    taken.add(`${best.cx},${best.cy}`);
    state.shown.set(w.i, [center(best.cx), center(best.cy)]);
  }
}

function layout() {
  const thumbnails = $("thumbnails").checked;
  const size = itemSize();
  for (const [i, el] of state.items) {
    const anchor = i < 0;
    el.classList.toggle("dot", !thumbnails && !anchor);
    // thumbnails are only requested once they are shown
    if (thumbnails && !anchor && !el.dataset.loaded) {
      el.style.backgroundImage = `url(/api/thumbnail?${query({ dataset: state.dataset, i })})`;
      el.dataset.loaded = "1";
    }
    el.classList.toggle("dim", state.focus !== null && labelOf(i) !== state.focus);
    el.classList.toggle("picked", i === state.pick.a || i === state.pick.b);
    const [x, y] = position(i);
    // anchors stay big enough to read their class name (except in the grids)
    const s = anchor && !isGrid() && !isQuantized() ? Math.max(size, 44) : size;
    el.style.width = el.style.height = `${s}px`;
    if (anchor) el.style.fontSize = `${Math.max(5, Math.min(11, s / 5))}px`;
    el.style.transform = `translate(${x - s / 2}px, ${y - s / 2}px)`;
  }
  layoutInterpolation();
}

// Make the elements match the result's images: remove the ones no longer
// used, add new ones in the middle of the map, then everything glides to its
// place.
function syncItems(result) {
  for (const [i, el] of state.items) {
    if (!state.rows.has(i)) { el.remove(); state.items.delete(i); }
  }
  const side = $("map").clientWidth;
  const added = [];
  result.indices.forEach((i, row) => {
    let el = state.items.get(i);
    if (!el) {
      el = document.createElement("div");
      el.className = "item entering" + (i < 0 ? " anchor" : "");
      el.dataset.i = i;
      el.title = i < 0 ? `class anchor: ${result.names[row]}` : result.names[row];
      if (i < 0) el.textContent = result.names[row];
      el.style.transform = `translate(${side / 2}px, ${side / 2}px)`;
      $("items").appendChild(el);
      state.items.set(i, el);
      added.push(el);
    }
    el.style.setProperty("--c", colorOf(result.labels[row]));
    el.classList.toggle("has-sound", Boolean(result.sounds && result.sounds[row]));
  });
  if (!added.length) {
    layout();
    return;
  }
  // wait until the new elements are drawn in the middle, so they glide out
  requestAnimationFrame(() => requestAnimationFrame(() => {
    added.forEach((el) => el.classList.remove("entering"));
    layout();
  }));
}

// Zoom with the wheel (around the mouse), pan by dragging, double-click resets.
// The transitions are switched off meanwhile so the map follows the mouse.
const map = $("map");
let instantTimer = null;
function instantly(fn) {
  map.classList.add("instant");
  fn();
  clearTimeout(instantTimer);
  instantTimer = setTimeout(() => map.classList.remove("instant"), 100);
}

map.addEventListener("wheel", (event) => {
  event.preventDefault();
  const r = map.getBoundingClientRect();
  const mx = event.clientX - r.left, my = event.clientY - r.top;
  const v = state.view;
  const factor = Math.exp(-event.deltaY * 0.0015);
  const scale = Math.min(40, Math.max(0.5, v.scale * factor));
  const f = scale / v.scale;
  state.view = { scale, tx: mx - (mx - v.tx) * f, ty: my - (my - v.ty) * f };
  instantly(layout);
}, { passive: false });

let drag = null;
let dragged = false;    // the last press moved the map: its click is not a pick
map.addEventListener("pointerdown", (event) => {
  drag = { x: event.clientX, y: event.clientY, tx: state.view.tx, ty: state.view.ty };
  dragged = false;
  map.setPointerCapture(event.pointerId);
});
map.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
  if (!map.classList.contains("panning") && Math.hypot(dx, dy) < 3) return;
  dragged = true;
  map.classList.add("panning");
  state.view = { ...state.view, tx: drag.tx + dx, ty: drag.ty + dy };
  layout();
});
const endDrag = () => { drag = null; map.classList.remove("panning"); };
map.addEventListener("pointerup", endDrag);
map.addEventListener("pointercancel", endDrag);
map.addEventListener("dblclick", () => {
  state.view = { scale: 1, tx: 0, ty: 0 };
  layout();
});

// The items are found by event delegation: one listener for all of them
let hoverTimer = null;
map.addEventListener("mouseover", (event) => {
  const el = event.target.closest(".item");
  if (!el || map.classList.contains("panning")) return;
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    if (el.dataset.interp !== undefined) showInterpolatedInfo(Number(el.dataset.interp));
    else showInfo(Number(el.dataset.i));
  }, 60);
});

// One sound at a time: a new click stops the one playing. The image playing
// is outlined until its sound ends.
const player = { audio: new Audio(), el: null };
player.audio.addEventListener("ended", () => player.el && player.el.classList.remove("playing"));

function soundURL(i) {
  return `/api/sound?${query({ dataset: state.dataset, i })}`;
}

function playSound(i, el) {
  player.audio.pause();
  if (player.el) player.el.classList.remove("playing");
  player.el = el;
  el.classList.add("playing");
  player.audio.src = soundURL(i);
  player.audio.play().catch((e) => setStatus(`can't play the sound: ${e.message}`, true));
}

// Picking an interpolation end: a click (not a drag) on an image of the map.
// The map captures the pointer while pressed, so the click's target is the map
// itself: look up what is under the mouse instead.
// Otherwise, a click on an image with a sound plays it.
map.addEventListener("click", (event) => {
  if (dragged) return;
  const el = document.elementFromPoint(event.clientX, event.clientY)?.closest(".item");
  if (!state.picking) {
    if (el && el.classList.contains("has-sound")) playSound(Number(el.dataset.i), el);
    return;
  }
  if (!el || el.classList.contains("anchor") || el.classList.contains("interp")) return;
  setPick(state.picking, Number(el.dataset.i));
  setPicking(null);
  layout();
  runInterpolation();
});

new ResizeObserver(() => instantly(layout)).observe(map);

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

// cluster -> number of images, and the clusters in legend order (noise last)
function clusterCounts() {
  const counts = new Map();
  for (const l of state.result.labels) counts.set(l, (counts.get(l) || 0) + 1);
  const order = [...counts.keys()].sort((a, b) => (a < 0) - (b < 0) || a - b);
  return { counts, order };
}

function drawLegend() {
  const { counts, order } = clusterCounts();
  $("legend").innerHTML = "";
  if (order.length < 2) return;
  for (const label of order) {
    const button = document.createElement("button");
    button.className = "legend-item" + (state.focus === label ? " active" : "");
    button.innerHTML = `<span class="swatch" style="background:${colorOf(label)}"></span>`
      + `${label < 0 ? "noise" : label + 1} <span class="note">(${counts.get(label)})</span>`;
    button.title = "highlight this cluster";
    button.addEventListener("click", () => {
      state.focus = state.focus === label ? null : label;
      drawLegend();
      layout();
    });
    $("legend").appendChild(button);
  }
}

// ---------------------------------------------------------------------------
// Side panel: the hovered image
// ---------------------------------------------------------------------------

async function showInfo(i) {
  const result = state.result;
  if (!result) return;
  state.hovered = i;
  const params = { model: result.model, dataset: result.dataset, i };
  const anchor = i < 0;
  $("info-empty").hidden = true;
  $("info-body").hidden = false;
  document.querySelector("#info .pair").hidden = anchor;
  if (!anchor) {
    $("info-image").src = `/api/preview?${query(params)}`;
    $("info-recon-figure").hidden = !result.reconstructs;
    if (result.reconstructs) $("info-recon").src = `/api/reconstruction?${query(params)}`;
  }
  const name = result.names[state.rows.get(i)];
  $("info-name").textContent = anchor ? `class anchor: ${name}` : name;
  // the image's sound, if it has one
  const hasSound = Boolean(result.sounds && result.sounds[state.rows.get(i)]);
  $("info-audio").hidden = !hasSound;
  if (hasSound && $("info-audio").dataset.src !== soundURL(i)) {
    $("info-audio").dataset.src = $("info-audio").src = soundURL(i);
  }
  const label = labelOf(i);
  $("info-cluster").innerHTML = `<span class="swatch" style="background:${colorOf(label)}"></span>`
    + (label < 0 ? "noise" : `cluster ${label + 1}`);

  if (!state.vectors.has(i)) {
    let values;
    if (anchor) {
      // an anchor's logits: anchor_logit for its class, 0 for the others
      values = result.class_names.map((_, c) => (c === -i - 1 ? result.anchor_logit : 0));
    } else {
      values = (await api(`/api/embedding?${query(params)}`)).values;
      if (state.result !== result) return;
    }
    state.vectors.set(i, values);
  }
  if (state.hovered === i) showVector(state.vectors.get(i));
}

function showVector(values) {
  state.shownVector = values;
  const classes = state.result.class_names;
  $("info-classes").hidden = !classes;
  if (!classes) {
    drawBarcode(values);
    return;
  }
  const scores = softmax(values);
  if ($("barcode-raw").checked) drawBarcode(values);
  else drawSoftmaxBarcode(scores);
  const best = scores.indexOf(Math.max(...scores));
  $("info-classes").innerHTML = scores.map((score, c) => `
    <div class="bar-row${c === best ? " best" : ""}">
      <div class="label"><span>${classes[c]}</span><span>${score.toFixed(2)}</span></div>
      <div class="track"><div class="fill" style="width:${(score * 100).toFixed(1)}%"></div></div>
    </div>`).join("");
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Classifier: one stripe per class, its softmax score from 0 (light) to 1 (dark)
function drawSoftmaxBarcode(scores) {
  const canvas = $("barcode");
  const ctx = canvas.getContext("2d");
  const width = canvas.width / scores.length;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  scores.forEach((s, c) => {
    const [r, g, b] = MIDDLE.map((m, k) => Math.round(m + s * (SCORE[k] - m)));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(Math.floor(c * width), 0, Math.ceil(width), canvas.height);
  });
  $("barcode-label").textContent = `softmax over ${scores.length} classes`;
  document.querySelector(".barcode-scale span:first-child").textContent = "0";
  document.querySelector(".barcode-scale span:last-child").textContent = "1";
}

// One stripe per dimension of the vector. z-score: how far each number is
// from the dataset average of that number, in standard deviations (blue below,
// red above). Raw: the numbers themselves, scaled by the largest |value|.
function drawBarcode(values) {
  const canvas = $("barcode");
  const ctx = canvas.getContext("2d");
  const raw = $("barcode-raw").checked;
  const { dim_mean: mean, dim_std: std } = state.result;
  const peak = Math.max(...values.map(Math.abs)) || 1;
  const width = canvas.width / values.length;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  values.forEach((v, d) => {
    const t = raw ? v / peak : (v - mean[d]) / (std[d] || 1) / 2;
    ctx.fillStyle = diverging(t);
    ctx.fillRect(Math.floor(d * width), 0, Math.ceil(width), canvas.height);
  });
  $("barcode-label").textContent = raw
    ? `${values.length} raw values`
    : `${values.length} dims vs. dataset mean`;
  document.querySelector(".barcode-scale span:first-child").textContent = raw ? `−${peak.toFixed(2)}` : "−2σ";
  document.querySelector(".barcode-scale span:last-child").textContent = raw ? `+${peak.toFixed(2)}` : "+2σ";
}

function diverging(t) {
  t = Math.max(-1, Math.min(1, t));
  const end = t < 0 ? NEGATIVE : POSITIVE;
  const a = Math.abs(t);
  const [r, g, b] = MIDDLE.map((m, k) => Math.round(m + a * (end[k] - m)));
  return `rgb(${r},${g},${b})`;
}

// ---------------------------------------------------------------------------
// Interpolation (autoencoders)
// ---------------------------------------------------------------------------
//
// The server walks in a straight line from the code of A to the code of B,
// decodes the points in between into images and encodes those again. Each
// one comes with its nearest dataset images and weights: it is drawn at their
// weighted average position on this map, so it follows the map through every
// projection, alignment, zoom and pan.

const INTERP_HINT = "Click A or B, then an image on the map. The codes in between are decoded, "
  + "encoded again and placed among their nearest images.";

function setPicking(which) {
  state.picking = which;
  $("pick-a").classList.toggle("active", which === "a");
  $("pick-b").classList.toggle("active", which === "b");
  map.classList.toggle("picking", which !== null);
  $("interp-hint").textContent = which ? `Click an image on the map for ${which.toUpperCase()}.` : INTERP_HINT;
}

function setPick(which, i) {
  state.pick[which] = i;
  const button = $(`pick-${which}`);
  const img = button.querySelector("img");
  img.hidden = i === null;
  button.querySelector("span").hidden = i !== null;
  if (i !== null) {
    img.src = `/api/thumbnail?${query({ dataset: state.dataset, i })}`;
    button.title = state.result.names[state.rows.get(i)];
  }
}

// What an interpolation depends on: another model, vector preparation or set
// of images on the map needs a new one; another projection or clustering not.
const interpKey = (result) => [result.model, result.dataset, result.prep, result.indices.length,
  state.pick.a, state.pick.b, $("interp-steps").value].join("|");

function updateInterpolation(result) {
  $("interp-panel").hidden = !result.reconstructs;
  if (!result.reconstructs) setPicking(null);
  for (const which of ["a", "b"]) {
    if (state.pick[which] !== null && !state.rows.has(state.pick[which])) setPick(which, null);
  }
  if (!result.reconstructs || state.pick.a === null || state.pick.b === null) clearInterpolation();
  else if (!state.interp || state.interp.key !== interpKey(result)) runInterpolation();
}

function clearInterpolation() {
  state.interp = null;
  state.interpRun++;        // an answer still on its way is ignored
  $("interp-items").innerHTML = "";
  document.querySelector("#interp-line polyline").setAttribute("points", "");
  stopAnimation();
}

// ---------------------------------------------------------------------------
// Interpolation, animated: decoded A -> the images in between -> decoded B,
// then back (ping-pong). The image on the map that is showing gets outlined.
// ---------------------------------------------------------------------------

const anim = { frames: [], k: 0, step: 1, timer: null, paused: false };

function startAnimation(data) {
  stopAnimation();
  const urls = [data.a_image, ...data.steps.map((s) => s.image), data.b_image];
  // load every frame before playing, so the first round doesn't flicker
  anim.frames = urls.map((url) => Object.assign(new Image(), { src: url }));
  anim.labels = ["A (decoded)", ...data.steps.map((s, k) => `${k + 1} / ${data.steps.length}  t = ${s.t}`),
                 "B (decoded)"];
  anim.k = 0;
  anim.step = 1;
  $("interp-anim").hidden = false;
  showFrame();
  scheduleFrame();
}

function stopAnimation() {
  clearTimeout(anim.timer);
  anim.timer = null;
  $("interp-anim").hidden = true;
  markCurrentInterpolated(null);
}

function scheduleFrame() {
  clearTimeout(anim.timer);
  if (anim.paused || !anim.frames.length) return;
  anim.timer = setTimeout(() => {
    // turn around at both ends (with a single frame, stay)
    if (anim.frames.length > 1) {
      if (anim.k + anim.step < 0 || anim.k + anim.step >= anim.frames.length) anim.step = -anim.step;
      anim.k += anim.step;
    }
    showFrame();
    scheduleFrame();
  }, 1000 / Number($("interp-fps").value));
}

function showFrame() {
  $("interp-frame").src = anim.frames[anim.k].src;
  $("interp-frame-label").textContent = anim.labels[anim.k];
  // frames 1 .. n are the images in between, drawn on the map
  markCurrentInterpolated(anim.k >= 1 && anim.k <= anim.frames.length - 2 ? anim.k - 1 : null);
}

function markCurrentInterpolated(k) {
  if (!state.interp || !state.interp.elements) return;
  state.interp.elements.forEach((el, j) => el.classList.toggle("current", j === k));
}

$("interp-frame").addEventListener("click", () => {
  anim.paused = !anim.paused;
  $("interp-frame").classList.toggle("paused", anim.paused);
  scheduleFrame();
});
$("interp-fps").addEventListener("input", scheduleFrame);

async function runInterpolation() {
  const result = state.result;
  const { a, b } = state.pick;
  if (!result || !result.reconstructs || a === null || b === null) return;
  const run = ++state.interpRun;
  const key = interpKey(result);
  $("interp-hint").textContent = "interpolating…";
  try {
    const data = await postJSON("/api/interpolate", {
      model: result.model, dataset: result.dataset, a, b,
      steps: Number($("interp-steps").value), prep: result.prep, max_images: settings().max_images,
    });
    if (run !== state.interpRun || state.result !== result) return;
    showInterpolation({ ...data, key });
    $("interp-hint").textContent = `${data.steps.length} images between A and B. Hover one to see it.`;
  } catch (e) {
    if (run === state.interpRun) $("interp-hint").textContent = e.message;
  }
}

// Replaces the previous interpolation. The new images start on A and glide
// to their places.
function showInterpolation(data) {
  $("interp-items").innerHTML = "";
  state.interp = data;
  const [ax, ay] = position(data.a);
  data.elements = data.steps.map((step, k) => {
    const el = document.createElement("div");
    el.className = "item interp";
    el.dataset.interp = k;
    el.title = `interpolation ${k + 1}/${data.steps.length}, t = ${step.t}`;
    el.style.backgroundImage = `url(${step.image})`;
    el.style.transform = `translate(${ax}px, ${ay}px)`;
    $("interp-items").appendChild(el);
    return el;
  });
  requestAnimationFrame(() => requestAnimationFrame(layoutInterpolation));
  startAnimation(data);
}

// weighted average of the neighbours' positions on the current map
function interpolatedPoint(step) {
  let x = 0, y = 0, total = 0;
  step.neighbors.forEach((i, k) => {
    const p = state.shown.get(i);       // where the neighbour is drawn (snapped, if tiles are on)
    if (!p) return;
    x += step.weights[k] * p[0];
    y += step.weights[k] * p[1];
    total += step.weights[k];
  });
  return total ? [x / total, y / total] : null;
}

function layoutInterpolation() {
  const interp = state.interp;
  const line = document.querySelector("#interp-line polyline");
  if (!interp || !interp.elements || !state.coords.has(interp.a) || !state.coords.has(interp.b)) {
    line.setAttribute("points", "");
    return;
  }
  const size = Math.max(22, itemSize() * 0.9);
  const points = [position(interp.a)];
  interp.steps.forEach((step, k) => {
    const el = interp.elements[k];
    const p = interpolatedPoint(step);
    el.hidden = !p;
    if (!p) return;
    const [x, y] = toScreen(p);
    points.push([x, y]);
    el.style.width = el.style.height = `${size}px`;
    el.style.transform = `translate(${x - size / 2}px, ${y - size / 2}px)`;
  });
  points.push(position(interp.b));
  line.setAttribute("points", points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" "));
}

function showInterpolatedInfo(k) {
  const step = state.interp.steps[k];
  state.hovered = null;
  $("info-empty").hidden = true;
  $("info-body").hidden = false;
  document.querySelector("#info .pair").hidden = false;
  $("info-image").src = step.image;
  $("info-recon-figure").hidden = true;
  $("info-audio").hidden = true;
  $("info-name").textContent = `interpolation ${k + 1}/${state.interp.steps.length} (t = ${step.t})`;
  $("info-cluster").textContent = "decoded from between A and B, vector = its encoding";
  showVector(step.vector);
}

$("pick-a").addEventListener("click", () => setPicking(state.picking === "a" ? null : "a"));
$("pick-b").addEventListener("click", () => setPicking(state.picking === "b" ? null : "b"));
$("interp-steps").addEventListener("change", runInterpolation);
$("interp-clear").addEventListener("click", () => {
  setPick("a", null);
  setPick("b", null);
  setPicking(null);
  clearInterpolation();
  layout();
});

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

const PROJECTION_HINTS = {
  tsne: "Keeps neighbours together. Distances between far-apart groups mean little.",
  umap: "Keeps neighbours together, and somewhat more of the overall layout than t-SNE.",
  pca: "The two directions of largest spread. Linear: big distances are kept, but most images crowd in the middle.",
  clusters: "A grid, one section of rows per cluster (clustered on the vectors). Left to right: "
    + "from the most representative image of the cluster (closest to its center) to the least.",
};
const CLUSTER_HINTS = {
  kmeans: "k groups around k centers. Prefers round groups of similar size.",
  spectral: "k groups that are chains of neighbours: any shape.",
  dbscan: "Dense regions become groups, their number is found by itself. Gray = noise. "
    + "eps is a distance, its scale depends on the vectors: ~0.3–0.8 when L2-normalized; "
    + "standardized, two unrelated images are ~√(2 × dimensions) apart (16 for 128), try half of that; "
    + "~0.01–0.05 on the map.",
  none: "No clustering.",
};

function updateControls() {
  const s = settings();
  $("perplexity-row").hidden = s.projection !== "tsne";
  $("neighbors-row").hidden = s.projection !== "umap";
  $("k-row").hidden = !["kmeans", "spectral"].includes(s.clustering);
  $("eps-row").hidden = $("min-samples-row").hidden = s.clustering !== "dbscan";
  // the clusters grid is always clustered on the vectors
  $("cluster-on").disabled = s.clustering === "none" || s.projection === "clusters";
  if (s.projection === "clusters") $("cluster-on").value = "embedding";
  $("align").disabled = s.projection === "clusters";
  $("item-size").disabled = s.projection === "clusters";
  $("projection-hint").textContent = PROJECTION_HINTS[s.projection] || "";
  $("cluster-hint").textContent = CLUSTER_HINTS[s.clustering] || "";
  for (const id of ["perplexity", "neighbors", "k", "subdivisions", "gap"]) $(`${id}-out`).textContent = $(id).value;
  // quantized tiles: not for the clusters grid, which is a grid already
  $("quantize").disabled = s.projection === "clusters";
  $("subdivisions-row").hidden = $("gap-row").hidden = !$("quantize").checked || s.projection === "clusters";
  $("item-size").disabled = s.projection === "clusters" || $("quantize").checked;
  updateDefaultPrep();
}

// The first entry of the vectors selector follows the model's default, named
// in brackets; the others override it.
function updateDefaultPrep() {
  if (!state.options) return;
  const model = state.options.models.find((m) => m.id === $("model").value);
  const option = $("prep").options[0];
  if (option) option.text = model ? `default (${state.options.preps[model.default_prep]})` : "default";
}

function fillSelect(select, entries, value) {
  select.innerHTML = "";
  for (const [key, text] of entries) select.add(new Option(text, key));
  if (value !== undefined && [...select.options].some((o) => o.value === value)) select.value = value;
}

// Datasets grouped by their folder ({id, label, group, count} from the
// server): in each group "all of folder/" first, then its image folders. Used
// by the three tabs.
function fillDatasetSelect(select, list, value) {
  select.innerHTML = "";
  const groups = new Map();
  for (const d of list) {
    if (!groups.has(d.group)) groups.set(d.group, []);
    groups.get(d.group).push(d);
  }
  for (const [group, entries] of groups) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = group === "." ? "datasets folder" : `${group}/`;
    for (const d of entries) optgroup.appendChild(new Option(`${d.label} (${d.count})`, d.id));
    select.appendChild(optgroup);
  }
  if (value !== undefined && list.some((d) => d.id === value)) select.value = value;
}

const KIND_LABELS ={ autoencoder: "autoencoders", classifier: "classifiers", pretrained: "pretrained" };

function fillModels(models, value) {
  const select = $("model");
  select.innerHTML = "";
  for (const [kind, title] of Object.entries(KIND_LABELS)) {
    const group = models.filter((m) => m.kind === kind);
    if (!group.length) continue;
    const optgroup = document.createElement("optgroup");
    optgroup.label = title;
    for (const m of group) optgroup.appendChild(new Option(m.label, m.id));
    select.appendChild(optgroup);
  }
  if (models.some((m) => m.id === value)) select.value = value;
}

function applyOptions(options, keep) {
  state.options = options;
  fillModels(options.models, keep.model ?? options.default_model);
  fillDatasetSelect($("dataset"), options.datasets, keep.dataset);
  if (!keep.projection) {
    fillSelect($("projection"), Object.entries(options.projections), "tsne");
    fillSelect($("clustering"), Object.entries(options.clusterings), "kmeans");
    fillSelect($("prep"), [["default", "default"], ...Object.entries(options.preps)], "default");
    $("max-images").value = options.max_images;
  }
  updateDefaultPrep();
  if (!options.datasets.length) setStatus(`no folders with images in ${options.dataset_root}`, true);
}

for (const id of ["model", "dataset", "max-images", "projection", "prep", "clustering", "cluster-on",
                  "perplexity", "neighbors", "k", "eps", "min-samples"]) {
  $(id).addEventListener("change", () => { updateControls(); scheduleRun(); });
}
for (const id of ["perplexity", "neighbors", "k"]) {
  $(id).addEventListener("input", updateControls);      // number follows the slider, run on release
}
$("thumbnails").addEventListener("change", layout);

// quantized tiles: snapping is done here, the server's map doesn't change
const requantize = () => { updateControls(); quantizeCoords(); layout(); };
$("quantize").addEventListener("change", requantize);
$("subdivisions").addEventListener("input", () => { updateControls(); });
$("subdivisions").addEventListener("change", requantize);
$("gap").addEventListener("input", () => { updateControls(); instantly(layout); });

// Display: borders and background (the background is remembered in this browser)
$("borders").addEventListener("change", () => $("map").classList.toggle("no-borders", !$("borders").checked));
function setMapBackground(color) {
  $("map").style.setProperty("--map-bg", color);
  try { localStorage.setItem("map-bg", color); } catch { /* storage blocked: fine */ }
}
$("map-bg").addEventListener("input", () => setMapBackground($("map-bg").value));
try {
  const saved = localStorage.getItem("map-bg");
  if (saved) {
    $("map-bg").value = saved;
    setMapBackground(saved);
  }
} catch { /* storage blocked: default background */ }
$("item-size").addEventListener("input", () => instantly(layout));
$("barcode-raw").addEventListener("change", () => {
  if (state.shownVector) showVector(state.shownVector);
});
$("rescan").addEventListener("click", async () => {
  setStatus("scanning folders…");
  try {
    applyOptions(await postJSON("/api/rescan", {}), settings());
    setStatus(`${state.options.models.length} models, ${state.options.datasets.length} datasets`);
  } catch (e) {
    setStatus(e.message, true);
  }
});

// ---------------------------------------------------------------------------
// Export: the map as it is shown, as a PDF (POST /api/export/pdf, map_pdf.py)
// ---------------------------------------------------------------------------

// Every mark of the map measured where it is drawn right now (zoom, pan,
// tiles...), in the order the browser paints them: by z-index, then in the
// order of the page (the interpolation line sits between the images and the
// interpolated images). Marks outside the view are left out.
function mapMarks() {
  const map = $("map");
  const box = map.getBoundingClientRect();
  const px = (value) => parseFloat(value) || 0;
  const marks = [];
  [...map.querySelectorAll(".item, #interp-line")].forEach((el, order) => {
    if (el.id === "interp-line") {
      const points = el.querySelector("polyline").getAttribute("points");
      if (points) {
        const xy = points.trim().split(/\s+/).map((p) => p.split(",").map(Number));
        marks.push({ kind: "line", points: xy, z: 0, order });
      }
      return;
    }
    const style = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    if (el.hidden || style.display === "none" || Number(style.opacity) === 0) return;
    if (r.right < box.left || r.left > box.right || r.bottom < box.top || r.top > box.bottom) return;
    const kind = el.classList.contains("anchor") ? "anchor" : el.classList.contains("dot") ? "dot" : "image";
    const mark = {
      kind, x: r.left - box.left, y: r.top - box.top, w: r.width, h: r.height,
      color: style.borderTopColor,
      border: px(style.borderTopWidth),
      ring: style.boxShadow === "none" ? 0 : 2,
      round: px(style.borderTopLeftRadius) > 0,
      outline: style.outlineStyle === "none" ? 0 : px(style.outlineWidth),
      opacity: Number(style.opacity),
      z: style.zIndex === "auto" ? 0 : Number(style.zIndex),
      order,
    };
    if (kind === "anchor") Object.assign(mark, { text: el.textContent, font_size: px(style.fontSize) });
    if (kind === "dot") mark.color = style.backgroundColor;
    if (kind === "image") {
      if (el.dataset.interp !== undefined) mark.src = state.interp.steps[Number(el.dataset.interp)].image;
      else mark.i = Number(el.dataset.i);
    }
    marks.push(mark);
  });
  marks.sort((a, b) => a.z - b.z || a.order - b.order);
  marks.forEach((m) => { delete m.z; delete m.order; });
  return marks;
}

const selectedText = (id) => $(id).selectedOptions[0]?.text || $(id).value;

function exportSpec() {
  const s = settings();
  const projection = selectedText("projection")
    + (s.projection === "tsne" ? `, perplexity ${s.perplexity}` : s.projection === "umap" ? `, neighbours ${s.neighbors}` : "");
  const clustering = s.clustering === "none" ? "no clustering"
    : selectedText("clustering") + (s.clustering === "dbscan" ? `, eps ${s.eps}, min samples ${s.min_samples}` : `, k ${s.k}`)
      + ` on ${selectedText("cluster-on")}`;
  const { counts, order } = clusterCounts();
  const legend = order.length < 2 ? [] : order.map((label) => ({
    color: colorOf(label), label: `${label < 0 ? "noise" : label + 1} (${counts.get(label)})`,
  }));
  return {
    dataset: state.dataset,
    width: $("map").clientWidth,
    height: $("map").clientHeight,
    background: getComputedStyle($("map")).backgroundColor,
    title: `${selectedText("model")}  ·  ${selectedText("dataset")}`,
    subtitle: `${projection}  ·  ${clustering}  ·  ${$("status").textContent}`,
    legend,
    marks: mapMarks(),
  };
}

$("export-pdf").addEventListener("click", async () => {
  if (!state.result) return;
  const button = $("export-pdf");
  button.disabled = true;
  button.textContent = "Exporting…";
  try {
    const response = await fetch("/api/export/pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(exportSpec()),
    });
    if (!response.ok) throw new Error(`/api/export/pdf: ${response.status} ${await response.text()}`);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = `map_${state.dataset}_${$("model").value}.pdf`.replace(/[^\w.-]+/g, "_").replace(/_+/g, "_");
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Export PDF";
  }
});

async function init() {
  applyOptions(await api("/api/options"), {});
  updateControls();
  startRun();
}

init().catch((e) => setStatus(e.message, true));
