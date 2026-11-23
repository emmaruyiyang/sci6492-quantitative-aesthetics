// Embedding Map front end.
//
// Every change of a setting asks the server for a new map (POST /api/run) and
// polls the job until it is done. The server returns, for every image of the
// dataset, a position in 0..1 x 0..1 and a cluster number. The images are
// placed with a CSS transform, and the .item transition in the stylesheet
// makes them glide from the previous map to the new one. With the three.js
// renderer (Display > renderer) there are no elements: looks() describes the
// marks as the stylesheet would draw them and map_three.js draws them in WebGL.

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
  selected: null,       // image index of the image clicked (outlined; Remove, and Pin with the springs)
  hoverMark: null,      // three.js renderer: the mark under the mouse ({ i } or { interp: k }), outlined
  run: 0,               // number of the latest run, older ones are ignored
  vectors: new Map(),   // image index -> vector of the hovered images (current result)
  shownVector: null,    // the vector in the side panel (redrawn when "raw values" changes)
  pick: { a: null, b: null },   // interpolation: image index of each end
  picking: null,        // "a" or "b" while waiting for a click on the map
  interp: null,         // the last interpolation from the server (+ key: what it was made for)
  interpRun: 0,         // number of the latest interpolation request
  queryImage: null,     // histogram ordered by an image: the server's id of the uploaded image
  applied: null,        // the Data tab's settings the map is computed with (auto update off: may lag behind)
};

let three = null;       // the three.js renderer (map_three.js) while it is the one chosen

const labelOf = (i) => state.result.labels[state.rows.get(i)];
// grid layouts (clusters, histogram): every image fills a cell of the size the result gives
const isGrid = () => Boolean(state.result && state.result.cell);
// central score layout, "fade by score distance": opacity from 1 (the middle image) down to FADE_MIN
const FADE_MIN = 0.1;
function fadeOf(row) {
  const distance = state.result.score_distance;
  return distance && $("score-fade").checked ? 1 - (1 - FADE_MIN) * distance[row] : 1;
}
const GRID_LAYOUTS = ["clusters", "histogram"];
// the grids and the selected axes have their own fixed orientation
const FIXED_ORIENTATION = ["clusters", "histogram", "axes"];

// grayscale: the server is asked for the dataset as "<id>~gray" (its images
// converted to gray as they are loaded, computed apart from the color ones)
const GRAY_SUFFIX = "~gray";
const baseDataset = (id) => (id && id.endsWith(GRAY_SUFFIX) ? id.slice(0, -GRAY_SUFFIX.length) : id);

const settings = () => ({
  model: $("model").value,
  prompt: $("prompt").value.trim(),
  dataset: $("dataset").value + ($("grayscale").checked ? GRAY_SUFFIX : ""),
  max_images: Number($("max-images").value),
  projection: $("projection").value,
  perplexity: Number($("perplexity").value),
  neighbors: Number($("neighbors").value),
  axis_x: Number($("axis-x").value || 0),
  axis_y: Number($("axis-y").value || 1),
  axis_mode: $("axis-mode").value || "component",
  axis_prompt: $("axis-prompt").value.trim(),
  axis_image: state.queryImage,
  score_layout: $("score-layout").value,
  central_top: $("central-top").value,
  central_shape: $("central-shape").value,
  radii_by_distance: $("radii-by-distance").checked,
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

// The Data tab's settings: changing them can mean embedding the images again.
// Auto update off, a change of them is only marked and Update now applies
// them; the other settings (projection, clustering) always run at once, with
// the Data tab's settings as last applied.
const DATA_SETTINGS = ["model", "prompt", "dataset", "max_images"];
const runModel = () => (state.applied ? state.applied.model : $("model").value);

let runTimer = null;
function scheduleRun(data = false) {
  clearTimeout(runTimer);
  if (data && !$("auto-update").checked) {
    $("update-now").classList.add("stale");
    if (!state.result) setStatus("Update now (Data tab) computes the map");
    return;
  }
  if (!state.applied && !$("auto-update").checked) return;     // no map yet: waits for Update now
  runTimer = setTimeout(() => startRun(data), 250);
}

// apply: the Data tab's settings as they are now (otherwise as last applied)
async function startRun(apply = false) {
  if (apply || !state.applied) {
    clearTimeout(runTimer);
    $("update-now").classList.remove("stale");
    const current = settings();
    const modelChanged = state.applied?.model !== current.model;
    state.applied = Object.fromEntries(DATA_SETTINGS.map((k) => [k, current[k]]));
    if (modelChanged) {
      fillAxes();         // the components of the model now used
      updateControls();
    }
  }
  const s = { ...settings(), ...state.applied };
  if (!s.model || !s.dataset) return;
  // the histogram by a prompt or an image: nothing to order by yet
  if (s.projection === "histogram" && s.axis_mode === "prompt" && !s.axis_prompt) {
    setStatus("type a prompt to order the images by");
    return;
  }
  if (s.projection === "histogram" && s.axis_mode === "image" && !s.axis_image) {
    setStatus("choose an image to order the images by");
    return;
  }
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
  spring.beforeResult(result);
  if (state.dataset && baseDataset(state.dataset) === baseDataset(result.dataset)) {
    // grayscale on or off: the same images, the map stays, the thumbnails are loaded again
    if (state.dataset !== result.dataset) for (const el of state.items.values()) delete el.dataset.loaded;
  } else if (state.dataset !== result.dataset) {
    $("items").innerHTML = "";
    state.items.clear();
    if (three) three.clear();
    state.coords.clear();
    state.view = { scale: 1, tx: 0, ty: 0 };
    state.hovered = null;
    setPick("a", null);
    setPick("b", null);
  }
  state.dataset = result.dataset;
  // A grid or the selected axes have their own fixed orientation: no aligning to them or from them
  let xy = result.xy;
  const fixed = FIXED_ORIENTATION.includes(result.projection)
    || (state.result && FIXED_ORIENTATION.includes(state.result.projection));
  if (result.projection === "spring") {
    // the springs start from the map as it is: images already on it stay where they are
    xy = result.indices.map((i, row) => (state.coords.has(i) ? [...state.coords.get(i)] : [...xy[row]]));
  } else if ($("align").checked && !fixed && state.coords.size) {
    xy = align(state.coords, result.indices, xy);
  }

  state.result = result;
  // the pretrained models' number of components is known from their first map
  const model = state.options.models.find((m) => m.id === result.model);
  if (model && !model.components) {
    model.components = Array.from({ length: result.dimensions }, (_, d) => String(d));
    fillAxes();
  }
  state.rows = new Map(result.indices.map((i, row) => [i, row]));
  state.coords = new Map(result.indices.map((i, row) => [i, xy[row]]));
  quantizeCoords();
  state.vectors.clear();
  if (state.focus !== null && !result.labels.includes(state.focus)) state.focus = null;
  if (state.selected !== null && !state.rows.has(state.selected)) {
    state.selected = null;      // drawn by syncItems below, the pin button by spring.onResult
    $("info-actions").hidden = true;
  }
  $("restore-removed").hidden = !result.removed;
  $("restore-removed").textContent = `Restore ${result.removed} removed`;

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
    + (result.note ? `, ${result.note}` : "")
    + (noise ? `, ${noise} noise` : "") + (seconds ? ` (${seconds})` : ""));
  if (state.hovered !== null && state.rows.has(state.hovered)) showInfo(state.hovered);
  spring.onResult(result);      // spring.js: starts (or stops) the simulation
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

// Width of the map, kept up to date by the ResizeObserver. Reading clientWidth
// inside layout(), right after moving the previous item, would make the
// browser redo the page layout once per image.
let mapSide = 0;
const mapWidth = () => mapSide || (mapSide = $("map").clientWidth);

// In the clusters grid every image fills its cell (with a small gap) and the
// size slider is ignored; elsewhere the slider sets it.
function itemSize() {
  const thumbnails = $("thumbnails").checked;
  if (isQuantized()) {
    // a tile is one cell of the grid, minus the gap: fixed in map units, so it zooms with the map
    const cell = (1 - 2 * MARGIN) * mapWidth() * state.view.scale / Number($("subdivisions").value);
    return Math.max(1, cell * (1 - Number($("gap").value)) * (thumbnails ? 1 : 0.5));
  }
  if (isGrid()) {
    const cell = state.result.cell * (1 - 2 * MARGIN) * mapWidth() * state.view.scale;
    return Math.max(2, cell * (thumbnails ? 0.92 : 0.5));
  }
  const size = Number($("item-size").value);
  return thumbnails ? size : Math.max(5, Math.round(size / 3));
}

// map coordinates (0..1) -> pixels, with zoom and pan
function toScreen([fx, fy]) {
  const side = mapWidth();
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
  if (three) {
    three.draw(looks(), {
      instant: map.classList.contains("instant") || map.classList.contains("panning"),
      background: getComputedStyle(map).backgroundColor,
      ink: INK,
    });
    return;
  }
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
    const fade = fadeOf(state.rows.get(i));
    if (fade < 1) el.style.setProperty("--fade", fade.toFixed(3));
    else el.style.removeProperty("--fade");
    el.classList.toggle("picked", i === state.pick.a || i === state.pick.b);
    el.classList.toggle("pinned", spring.isPinned(i));
    el.classList.toggle("selected", i === state.selected);
    const [x, y] = position(i);
    // anchors stay big enough to read their class name (except in the grids)
    const s = anchor && !isGrid() && !isQuantized() ? Math.max(size, 44) : size;
    el.style.width = el.style.height = `${s}px`;
    if (anchor) el.style.fontSize = `${Math.max(5, Math.min(11, s / 5))}px`;
    el.style.transform = `translate(${x - s / 2}px, ${y - s / 2}px)`;
  }
  layoutInterpolation();
  layoutPairLines();
}

// Spring forces: the lines between images and their most similar partners,
// in pixels, x1 y1 x2 y2 per line (null: none)
function pairSegments() {
  if (!state.result) return null;
  const lines = spring.lines();
  if (!lines) return null;
  const { ids } = lines;
  const points = new Float32Array(ids.length * 2);
  for (let k = 0; k < ids.length; k += 2) {
    const [x1, y1] = position(ids[k]), [x2, y2] = position(ids[k + 1]);
    points.set([x1, y1, x2, y2], k * 2);
  }
  return { points, color: lines.color };
}

function layoutPairLines() {
  const path = document.querySelector("#pair-lines path");
  const segments = pairSegments();
  if (!segments) {
    path.setAttribute("d", "");
    return;
  }
  const { points } = segments;
  const parts = [];
  for (let k = 0; k < points.length; k += 4) {
    parts.push(`M${points[k].toFixed(1)} ${points[k + 1].toFixed(1)}L${points[k + 2].toFixed(1)} ${points[k + 3].toFixed(1)}`);
  }
  path.setAttribute("d", parts.join(""));
  path.style.stroke = segments.color;
}

// Make the elements match the result's images: remove the ones no longer
// used, add new ones in the middle of the map, then everything glides to its
// place.
function syncItems(result) {
  if (three) {
    layout();       // the renderer adds and removes its marks itself
    return;
  }
  for (const [i, el] of state.items) {
    if (!state.rows.has(i)) { el.remove(); state.items.delete(i); }
  }
  const side = mapWidth();
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

// Wheel and pointer events can come several times per frame: lay out once per frame
let layoutFrame = null;
function layoutNextFrame() {
  if (layoutFrame === null) layoutFrame = requestAnimationFrame(() => { layoutFrame = null; layout(); });
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
  instantly(layoutNextFrame);
}, { passive: false });

let drag = null;
let dragged = false;    // the last press moved the map (or an image): its click is not a pick
map.addEventListener("pointerdown", (event) => {
  dragged = false;
  if (spring.grab(event)) return;       // spring forces: an image is dragged, not the map
  drag = { x: event.clientX, y: event.clientY, tx: state.view.tx, ty: state.view.ty };
  map.setPointerCapture(event.pointerId);
});
map.addEventListener("pointermove", (event) => {
  const held = spring.drag(event);
  if (held) {
    dragged ||= held === "moved";
    return;
  }
  if (!drag) return;
  const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
  if (!map.classList.contains("panning") && Math.hypot(dx, dy) < 3) return;
  dragged = true;
  map.classList.add("panning");
  state.view = { ...state.view, tx: drag.tx + dx, ty: drag.ty + dy };
  layoutNextFrame();
});
const endDrag = () => {
  drag = null;
  map.classList.remove("panning");
  spring.release();
};
map.addEventListener("pointerup", endDrag);
map.addEventListener("pointercancel", endDrag);
map.addEventListener("dblclick", () => {
  state.view = { scale: 1, tx: 0, ty: 0 };
  layout();
});

// The mark under the mouse: { i } for an image of the dataset (or a class
// anchor, i < 0), { interp: k } for an interpolated image, or null.
function markAt(clientX, clientY) {
  if (three) {
    const r = map.getBoundingClientRect();
    return three.hit(clientX - r.left, clientY - r.top);
  }
  const el = document.elementFromPoint(clientX, clientY)?.closest(".item");
  if (!el || !map.contains(el)) return null;
  return el.dataset.interp !== undefined ? { interp: Number(el.dataset.interp) } : { i: Number(el.dataset.i) };
}

const sameMark = (a, b) => a === b || Boolean(a && b && a.i === b.i && a.interp === b.interp);
const hasSound = (i) => Boolean(state.result?.sounds && state.result.sounds[state.rows.get(i)]);

let hoverTimer = null;
function hover(mark) {
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    if (mark.interp !== undefined) showInterpolatedInfo(mark.interp);
    else showInfo(mark.i);
  }, 60);
}

// HTML: the items are found by event delegation, one listener for all of them
map.addEventListener("mouseover", (event) => {
  if (three || map.classList.contains("panning")) return;
  const mark = markAt(event.clientX, event.clientY);
  if (mark) hover(mark);
});
// three.js: the canvas is one element, look up the mark whenever the mouse moves
map.addEventListener("pointermove", (event) => {
  if (!three || map.classList.contains("panning")) return;
  const mark = markAt(event.clientX, event.clientY);
  if (sameMark(mark, state.hoverMark)) return;
  state.hoverMark = mark;
  map.style.cursor = mark && mark.i !== undefined && hasSound(mark.i) && !state.picking ? "pointer" : "";
  layoutNextFrame();      // its outline
  if (mark) hover(mark);
});
map.addEventListener("pointerleave", () => {
  if (!three || !state.hoverMark) return;
  state.hoverMark = null;
  layoutNextFrame();
});

// One sound at a time: a new click stops the one playing. The image playing
// is outlined until its sound ends.
const player = { audio: new Audio(), i: null };
player.audio.addEventListener("ended", () => setPlaying(null));

function soundURL(i) {
  return `/api/sound?${query({ dataset: state.dataset, i })}`;
}

function setPlaying(i) {
  state.items.get(player.i)?.classList.remove("playing");
  player.i = i;
  state.items.get(i)?.classList.add("playing");
  if (three) layout();
}

function playSound(i) {
  player.audio.pause();
  setPlaying(i);
  player.audio.src = soundURL(i);
  player.audio.play().catch((e) => setStatus(`can't play the sound: ${e.message}`, true));
}

// Picking an interpolation end: a click (not a drag) on an image of the map.
// The map captures the pointer while pressed, so the click's target is the map
// itself: look up what is under the mouse instead.
// Otherwise, a click on an image with a sound plays it.
map.addEventListener("click", (event) => {
  if (dragged) return;
  const mark = markAt(event.clientX, event.clientY);
  if (!state.picking) {
    select(mark && mark.i !== undefined ? mark.i : null);
    if (mark && mark.i !== undefined && hasSound(mark.i)) playSound(mark.i);
    return;
  }
  if (!mark || mark.i === undefined || mark.i < 0) return;     // not an anchor or an interpolated image
  setPick(state.picking, mark.i);
  setPicking(null);
  layout();
  runInterpolation();
});

new ResizeObserver(() => {
  mapSide = map.clientWidth;
  instantly(layout);
}).observe(map);

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
  $("info-actions").hidden = anchor || i !== state.selected;     // class anchors stay
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
// Selection: an image clicked on the map. The side panel goes back to it when
// the mouse leaves the map, with a button to take it off the map. Images from
// this computer can be added (Data tab). The server keeps both changes
// (/api/set/...), every map after that is made without / with them.
// ---------------------------------------------------------------------------

function select(i) {
  state.selected = i;
  if (i !== null) showInfo(i);
  else $("info-actions").hidden = true;
  spring.selectionChanged();
  layout();
}

map.addEventListener("mouseleave", () => {
  clearTimeout(hoverTimer);
  if (state.selected !== null && state.rows.has(state.selected)) showInfo(state.selected);
});

async function removeSelected() {
  const i = state.selected;
  if (i === null || i < 0 || !state.dataset) return;
  try {
    await postJSON("/api/set/remove", { dataset: state.dataset, i });
    select(null);
    startRun();
  } catch (e) {
    setStatus(e.message, true);
  }
}

$("info-remove").addEventListener("click", removeSelected);
// Delete (or Backspace): the selected image off the map (not while typing)
document.addEventListener("keydown", (event) => {
  if (!["Delete", "Backspace"].includes(event.key) || state.selected === null) return;
  if (event.target instanceof Element && event.target.closest("input, select, textarea")) return;
  if ($("view-map").hidden) return;
  event.preventDefault();
  removeSelected();
});

$("restore-removed").addEventListener("click", async () => {
  try {
    await postJSON("/api/set/restore", { dataset: state.dataset });
    startRun();
  } catch (e) {
    setStatus(e.message, true);
  }
});

$("add-images").addEventListener("click", () => $("add-file").click());
$("add-file").addEventListener("change", async () => {
  const files = [...$("add-file").files];
  $("add-file").value = "";       // the same file can be picked again
  const dataset = state.dataset || $("dataset").value;      // the map's (a new choice may wait for Update now)
  if (!files.length || !dataset) return;
  const form = new FormData();
  form.append("dataset", dataset);
  for (const file of files) form.append("images", file);
  setStatus(`adding ${files.length} image${files.length > 1 ? "s" : ""}…`);
  try {
    await api("/api/set/add", { method: "POST", body: form });
    startRun();
  } catch (e) {
    setStatus(e.message, true);
  }
});

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
  if (!state.interp) return;
  state.interp.current = k;
  if (three) layoutNextFrame();
  else if (state.interp.elements) state.interp.elements.forEach((el, j) => el.classList.toggle("current", j === k));
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
  data.elements = null;
  if (three) {
    layout();
    startAnimation(data);
    return;
  }
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

// Where the interpolated images are drawn (pixels; null: none of its
// neighbours is on the map) and the line A -> them -> B. null when A or B is
// not on the map.
function interpolationPlaces() {
  const interp = state.interp;
  if (!interp || !state.coords.has(interp.a) || !state.coords.has(interp.b)) return null;
  const places = interp.steps.map((step) => {
    const p = interpolatedPoint(step);
    return p && toScreen(p);
  });
  return { places, points: [position(interp.a), ...places.filter(Boolean), position(interp.b)] };
}

const interpolatedSize = () => Math.max(22, itemSize() * 0.9);

function layoutInterpolation() {
  const line = document.querySelector("#interp-line polyline");
  const placed = state.interp?.elements && interpolationPlaces();
  if (!placed) {
    line.setAttribute("points", "");
    return;
  }
  const size = interpolatedSize();
  placed.places.forEach((p, k) => {
    const el = state.interp.elements[k];
    el.hidden = !p;
    if (!p) return;
    const [x, y] = p;
    el.style.width = el.style.height = `${size}px`;
    el.style.transform = `translate(${x - size / 2}px, ${y - size / 2}px)`;
  });
  line.setAttribute("points", placed.points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" "));
}

function showInterpolatedInfo(k) {
  const step = state.interp.steps[k];
  state.hovered = null;
  $("info-empty").hidden = true;
  $("info-body").hidden = false;
  $("info-actions").hidden = true;
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
// Renderer: HTML elements or three.js
// ---------------------------------------------------------------------------
//
// The three.js renderer draws marks described here, the way the stylesheet
// draws the .item elements and their classes (sizes in pixels, x and y the
// center). It glides them to new places itself. The PDF export of that map is
// made from the same descriptions.

const INK = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#000000";

function looks() {
  const marks = [];
  const result = state.result;
  if (!result) return marks;
  const thumbnails = $("thumbnails").checked;
  const borders = $("borders").checked;
  const size = itemSize();
  const center = mapWidth() / 2;
  const hovered = state.hoverMark;
  const segments = pairSegments();      // under everything
  if (segments) marks.push({ kind: "segments", points: segments.points, color: segments.color, z: -1 });
  result.indices.forEach((i, row) => {
    const anchor = i < 0;
    const [x, y] = position(i);
    const label = result.labels[row];
    const look = {
      key: anchor ? `${i}|${state.classes}` : String(i),     // other classes: new anchors
      ref: { i }, x, y, s: anchor && !isGrid() && !isQuantized() ? Math.max(size, 44) : size,
      kind: anchor ? "anchor" : thumbnails ? "image" : "dot",
      color: colorOf(label), fill: "#ffffff", border: 2, ring: 2, outline: 0, outlineOffset: 0, round: false,
      opacity: (state.focus !== null && label !== state.focus ? 0.12 : 1) * fadeOf(row), z: 0,
      start: { x: center, y: center, opacity: 0 },              // new ones come out of the middle
    };
    if (look.kind === "image") look.src = `/api/thumbnail?${query({ dataset: state.dataset, i })}`;
    if (look.kind === "dot") Object.assign(look, { fill: look.color, border: 0, round: true });
    if (anchor) {
      Object.assign(look, { fill: INK, border: 3, z: 5, text: result.names[row],
                            fontSize: Math.max(5, Math.min(11, look.s / 5)) });
    } else if (!borders) {
      look.border = look.ring = 0;
    }
    if (spring.isPinned(i)) Object.assign(look, { outline: 2, outlineOffset: 2 });
    if (hovered && hovered.i === i) Object.assign(look, { outline: 2, z: 10 });
    if (i === state.pick.a || i === state.pick.b || i === state.selected) Object.assign(look, { outline: 3, z: 15 });
    if (i === player.i) Object.assign(look, { outline: 3, z: 17 });
    marks.push(look);
  });

  const placed = interpolationPlaces();
  if (!placed) return marks;
  marks.push({ kind: "line", points: placed.points, color: INK, z: 0 });
  const s = interpolatedSize();
  const [ax, ay] = position(state.interp.a);
  placed.places.forEach((p, k) => {
    if (!p) return;
    const current = state.interp.current === k;
    marks.push({
      key: `${k}|${state.interp.key}`, ref: { interp: k }, x: p[0], y: p[1], s,
      kind: "image", src: state.interp.steps[k].image,
      color: INK, fill: "#ffffff", border: 2, ring: 2, round: true, opacity: 1,
      outline: current ? 3 : hovered && hovered.interp === k ? 2 : 0, outlineOffset: current ? 1 : 0,
      z: current ? 16 : 12,
      start: { x: ax, y: ay, opacity: 1 },                       // new ones start on A
    });
  });
  return marks;
}

// The marks in the view, in the order they are painted, as mapMarks() gives
// them for the HTML renderer
function exportMarks() {
  const w = map.clientWidth, h = map.clientHeight;
  const inView = (m) => m.x + m.s / 2 >= 0 && m.x - m.s / 2 <= w && m.y + m.s / 2 >= 0 && m.y - m.s / 2 <= h;
  return looks()
    .map((m, order) => ({ m, order }))
    .filter(({ m }) => m.kind === "line" || (m.kind !== "segments" && inView(m)))
    .sort((a, b) => a.m.z - b.m.z || a.order - b.order)
    .map(({ m }) => {
      if (m.kind === "line") return { kind: "line", points: m.points };
      const mark = {
        kind: m.kind, x: m.x - m.s / 2, y: m.y - m.s / 2, w: m.s, h: m.s,
        color: m.color, border: m.border, ring: m.ring, round: m.round, outline: m.outline, opacity: m.opacity,
      };
      if (m.kind === "anchor") Object.assign(mark, { text: m.text, font_size: m.fontSize });
      if (m.kind === "image") {
        if (m.ref.interp !== undefined) mark.src = m.src;
        else mark.i = m.ref.i;
      }
      return mark;
    });
}

// Switching renderer: the other one's marks go, the new one's come out of the
// middle of the map. three.js is only loaded once it is chosen.
async function setRenderer(kind) {
  try { localStorage.setItem("map-renderer", kind); } catch { /* storage blocked: fine */ }
  if (kind === "three" && !three) {
    try {
      const { createThreeMap } = await import("./map_three.js");
      if ($("renderer").value !== "three" || three) return;     // changed again meanwhile
      three = createThreeMap(map);
    } catch (e) {
      setStatus(`can't start three.js: ${e.message}`, true);
      $("renderer").value = "html";
      return;
    }
  } else if (kind !== "three" && three) {
    three.dispose();
    three = null;
    map.style.cursor = "";
  } else {
    return;
  }
  map.classList.toggle("webgl", Boolean(three));
  $("items").innerHTML = "";
  state.items.clear();
  state.hoverMark = null;
  if (state.result) syncItems(state.result);
  if (state.interp) showInterpolation(state.interp);
}

$("renderer").addEventListener("change", () => setRenderer($("renderer").value));
try {
  if (localStorage.getItem("map-renderer") === "three") {
    $("renderer").value = "three";
    setRenderer("three");
  }
} catch { /* storage blocked: HTML renderer */ }

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

// Sidebar tabs: Data | Projection | Clustering | Display (the tab is
// remembered in this browser). The status line and the interpolation are on
// all of them.
function setMapTab(tab) {
  document.querySelectorAll("#map-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll("#view-map section[data-tab]").forEach((s) => { s.hidden = s.dataset.tab !== tab; });
  try { localStorage.setItem("map-tab", tab); } catch { /* storage blocked: fine */ }
}
document.querySelectorAll("#map-tabs button").forEach((b) => b.addEventListener("click", () => setMapTab(b.dataset.tab)));
try {
  const saved = localStorage.getItem("map-tab");
  if (document.querySelector(`#map-tabs button[data-tab="${saved}"]`)) setMapTab(saved);
} catch { /* storage blocked: Data tab */ }

const PROJECTION_HINTS = {
  tsne: "Keeps neighbours together. Distances between far-apart groups mean little.",
  umap: "Keeps neighbours together, and somewhat more of the overall layout than t-SNE.",
  axes: "Two components of the vectors as they are, x to the right, y up. "
    + "For a classifier, the scores of two classes.",
  histogram: "One number per image. Histogram: along x, the images stacked in columns from the lowest value up "
    + "(a RankNet or an aesthetic score: worst left, best right). Central: the highest (or lowest) in the middle, "
    + "the others around it from the most to the least, in a square block or in rings. "
    + "Pretrained encoders can order by cosine similarity to a prompt (CLIP) or an image instead: most similar right.",
  pca: "The two directions of largest spread. Linear: big distances are kept, but most images crowd in the middle.",
  clusters: "A grid, one section of rows per cluster (clustered on the vectors). Left to right: "
    + "from the most representative image of the cluster (closest to its center) to the least.",
  spring: "Springs between images, computed live in the page: the more similar two images, the shorter "
    + "their spring (see rest length). Starts from the map shown before. Clustered on the vectors.",
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
  // PickScore, HPSv2: the score is how well an image fits a prompt
  $("prompt-row").hidden = !state.options?.models.find((m) => m.id === s.model)?.prompt;
  // the histogram: by a component, or by the similarity to a prompt or an image
  const histogram = s.projection === "histogram";
  const central = histogram && s.score_layout === "central";
  $("score-layout-row").hidden = !histogram;
  $("central-top-row").hidden = $("central-shape-row").hidden = !central;
  $("radii-row").hidden = !central || s.central_shape !== "radial";
  $("score-fade-row").hidden = !central;
  $("axis-mode-row").hidden = !histogram || $("axis-mode").options.length < 2;
  $("axis-prompt-row").hidden = !histogram || s.axis_mode !== "prompt";
  $("axis-image-row").hidden = !histogram || s.axis_mode !== "image";
  $("axis-x-row").hidden = !(s.projection === "axes" || (histogram && s.axis_mode === "component"));
  $("axis-y-row").hidden = s.projection !== "axes";
  $("axis-x-label").textContent = s.projection === "histogram" ? "component" : "x axis";
  $("k-row").hidden = !["kmeans", "spectral"].includes(s.clustering);
  $("eps-row").hidden = $("min-samples-row").hidden = s.clustering !== "dbscan";
  // the clusters grid and the springs are always clustered on the vectors
  const onVectors = ["clusters", "spring"].includes(s.projection);
  $("cluster-on").disabled = s.clustering === "none" || onVectors;
  if (onVectors) $("cluster-on").value = "embedding";
  $("align").disabled = FIXED_ORIENTATION.includes(s.projection) || s.projection === "spring";
  $("spring-controls").hidden = s.projection !== "spring";
  $("item-size").disabled = GRID_LAYOUTS.includes(s.projection);
  $("projection-hint").textContent = PROJECTION_HINTS[s.projection] || "";
  $("cluster-hint").textContent = CLUSTER_HINTS[s.clustering] || "";
  for (const id of ["perplexity", "neighbors", "k", "subdivisions", "gap"]) $(`${id}-out`).textContent = $(id).value;
  // quantized tiles: not for the clusters grid, which is a grid already
  $("quantize").disabled = GRID_LAYOUTS.includes(s.projection);
  $("subdivisions-row").hidden = $("gap-row").hidden = !$("quantize").checked || GRID_LAYOUTS.includes(s.projection);
  $("item-size").disabled = GRID_LAYOUTS.includes(s.projection) || $("quantize").checked;
  updateDefaultPrep();
}

// The first entry of the vectors selector follows the model's default, named
// in brackets; the others override it.
function updateDefaultPrep() {
  if (!state.options) return;
  const model = state.options.models.find((m) => m.id === runModel());
  const option = $("prep").options[0];
  if (option) option.text = model ? `default (${state.options.preps[model.default_prep]})` : "default";
}

// The x and y axis selectors list the components of the model's vectors:
// "class" names for a classifier, numbers otherwise. They keep their choice
// when the model has that component too.
function fillAxes() {
  const model = state.options?.models.find((m) => m.id === runModel());
  const components = model?.components || ["0", "1"];
  const named = ["classifier", "ranknet", "aesthetic"].includes(model?.kind);
  const entries = components.map((name, d) => [String(d), named ? `${d}: ${name}` : name]);
  for (const [id, fallback] of [["axis-x", 0], ["axis-y", 1]]) {
    const previous = $(id).value;
    fillSelect($(id), entries, previous !== "" && Number(previous) < components.length
      ? previous : String(Math.min(fallback, components.length - 1)));
  }
  // the histogram's order: a component, or (pretrained encoders) the similarity to a prompt or an image
  const modes = ["component", ...(model?.queries || [])];
  fillSelect($("axis-mode"), modes.map((m) => [m, m]), $("axis-mode").value || "component");
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

const KIND_LABELS = { autoencoder: "autoencoders", classifier: "classifiers", ranknet: "RankNets",
                      aesthetic: "aesthetic scores", pretrained: "pretrained" };

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
  fillAxes();
  if (!options.datasets.length) setStatus(`no folders with images in ${options.dataset_root}`, true);
}

for (const id of ["model", "dataset", "grayscale", "max-images", "prompt"]) {
  $(id).addEventListener("change", () => { updateControls(); scheduleRun(true); });
}
for (const id of ["projection", "prep", "clustering", "cluster-on",
                  "perplexity", "neighbors", "axis-x", "axis-y", "axis-mode", "axis-prompt",
                  "score-layout", "central-top", "central-shape", "radii-by-distance",
                  "k", "eps", "min-samples"]) {
  $(id).addEventListener("change", () => { updateControls(); scheduleRun(); });
}
for (const id of ["perplexity", "neighbors", "k"]) {
  $(id).addEventListener("input", updateControls);      // number follows the slider, run on release
}
$("thumbnails").addEventListener("change", layout);
$("score-fade").addEventListener("change", layout);      // drawn here, the map stays

// auto update: remembered in this browser (off by default)
try { $("auto-update").checked = localStorage.getItem("map-auto-update") === "on"; } catch { /* storage blocked */ }
const updateAutoUpdate = () => { $("update-now").hidden = $("auto-update").checked; };
updateAutoUpdate();
$("auto-update").addEventListener("change", () => {
  try { localStorage.setItem("map-auto-update", $("auto-update").checked ? "on" : "off"); } catch { /* storage blocked */ }
  updateAutoUpdate();
  if ($("auto-update").checked && $("update-now").classList.contains("stale")) startRun(true);   // catch up
});
$("update-now").addEventListener("click", () => startRun(true));

// the histogram ordered by an image from this computer: kept by the server
// (not in a dataset), shown here as the button's thumbnail
$("axis-image").addEventListener("click", () => $("axis-image-file").click());
$("axis-image-file").addEventListener("change", async () => {
  const file = $("axis-image-file").files[0];
  $("axis-image-file").value = "";
  if (!file) return;
  const form = new FormData();
  form.append("image", file);
  try {
    const { id } = await api("/api/query-image", { method: "POST", body: form });
    state.queryImage = id;
    const previous = $("axis-image").querySelector("img");
    if (previous) URL.revokeObjectURL(previous.src);
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    $("axis-image").replaceChildren(img);
    $("axis-image").title = `${file.name}: most similar images on the right. Click to choose another`;
    scheduleRun();
  } catch (e) {
    setStatus(e.message, true);
  }
});

// quantized tiles: snapping is done here, the server's map doesn't change
const requantize = () => { updateControls(); quantizeCoords(); layout(); };
$("quantize").addEventListener("change", requantize);
$("subdivisions").addEventListener("input", () => { updateControls(); });
$("subdivisions").addEventListener("change", requantize);
$("gap").addEventListener("input", () => { updateControls(); instantly(layout); });

// Display: borders and background (the background is remembered in this browser)
$("borders").addEventListener("change", () => {
  $("map").classList.toggle("no-borders", !$("borders").checked);
  if (three) layout();
});
function setMapBackground(color) {
  $("map").style.setProperty("--map-bg", color);
  try { localStorage.setItem("map-bg", color); } catch { /* storage blocked: fine */ }
  if (three) instantly(layoutNextFrame);       // the rings around the marks are in the map's color
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
  if (three) return exportMarks();
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
// the map's model and dataset (a new choice may be waiting for Update now)
const appliedText = (id, key) => [...$(id).options].find((o) => o.value === baseDataset(state.applied?.[key]))?.text
  || selectedText(id);

function exportSpec() {
  const s = settings();
  const projection = selectedText("projection")
    + (s.projection === "tsne" ? `, perplexity ${s.perplexity}` : s.projection === "umap" ? `, neighbours ${s.neighbors}`
      : s.projection === "axes" ? `, x ${selectedText("axis-x")}, y ${selectedText("axis-y")}`
      : s.projection === "histogram" ? (s.axis_mode === "prompt" ? `, similarity to "${s.axis_prompt}"`
        : s.axis_mode === "image" ? ", similarity to an image" : `, ${selectedText("axis-x")}`)
        + (s.score_layout === "central" ? `, central ${s.central_shape}, ${selectedText("central-top")} in the middle`
          + (s.central_shape === "radial" && s.radii_by_distance ? ", radii by distance" : "") : "") : "");
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
    title: `${appliedText("model", "model")}  ·  ${appliedText("dataset", "dataset")}`
      + (state.dataset.endsWith(GRAY_SUFFIX) ? " (grayscale)" : ""),
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
    link.download = `map_${state.dataset}_${runModel()}.pdf`.replace(/[^\w.-]+/g, "_").replace(/_+/g, "_");
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
  scheduleRun(true);
}

init().catch((e) => setStatus(e.message, true));
