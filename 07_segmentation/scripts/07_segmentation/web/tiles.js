// Tile Map front end.
//
// The image is uploaded once (POST /api/upload), the server keeps it and
// cuts it into tiles. Every change of a setting asks the server for a new map
// (POST /api/run) and polls the job until it is done. The server returns, for
// every tile, a position in 0..1 x 0..1 and a cluster number. The tiles are
// placed with a CSS transform, and the .item transition in the stylesheet
// makes them glide from the previous map to the new one.
//
// The page cuts the tiles out of the image itself (/api/image, the image as
// the server has it): on the map every tile is an element whose background is
// the whole image, scaled and shifted; next to the map a canvas puts the
// image back together from its tiles, colored by cluster.

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

// Tiles are identified by their index (row by row), anchors of a classifier's
// classes by -1, -2, ... The same tile keeps its element, and glides, as long
// as the image and the tiling stay the same.
const state = {
  options: null,
  image: null,          // the uploaded image: {token, name, width, height, url, el, gray}
  file: null,           // the file it came from, uploaded again if the server forgot it
  result: null,         // the last map from the server
  averages: null,       // its average tiles: {average, weighted} -> per cluster {img, gray}
  rows: new Map(),      // tile index -> its row in the result's arrays
  coords: new Map(),    // tile index -> [x, y], after alignment
  shown: new Map(),     // tile index -> [x, y] where it is drawn (coords, or snapped to tiles)
  items: new Map(),     // tile index -> element on the map
  tilingKey: null,      // image and tiling the elements belong to
  view: { scale: 1, tx: 0, ty: 0 },        // map zoom and pan
  tileView: { scale: 1, tx: 0, ty: 0 },    // image zoom and pan
  focus: null,          // highlighted cluster (legend click)
  hovered: null,        // tile index of the tile in the side panel
  run: 0,               // number of the latest run, older ones are ignored
  vectors: new Map(),   // tile index -> vector of the hovered tiles (current result)
  shownVector: null,    // the vector in the side panel (redrawn when "raw values" changes)
};

const labelOf = (i) => state.result.labels[state.rows.get(i)];
const isGrid = () => state.result && state.result.projection === "clusters";

const settings = () => ({
  model: $("model").value,
  token: state.image && state.image.token,
  size: Number($("tile-size").value),
  overlap: Number($("overlap").value),
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

// Errors carry the HTTP status (410: the server doesn't have the image any more)
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = (await response.text()).replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    throw Object.assign(new Error(`${path}: ${text}`), { status: response.status });
  }
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
// The image and its tiles
// ---------------------------------------------------------------------------

// Same as tiling() on the server: tile k is at column k % cols, row
// k // cols, its top left corner at (column * stride, row * stride)
function tilingFor(image, size, overlap) {
  const o = state.options;
  size = Math.max(1, Math.min(Math.max(o.min_tile, Math.round(size)), image.width, image.height));
  overlap = Math.min(Math.max(overlap, 0), o.max_overlap);
  const stride = Math.max(1, Math.round(size * (1 - overlap / 100)));
  const cols = Math.floor((image.width - size) / stride) + 1;
  const rows = Math.floor((image.height - size) / stride) + 1;
  return { size, overlap, stride, cols, rows, count: cols * rows, width: image.width, height: image.height };
}

const currentTiling = () => state.image
  && tilingFor(state.image, Number($("tile-size").value), Number($("overlap").value));
const keyOf = (token, t) => `${token}|${t.size}|${t.stride}`;
// the server's tiling, as the result has it, to ask it for one of its tiles
const tilingParams = (result) => ({ token: result.token, size: result.tiling.size, overlap: result.tiling.overlap });

// Does the last result show the image and tiling of the controls? (while a
// new one is computed, the image view shows the new tiles without clusters)
function resultIsCurrent() {
  const t = currentTiling();
  return Boolean(state.result && t && keyOf(state.result.token, state.result.tiling) === keyOf(state.image.token, t));
}

function tileCorner(t, k) {
  return [(k % t.cols) * t.stride, Math.floor(k / t.cols) * t.stride];
}

// A first tile size for a new image: about 16 tiles along its longer side
function suggestedTileSize(image) {
  const size = Math.round(Math.max(image.width, image.height) / 16 / 8) * 8;
  return Math.max(state.options.min_tile, Math.min(size, image.width, image.height));
}

// The image in gray, for the "gray + tint" coloring
function grayscale(img) {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const d = pixels.data;
  for (let p = 0; p < d.length; p += 4) {
    d[p] = d[p + 1] = d[p + 2] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  }
  ctx.putImageData(pixels, 0, 0);
  return canvas;
}

// fresh: a new image (a first tile size is suggested, the map starts over);
// otherwise the same file sent again because the server forgot it
async function uploadFile(file, fresh = true) {
  setStatus(`uploading ${file.name}…`);
  setProgress(0);
  const form = new FormData();
  form.append("file", file);
  const info = await api("/api/upload", { method: "POST", body: form });
  const el = new Image();
  el.src = `/api/image/${info.token}`;
  await el.decode();
  state.file = file;
  state.image = { ...info, url: el.src, el, gray: grayscale(el) };

  $("drop-image").src = el.src;
  $("drop-image").hidden = false;
  $("drop-text").hidden = true;
  $("drop").classList.add("has-image");
  const [w, h] = info.original;
  $("image-info").textContent = `${info.name}, ${info.width} × ${info.height} px`
    + (w !== info.width ? ` (scaled down from ${w} × ${h})` : "");
  if (fresh) {
    $("tile-size").value = suggestedTileSize(info);
    state.tileView = { scale: 1, tx: 0, ty: 0 };
  }
  clearMap();
  updateControls();
  requestDraw();
}

async function pickFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus(`${file.name} is not an image`, true);
    return;
  }
  try {
    await uploadFile(file);
    startRun();
  } catch (e) {
    setStatus(e.message, true);
    setProgress(null);
  }
}

$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", () => {
  pickFile($("file").files[0]);
  $("file").value = "";       // picking the same file again is a change too
});
// an image file can also be dropped on the frame or on the image view
for (const target of [$("drop"), $("tiles-view")]) {
  target.addEventListener("dragover", (event) => {
    event.preventDefault();
    $("drop").classList.add("dragging");
  });
  target.addEventListener("dragleave", () => $("drop").classList.remove("dragging"));
  target.addEventListener("drop", (event) => {
    event.preventDefault();
    $("drop").classList.remove("dragging");
    pickFile(event.dataTransfer.files[0]);
  });
}

function updateTilingInfo() {
  const t = currentTiling();
  $("overlap-out").textContent = $("overlap").value;
  if (!t) {
    $("tiling-info").textContent = "";
    return;
  }
  const covered = [(t.cols - 1) * t.stride + t.size, (t.rows - 1) * t.stride + t.size];
  const tooMany = t.count > state.options.max_tiles;
  $("tiling-info").textContent = `${t.cols} × ${t.rows} = ${t.count} tiles of ${t.size} px, one every ${t.stride} px`
    + (covered[0] < t.width || covered[1] < t.height ? `, covering ${covered[0]} × ${covered[1]}` : "")
    + (tooMany ? `. Too many: at most ${state.options.max_tiles}, make them bigger or overlap less.` : "");
  $("tiling-info").classList.toggle("warn", tooMany);
}

// ---------------------------------------------------------------------------
// Running a map
// ---------------------------------------------------------------------------

let runTimer = null;
function scheduleRun() {
  clearTimeout(runTimer);
  runTimer = setTimeout(startRun, 250);
}

async function startRun(retried = false) {
  const s = settings();
  if (!s.model || !s.token) return;
  const run = ++state.run;
  if (currentTiling().count > state.options.max_tiles) {
    setStatus("too many tiles", true);
    setProgress(null);
    return;
  }
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
    if (run !== state.run) return;
    // the server was restarted, or has had too many images since: send it again
    if (e.status === 410 && !retried && state.file) {
      try {
        await uploadFile(state.file, false);
        if (run === state.run) startRun(true);
        return;
      } catch (again) {
        e = again;
      }
    }
    setStatus(e.message, true);
    setProgress(null);
  }
}

// A new image: the map starts empty
function clearMap() {
  $("items").innerHTML = "";
  state.items.clear();
  state.coords.clear();
  state.result = null;
  state.tilingKey = null;
  state.hovered = null;
  state.focus = null;
  state.view = { scale: 1, tx: 0, ty: 0 };
  $("legend").innerHTML = "";
  $("info-empty").hidden = false;
  $("info-body").hidden = true;
  $("empty").hidden = false;
  $("empty").textContent = "computing…";
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
  // other tiles (image, size or overlap): other elements
  const key = keyOf(result.token, result.tiling);
  if (state.tilingKey !== key) {
    $("items").innerHTML = "";
    state.items.clear();
    state.coords.clear();
    state.tilingKey = key;
    state.hovered = null;
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

  state.averages = loadAverages(result.averages);

  syncItems(result);
  drawLegend();
  requestDraw();
  $("empty").hidden = true;
  setProgress(null);
  const clusters = new Set(result.labels.filter((l) => l >= 0)).size;
  const noise = result.labels.filter((l) => l < 0).length;
  const seconds = Object.entries(result.seconds).filter(([, v]) => v >= 0.05)
    .map(([k, v]) => `${k} ${v}s`).join(", ");
  const prep = state.options.preps[result.prep] + (result.prep_is_default ? " (default)" : "");
  setStatus(`${result.tiling.count} tiles, ${result.dimensions} dimensions, ${prep}, ${clusters} clusters`
    + (noise ? `, ${noise} noise` : "") + (seconds ? ` (${seconds})` : ""));
  if (state.hovered !== null && state.rows.has(state.hovered)) showInfo(state.hovered);
}

// Rotate (and maybe mirror) the new map so it lines up with the previous one
// as well as possible, using the tiles that are on both. t-SNE, UMAP and PCA
// don't care which way is up, so without this the tiles would spin around
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

const MARGIN = 0.05;    // fraction of a view left empty around its contents

// In the clusters grid every tile fills its cell (with a small gap) and the
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
// `subdivisions` square cells and every tile snaps to the center of its cell.
// Two tiles can't share a cell, they would hide each other: the tiles
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

// A tile's picture: the whole image as background, scaled so that a tile is
// s pixels and shifted so that this tile is the part that shows
function tilePicture(el, i, s) {
  const t = state.result.tiling;
  const f = s / t.size;
  const [x, y] = tileCorner(t, i);
  el.style.backgroundSize = `${t.width * f}px ${t.height * f}px`;
  el.style.backgroundPosition = `${-x * f}px ${-y * f}px`;
}

function layout() {
  if (!state.result) return;
  const thumbnails = $("thumbnails").checked;
  const size = itemSize();
  for (const [i, el] of state.items) {
    const anchor = i < 0;
    el.classList.toggle("dot", !thumbnails && !anchor);
    el.classList.toggle("dim", state.focus !== null && labelOf(i) !== state.focus);
    const [x, y] = position(i);
    // anchors stay big enough to read their class name (except in the grids)
    const s = anchor && !isGrid() && !isQuantized() ? Math.max(size, 44) : size;
    el.style.width = el.style.height = `${s}px`;
    if (anchor) el.style.fontSize = `${Math.max(5, Math.min(11, s / 5))}px`;
    else if (thumbnails) tilePicture(el, i, s);
    el.style.transform = `translate(${x - s / 2}px, ${y - s / 2}px)`;
  }
}

// Make the elements match the result's tiles: remove the ones no longer
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
      else el.style.backgroundImage = `url(${state.image.url})`;
      el.style.transform = `translate(${side / 2}px, ${side / 2}px)`;
      $("items").appendChild(el);
      state.items.set(i, el);
      added.push(el);
    }
    el.style.setProperty("--c", colorOf(result.labels[row]));
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

// Zoom with the wheel (around the mouse), pan by dragging, double-click
// resets: the map and the image view both. getView / setView read and write
// the view's {scale, tx, ty}, redraw(instant) draws it again.
function panZoom(el, getView, setView, redraw) {
  el.addEventListener("wheel", (event) => {
    event.preventDefault();
    const r = el.getBoundingClientRect();
    const mx = event.clientX - r.left, my = event.clientY - r.top;
    const v = getView();
    const factor = Math.exp(-event.deltaY * 0.0015);
    const scale = Math.min(40, Math.max(0.5, v.scale * factor));
    const f = scale / v.scale;
    setView({ scale, tx: mx - (mx - v.tx) * f, ty: my - (my - v.ty) * f });
    redraw(true);
  }, { passive: false });

  let drag = null;
  el.addEventListener("pointerdown", (event) => {
    const v = getView();
    drag = { x: event.clientX, y: event.clientY, tx: v.tx, ty: v.ty };
    el.setPointerCapture(event.pointerId);
  });
  el.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (!el.classList.contains("panning") && Math.hypot(dx, dy) < 3) return;
    el.classList.add("panning");
    setView({ ...getView(), tx: drag.tx + dx, ty: drag.ty + dy });
    redraw(true);
  });
  const endDrag = () => { drag = null; el.classList.remove("panning"); };
  el.addEventListener("pointerup", endDrag);
  el.addEventListener("pointercancel", endDrag);
  el.addEventListener("dblclick", () => {
    setView({ scale: 1, tx: 0, ty: 0 });
    redraw(false);
  });
}

// The transitions are switched off while zooming or panning, so the map
// follows the mouse
const map = $("map");
let instantTimer = null;
function instantly(fn) {
  map.classList.add("instant");
  fn();
  clearTimeout(instantTimer);
  instantTimer = setTimeout(() => map.classList.remove("instant"), 100);
}

panZoom(map, () => state.view, (v) => { state.view = v; }, (instant) => (instant ? instantly(layout) : layout()));

// The items are found by event delegation: one listener for all of them
let hoverTimer = null;
map.addEventListener("mouseover", (event) => {
  const el = event.target.closest(".item");
  if (!el || map.classList.contains("panning")) return;
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => showInfo(Number(el.dataset.i)), 60);
});

new ResizeObserver(() => instantly(layout)).observe(map);

// ---------------------------------------------------------------------------
// The image, put back together from its tiles
// ---------------------------------------------------------------------------
//
// keep overlaps on: every tile where it was cut out, one after the other (row
// by row), so a tile covers the parts of the previous ones it overlaps: the
// image as it is. Off: every tile whole, side by side; with overlap the
// shared regions repeat and the image looks bigger than it is.
//
// tiles: what every tile shows (noise tiles always show themselves)
//   original        itself
//   representative  the most representative tile of its cluster (closest to
//                   the cluster's center): the image as the clustering summarizes it
//   average         the average of all the cluster's tiles, pixel by pixel
//   weighted        the same, tiles closer to the cluster's center counting more
//
// The clusters are shown as
//   borders   a frame in the cluster's color (amount: its width)
//   dots      a dot instead of the tile (amount: its size)
//   squares   the tile under its cluster's color (amount: opacity)
//   tint      the tile in gray, plus the cluster's color times amount

const tv = { canvas: $("tiles-canvas"), frame: 0 };
const AMOUNT_LABELS = { borders: "width", dots: "size", squares: "opacity", tint: "tint" };
const SAVE_MAX_SIDE = 4096;

const keepOverlap = () => $("keep-overlap").checked;

// size of the put-together image, in image pixels
function compositeSize(t, keep) {
  return keep ? [(t.cols - 1) * t.stride + t.size, (t.rows - 1) * t.stride + t.size]
              : [t.cols * t.size, t.rows * t.size];
}

// where tile k goes in it
function compositeCorner(t, k, keep) {
  const pitch = keep ? t.stride : t.size;
  return [(k % t.cols) * pitch, Math.floor(k / t.cols) * pitch];
}

// image pixels -> pixels of the view: fitted in the square like the map's
// contents (same margin), then zoomed and panned
function viewFit(t, keep, side) {
  const [w, h] = compositeSize(t, keep);
  const base = side * (1 - 2 * MARGIN) / Math.max(w, h);
  const { scale, tx, ty } = state.tileView;
  return { k: base * scale, x: (side - w * base) / 2 * scale + tx, y: (side - h * base) / 2 * scale + ty };
}

// The tile drawn on top at this point of the put-together image: the last
// one (row by row) covering it
function tileAt(t, keep, x, y) {
  const [w, h] = compositeSize(t, keep);
  if (x < 0 || y < 0 || x >= w || y >= h) return null;
  const pitch = keep ? t.stride : t.size;
  return Math.min(t.rows - 1, Math.floor(y / pitch)) * t.cols + Math.min(t.cols - 1, Math.floor(x / pitch));
}

// The average tiles of a result (data URLs) as images, with their gray
// version for the tint once loaded. Not loaded yet: the tiles show themselves.
function loadAverages(urls) {
  const averages = {};
  for (const [kind, list] of Object.entries(urls)) {
    averages[kind] = list.map((url) => {
      const average = { img: new Image(), gray: null };
      average.img.onload = () => {
        average.gray = grayscale(average.img);
        requestDraw();
      };
      average.img.src = url;
      return average;
    });
  }
  return averages;
}

// What tile k of cluster `label` shows: {img, gray, sx, sy, side}, the part
// of img (or of its gray version) to draw
function tileSource(t, k, label, o) {
  const { el: img, gray } = state.image;
  let source = k;
  if (label !== null && label >= 0) {
    if (o.source === "representative") {
      source = o.representatives[label];
    } else if (o.averages) {
      const average = o.averages[label];
      if (average && average.gray) {
        return { img: average.img, gray: average.gray, sx: 0, sy: 0, side: average.img.naturalWidth };
      }
    }
  }
  const [sx, sy] = tileCorner(t, source);
  return { img, gray, sx, sy, side: t.size };
}

// Draws tiling t on ctx, fit = {k: scale, x, y offset}. labels: cluster of
// every tile, or null (no clusters yet: plain tiles). o.source: what the
// tiles show (see tileSource).
function drawComposite(ctx, t, fit, o) {
  ctx.imageSmoothingQuality = "high";
  const pitch = (o.keep ? t.stride : t.size) * fit.k;
  const mode = o.labels ? o.mode : "none";
  for (let k = 0; k < t.count; k++) {
    const [cx, cy] = compositeCorner(t, k, o.keep);
    // whole pixels, so tiles side by side leave no hairline between them
    const x = Math.round(fit.x + cx * fit.k), y = Math.round(fit.y + cy * fit.k);
    const d = Math.round(fit.x + (cx + t.size) * fit.k) - x;
    if (x > o.width || y > o.height || x + d < 0 || y + d < 0) continue;
    const label = o.labels ? o.labels[k] : null;
    const color = label === null ? null : colorOf(label);
    const dim = o.focus !== null && label !== o.focus;

    if (mode === "dots") {
      ctx.globalAlpha = dim ? 0.15 : 1;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x + d / 2, y + d / 2, pitch * (0.2 + 0.8 * o.amount) / 2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.globalAlpha = 1;
      continue;
    }
    const s = tileSource(t, k, label, o);
    ctx.drawImage(mode === "tint" || dim ? s.gray : s.img, s.sx, s.sy, s.side, s.side, x, y, d, d);
    if (dim) {
      // not in the highlighted cluster: gray, faded
      ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
      ctx.fillRect(x, y, d, d);
    } else if (mode === "squares") {
      ctx.globalAlpha = o.amount;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, d, d);
      ctx.globalAlpha = 1;
    } else if (mode === "tint") {
      ctx.globalCompositeOperation = "lighter";      // adds the color to the gray
      ctx.globalAlpha = o.amount;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, d, d);
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
    } else if (mode === "borders") {
      const width = Math.max(1, o.amount * d * 0.15);
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.strokeRect(x + width / 2, y + width / 2, d - width, d - width);
    }
  }
  // the hovered tile, outlined on top of everything
  if (o.hovered !== null && o.hovered >= 0 && o.hovered < t.count) {
    const [cx, cy] = compositeCorner(t, o.hovered, o.keep);
    const x = fit.x + cx * fit.k, y = fit.y + cy * fit.k, d = t.size * fit.k;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#000000";
    ctx.strokeRect(x - 1.5, y - 1.5, d + 3, d + 3);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#ffffff";
    ctx.strokeRect(x + 0.5, y + 0.5, d - 1, d - 1);
  }
}

// What the controls ask for, with the result's clusters if it shows these tiles
function compositeOptions(width, height, hovered) {
  const current = resultIsCurrent();
  const t = current ? state.result.tiling : currentTiling();
  return {
    t, width, height,
    keep: keepOverlap(),
    mode: $("color-mode").value,
    amount: Number($("amount").value),
    labels: current ? state.result.labels : null,
    source: current ? $("tile-source").value : "original",
    representatives: current ? state.result.representatives : null,
    averages: current && state.averages ? state.averages[$("tile-source").value] : null,
    focus: current ? state.focus : null,
    hovered: current ? hovered : null,
  };
}

// one redraw per frame, however many events ask for it
function requestDraw() {
  if (tv.frame) return;
  tv.frame = requestAnimationFrame(() => {
    tv.frame = 0;
    drawTileView();
  });
}

function drawTileView() {
  const canvas = tv.canvas;
  const side = canvas.clientWidth;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(side * ratio)) canvas.width = canvas.height = Math.round(side * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, side, side);
  $("tiles-empty").hidden = Boolean(state.image);
  if (!state.image) return;
  const o = compositeOptions(side, side, state.hovered);
  drawComposite(ctx, o.t, viewFit(o.t, o.keep, side), o);
}

const tileView = $("tiles-view");
panZoom(tileView, () => state.tileView, (v) => { state.tileView = v; }, requestDraw);
new ResizeObserver(requestDraw).observe(tileView);

// hovering a tile shows it in the side panel and outlines it on the map
tileView.addEventListener("mousemove", (event) => {
  if (!state.image || !resultIsCurrent() || tileView.classList.contains("panning")) return;
  const r = tileView.getBoundingClientRect();
  const t = state.result.tiling;
  const fit = viewFit(t, keepOverlap(), r.width);
  const k = tileAt(t, keepOverlap(), (event.clientX - r.left - fit.x) / fit.k, (event.clientY - r.top - fit.y) / fit.k);
  if (k !== null && k !== state.hovered) showInfo(k);
});

// The put-together image at its own resolution (at most SAVE_MAX_SIDE), as a PNG
$("save-png").addEventListener("click", () => {
  if (!state.image) return;
  const o = compositeOptions(0, 0, null);
  const [w, h] = compositeSize(o.t, o.keep);
  const f = Math.min(1, SAVE_MAX_SIDE / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = o.width = Math.round(w * f);
  canvas.height = o.height = Math.round(h * f);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = getComputedStyle(tileView).backgroundColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawComposite(ctx, o.t, { k: f, x: 0, y: 0 }, o);
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const name = state.image.name.replace(/\.[^.]*$/, "");
    link.download = `${name}_tiles${o.t.size}_${o.mode}${o.source !== "original" ? `_${o.source}` : ""}.png`.replace(/[^\w.-]+/g, "_");
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }, "image/png");
});

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

// cluster -> number of tiles, and the clusters in legend order (noise last)
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
      requestDraw();
    });
    $("legend").appendChild(button);
  }
}

// ---------------------------------------------------------------------------
// Side panel: the hovered tile
// ---------------------------------------------------------------------------

let linked = null;      // the map element of the tile hovered in the image view
function linkItem(i) {
  if (linked) linked.classList.remove("linked");
  linked = state.items.get(i) || null;
  if (linked) linked.classList.add("linked");
}

async function showInfo(i) {
  const result = state.result;
  if (!result) return;
  state.hovered = i;
  linkItem(i);
  requestDraw();
  const params = { model: result.model, ...tilingParams(result), i };
  const anchor = i < 0;
  $("info-empty").hidden = true;
  $("info-body").hidden = false;
  document.querySelector("#info .pair").hidden = anchor;
  if (!anchor) {
    $("info-image").src = `/api/tile?${query(params)}`;
    $("info-recon-figure").hidden = !result.reconstructs;
    if (result.reconstructs) $("info-recon").src = `/api/reconstruction?${query(params)}`;
    const t = result.tiling;
    const [x, y] = tileCorner(t, i);
    $("info-box").textContent = `x ${x}–${x + t.size}, y ${y}–${y + t.size} px`;
  } else {
    $("info-box").textContent = "";
  }
  const name = result.names[state.rows.get(i)];
  $("info-name").textContent = anchor ? `class anchor: ${name}` : name;
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
// from the image's average of that number (over all its tiles), in standard
// deviations (blue below, red above). Raw: the numbers themselves, scaled by
// the largest |value|.
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
    : `${values.length} dims vs. image mean`;
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
// Controls
// ---------------------------------------------------------------------------

const PROJECTION_HINTS = {
  tsne: "Keeps neighbours together. Distances between far-apart groups mean little.",
  umap: "Keeps neighbours together, and somewhat more of the overall layout than t-SNE.",
  pca: "The two directions of largest spread. Linear: big distances are kept, but most tiles crowd in the middle.",
  clusters: "A grid, one section of rows per cluster (clustered on the vectors). Left to right: "
    + "from the most representative tile of the cluster (closest to its center) to the least.",
};
const CLUSTER_HINTS = {
  kmeans: "k groups around k centers. Prefers round groups of similar size.",
  spectral: "k groups that are chains of neighbours: any shape.",
  dbscan: "Dense regions become groups, their number is found by itself. Gray = noise. "
    + "eps is a distance, its scale depends on the vectors: ~0.3–0.8 when L2-normalized; "
    + "standardized, two unrelated tiles are ~√(2 × dimensions) apart (16 for 128), try half of that; "
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
  $("projection-hint").textContent = PROJECTION_HINTS[s.projection] || "";
  $("cluster-hint").textContent = CLUSTER_HINTS[s.clustering] || "";
  for (const id of ["perplexity", "neighbors", "k", "subdivisions", "gap", "amount"]) $(`${id}-out`).textContent = $(id).value;
  // quantized tiles: not for the clusters grid, which is a grid already
  $("quantize").disabled = s.projection === "clusters";
  $("subdivisions-row").hidden = $("gap-row").hidden = !$("quantize").checked || s.projection === "clusters";
  $("item-size").disabled = s.projection === "clusters" || $("quantize").checked;
  const mode = $("color-mode").value;
  $("amount-row").hidden = mode === "none";
  $("amount-label").textContent = AMOUNT_LABELS[mode] || "amount";
  if (state.image) {
    $("tile-size").max = Math.min(state.image.width, state.image.height);
    $("tile-size").min = state.options.min_tile;
  }
  $("overlap").max = Math.min(90, state.options ? state.options.max_overlap : 90);
  updateTilingInfo();
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

const KIND_LABELS = { autoencoder: "autoencoders", classifier: "classifiers", pretrained: "pretrained" };

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
  if (!keep.projection) {
    fillSelect($("projection"), Object.entries(options.projections), "tsne");
    fillSelect($("clustering"), Object.entries(options.clusterings), "kmeans");
    fillSelect($("prep"), [["default", "default"], ...Object.entries(options.preps)], "default");
  }
  updateDefaultPrep();
}

for (const id of ["model", "projection", "prep", "clustering", "cluster-on",
                  "perplexity", "neighbors", "k", "eps", "min-samples"]) {
  $(id).addEventListener("change", () => { updateControls(); scheduleRun(); });
}
for (const id of ["perplexity", "neighbors", "k"]) {
  $(id).addEventListener("input", updateControls);      // number follows the slider, run on release
}

// The tiling: the image view shows the new tiles while they are dragged,
// the map follows when released
for (const id of ["tile-size", "overlap"]) {
  $(id).addEventListener("input", () => { updateTilingInfo(); requestDraw(); });
  $(id).addEventListener("change", () => { updateControls(); requestDraw(); scheduleRun(); });
}

// image view: drawn here, the server's map doesn't change
for (const id of ["keep-overlap", "tile-source", "color-mode", "amount"]) {
  $(id).addEventListener("input", () => { updateControls(); requestDraw(); });
}

$("thumbnails").addEventListener("change", layout);

// quantized tiles: snapping is done here, the server's map doesn't change
const requantize = () => { updateControls(); quantizeCoords(); layout(); };
$("quantize").addEventListener("change", requantize);
$("subdivisions").addEventListener("input", () => { updateControls(); });
$("subdivisions").addEventListener("change", requantize);
$("gap").addEventListener("input", () => { updateControls(); instantly(layout); });

// Display: borders and background of both views (the background is
// remembered in this browser)
$("borders").addEventListener("change", () => map.classList.toggle("no-borders", !$("borders").checked));
function setMapBackground(color) {
  document.documentElement.style.setProperty("--map-bg", color);
  requestDraw();
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
  setStatus("scanning models…");
  try {
    applyOptions(await postJSON("/api/rescan", {}), settings());
    setStatus(`${state.options.models.length} models`);
  } catch (e) {
    setStatus(e.message, true);
  }
});

// ---------------------------------------------------------------------------
// Export: the map as it is shown, as a PDF (POST /api/export/pdf, map_pdf.py)
// ---------------------------------------------------------------------------

// Every mark of the map measured where it is drawn right now (zoom, pan,
// tiles...), in the order the browser paints them: by z-index, then in the
// order of the page. Marks outside the view are left out.
function mapMarks() {
  const box = map.getBoundingClientRect();
  const px = (value) => parseFloat(value) || 0;
  const marks = [];
  [...map.querySelectorAll(".item")].forEach((el, order) => {
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
    if (kind === "image") mark.i = Number(el.dataset.i);
    marks.push(mark);
  });
  marks.sort((a, b) => a.z - b.z || a.order - b.order);
  marks.forEach((m) => { delete m.z; delete m.order; });
  return marks;
}

const selectedText = (id) => $(id).selectedOptions[0]?.text || $(id).value;

function exportSpec() {
  const s = settings();
  const t = state.result.tiling;
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
    ...tilingParams(state.result),
    width: map.clientWidth,
    height: map.clientHeight,
    background: getComputedStyle(map).backgroundColor,
    title: `${selectedText("model")}  ·  ${state.image.name}  ·  tiles ${t.size} px, overlap ${t.overlap}%`,
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
    link.download = `map_${state.image.name}_${$("model").value}.pdf`.replace(/[^\w.-]+/g, "_").replace(/_+/g, "_");
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
  requestDraw();
  setStatus("upload an image");
}

init().catch((e) => setStatus(e.message, true));
