// Image Remap front end.
//
// Both images are uploaded once (POST /api/upload), the server keeps them.
// Every change of the tiles asks the server where the tiles of both images
// are (POST /api/layout, boxes [x, y, w, h] row by row); Compute assignment
// starts a job (POST /api/compute) and polls it until it is done. The result
// is, for every destination tile, the source tile that goes there: the page
// draws the result itself, out of the source image (/api/image).

const $ = (id) => document.getElementById(id);

const MARGIN = 0.03;            // fraction of a view left empty around the image
const SAVE_MAX_SIDE = 4096;

const state = {
  options: null,
  images: { src: null, dst: null },     // {token, name, width, height, original, el}
  files: { src: null, dst: null },      // the files they came from, uploaded again if the server forgot them
  picking: null,                        // which image the file dialog is for
  layout: null,                         // the tiles of both images, from the server
  layoutKey: null,                      // the settings it is for
  result: null,                         // the last assignment
  resultKey: null,                      // the settings it was computed for
  uses: null,                           // how many times each source tile is used in it
  hovered: null,                        // destination tile under the mouse
  hoveredSrc: null,                     // source tile under the mouse
  view: { scale: 1, tx: 0, ty: 0 },     // result zoom and pan
  run: 0,                               // number of the latest computation, older ones are ignored
  layoutRun: 0,
};

// what decides the tiles (images, size, mode, overlap)
const settings = () => ({
  src: state.images.src && state.images.src.token,
  dst: state.images.dst && state.images.dst.token,
  size: Number($("tile-size").value),
  mode: $("mode").value,
  overlap: Number($("overlap").value),
});
const keyOf = (s) => JSON.stringify(s);
const resultIsCurrent = () => Boolean(state.result && state.layout && state.resultKey === state.layoutKey);

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

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function setProgress(fraction) {
  $("progress").hidden = fraction === null;
  $("progress-fill").style.width = `${Math.round((fraction || 0) * 100)}%`;
}

// fn(settings) -> answer; if the server forgot the images (restarted, or too
// many uploads since), they are uploaded again and fn tried once more.
// -> [the settings used, the answer]
async function withImages(fn) {
  try {
    const s = settings();
    return [s, await fn(s)];
  } catch (e) {
    if (e.status !== 410) throw e;
    for (const which of ["src", "dst"]) {
      if (state.files[which]) await upload(which, state.files[which]);
    }
    const s = settings();
    return [s, await fn(s)];
  }
}

// ---------------------------------------------------------------------------
// The two images
// ---------------------------------------------------------------------------

async function upload(which, file) {
  const form = new FormData();
  form.append("file", file);
  const info = await api("/api/upload", { method: "POST", body: form });
  const el = new Image();
  el.src = `/api/image/${info.token}`;
  await el.decode();
  state.images[which] = { ...info, el };
  state.files[which] = file;
  showThumbnail(which);
}

function showThumbnail(which) {
  const image = state.images[which];
  const zone = $(`drop-${which}`);
  const img = zone.querySelector("img");
  img.hidden = !image;
  zone.querySelector("span").hidden = Boolean(image);
  zone.classList.toggle("has-image", Boolean(image));
  if (image) img.src = image.el.src;
  zone.title = image ? `${image.name}, ${image.width} × ${image.height} px` : "click to pick an image, or drop one here";
}

function updateImagesInfo() {
  const line = (which, label) => {
    const image = state.images[which];
    if (!image) return `${label}: none`;
    const [w, h] = image.original;
    return `${label}: ${image.name}, ${image.width} × ${image.height} px`
      + (w !== image.width ? ` (scaled down from ${w} × ${h})` : "");
  };
  $("images-info").textContent = `${line("src", "source")}. ${line("dst", "destination")}.`;
}

// A first tile size for a new destination: about 32 tiles along its longer side
function suggestedTileSize(image) {
  const size = Math.round(Math.max(image.width, image.height) / 32 / 8) * 8;
  return Math.max(state.options.min_tile, Math.min(size, image.width, image.height));
}

async function pickFile(which, file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus(`${file.name} is not an image`, true);
    return;
  }
  setStatus(`uploading ${file.name}…`);
  try {
    await upload(which, file);
    if (which === "dst") $("tile-size").value = suggestedTileSize(state.images.dst);
    updateImagesInfo();
    setStatus(state.images.src && state.images.dst ? "press Compute assignment" : "pick the other image");
    await refreshLayout();
  } catch (e) {
    setStatus(e.message, true);
  }
}

for (const which of ["src", "dst"]) {
  const zone = $(`drop-${which}`);
  zone.addEventListener("click", () => {
    state.picking = which;
    $("file").click();
  });
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("dragging");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("dragging");
    pickFile(which, event.dataTransfer.files[0]);
  });
}
$("file").addEventListener("change", () => {
  pickFile(state.picking, $("file").files[0]);
  $("file").value = "";       // picking the same file again is a change too
});

$("swap").addEventListener("click", () => {
  [state.images.src, state.images.dst] = [state.images.dst, state.images.src];
  [state.files.src, state.files.dst] = [state.files.dst, state.files.src];
  showThumbnail("src");
  showThumbnail("dst");
  updateImagesInfo();
  refreshLayout();
});

// ---------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------

let layoutTimer = null;
function scheduleLayout() {
  clearTimeout(layoutTimer);
  layoutTimer = setTimeout(refreshLayout, 150);
}

async function refreshLayout() {
  if (!state.images.dst) {
    state.layout = null;
    updateTilesInfo();
    requestDraw();
    return;
  }
  const run = ++state.layoutRun;
  try {
    const [s, layout] = await withImages((s) => postJSON("/api/layout", s));
    if (run !== state.layoutRun) return;
    state.layout = layout;
    state.layoutKey = keyOf(s);
    state.hovered = state.hoveredSrc = null;
    updateTilesInfo();
    requestDraw();
  } catch (e) {
    if (run === state.layoutRun) setStatus(e.message, true);
  }
}

function updateTilesInfo() {
  const L = state.layout;
  const max = state.options.max_tiles;
  const lines = [];
  let tooMany = false;
  if (L) {
    lines.push(`destination: ${L.dst.cols} × ${L.dst.rows} = ${L.dst.count} tiles of ${L.dst.size} px`);
    tooMany = L.dst.count > max;
  }
  if (L && L.src) {
    const [w, h] = L.src.tile;
    const stretched = Math.abs(w / h - 1) > 0.02 || Math.abs(w - L.dst.size) > 0.5;
    lines.push(`source: ${L.src.cols} × ${L.src.rows} = ${L.src.count} tiles of `
      + (w === h ? `${w} px` : `${w} × ${h} px`)
      + (L.src.stride && L.src.stride !== w ? `, one every ${L.src.stride} px` : "")
      + (stretched ? " (stretched)" : ""));
    tooMany = tooMany || L.src.count > max;
  }
  if (tooMany) lines.push(`Too many tiles: at most ${max} per image, make them bigger.`);
  $("tiles-info").textContent = lines.join(". ") + (lines.length ? "." : "");
  $("tiles-info").classList.toggle("warn", tooMany);
  updateAssignmentHint();
}

// what "max uses" means with these tiles
function updateAssignmentHint() {
  const L = state.layout;
  const uses = Math.max(0, Math.round(Number($("max-uses").value) || 0));
  let text = "";
  if (L && L.src) {
    const nd = L.dst.count, ns = L.src.count;
    const k = uses && Math.max(uses, Math.ceil(nd / ns));
    if (!uses) {
      text = "Palette: every destination tile takes its nearest source tile, any number of times.";
    } else if (k === 1) {
      text = ns === nd
        ? "One to one: every source tile is used exactly once (the result is the source, rearranged)."
        : `Every source tile at most once: ${ns - nd} of the ${ns} source tiles are left out.`;
    } else {
      text = `Every source tile at most ${k} times.`;
    }
    if (k > uses && uses) text += ` (Raised from ${uses}: ${ns} source tiles are not enough for ${nd} places.)`;
  }
  $("assignment-hint").textContent = text;
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

async function compute() {
  if (!state.images.src || !state.images.dst) {
    setStatus("pick a source and a destination image", true);
    return;
  }
  const run = ++state.run;
  setStatus("starting…");
  setProgress(0);
  try {
    const [s, { job }] = await withImages((s) => postJSON("/api/compute", {
      ...s,
      model: $("model").value,
      color: Number($("color").value),
      max_uses: Math.max(0, Math.round(Number($("max-uses").value) || 0)),
    }));
    while (run === state.run) {
      const status = await api(`/api/job/${job}`);
      if (run !== state.run) return;       // a newer computation replaced this one
      if (status.error) throw new Error(status.error);
      if (status.finished) {
        if (status.result) showResult(status.result, s);
        else setProgress(null);
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

async function showResult(result, s) {
  state.result = result;
  state.resultKey = keyOf(s);
  if (state.layoutKey !== state.resultKey) await refreshLayout();     // the images were uploaded again
  const ns = state.layout.src.count;
  state.uses = new Array(ns).fill(0);
  for (const j of result.assignment) state.uses[j]++;
  setProgress(null);
  const seconds = Object.entries(result.seconds).filter(([, v]) => v >= 0.05)
    .map(([k, v]) => `${k} ${v}s`).join(", ");
  setStatus(`${result.assignment.length} tiles from ${result.used} of ${ns} source tiles`
    + (result.max_uses === 1 ? ", each at most once" : result.max_uses ? `, each at most ${result.max_uses} times` : ", no limit")
    + `, mean cost ${result.mean_cost}` + (seconds ? ` (${seconds})` : ""));
  requestDraw();
}

$("compute").addEventListener("click", compute);

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

// A w x h image in a square view of `side` pixels: {k: scale, x, y offset},
// zoomed and panned by view (the result) or not (the two images)
function fitOf(w, h, side, view = null) {
  const base = side * (1 - 2 * MARGIN) / Math.max(w, h);
  const { scale, tx, ty } = view || { scale: 1, tx: 0, ty: 0 };
  return { k: base * scale, x: (side - w * base) / 2 * scale + tx, y: (side - h * base) / 2 * scale + ty };
}

function prepareCanvas(canvas) {
  const side = canvas.clientWidth;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(side * ratio)) canvas.width = canvas.height = Math.round(side * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, side, side);
  ctx.imageSmoothingQuality = "high";
  return { ctx, side };
}

// box [x, y, w, h] in image pixels -> whole pixels on the canvas, so tiles
// side by side leave no hairline between them
function onCanvas([x, y, w, h], fit) {
  const x0 = Math.round(fit.x + x * fit.k), y0 = Math.round(fit.y + y * fit.k);
  return [x0, y0, Math.round(fit.x + (x + w) * fit.k) - x0, Math.round(fit.y + (y + h) * fit.k) - y0];
}

function drawGrid(ctx, boxes, fit) {
  if (!$("grid").checked) return;
  ctx.beginPath();
  for (const box of boxes) {
    const [x, y, w, h] = onCanvas(box, fit);
    ctx.rect(x + 0.5, y + 0.5, w, h);
  }
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
  ctx.stroke();
}

function drawOutlines(ctx, boxes, fit) {
  for (const box of boxes) {
    const [x, y, w, h] = onCanvas(box, fit);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#000000";
    ctx.strokeRect(x - 1.5, y - 1.5, w + 3, h + 3);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#ffffff";
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  }
}

// the destination tiles to outline: the hovered one, or every one that took
// the hovered source tile
function hoveredDestinations() {
  if (state.hovered !== null) return [state.hovered];
  if (state.hoveredSrc === null || !resultIsCurrent()) return [];
  return state.result.assignment.flatMap((j, k) => (j === state.hoveredSrc ? [k] : []));
}

function hoveredSource() {
  if (state.hoveredSrc !== null) return state.hoveredSrc;
  if (state.hovered !== null && resultIsCurrent()) return state.result.assignment[state.hovered];
  return null;
}

// The result: every destination tile drawn with its source tile (stretched
// into the square if it isn't one), then the destination over it at `blend`
function drawResultTiles(ctx, fit, blend) {
  const src = state.images.src.el, dst = state.images.dst.el;
  const { dst: D, src: S } = state.layout;
  state.result.assignment.forEach((j, k) => {
    const [sx, sy, sw, sh] = S.boxes[j];
    ctx.drawImage(src, sx, sy, sw, sh, ...onCanvas(D.boxes[k], fit));
  });
  if (blend > 0) {
    const w = D.cols * D.size, h = D.rows * D.size;
    ctx.globalAlpha = blend;
    ctx.drawImage(dst, 0, 0, w, h, fit.x, fit.y, w * fit.k, h * fit.k);
    ctx.globalAlpha = 1;
  }
}

function drawResult() {
  const { ctx, side } = prepareCanvas($("result-canvas"));
  const L = state.layout;
  const current = resultIsCurrent();
  $("result-empty").hidden = current;
  $("result-empty").textContent = !state.images.dst || !state.images.src
    ? "pick a source and a destination" : "press Compute assignment";
  if (!state.images.dst || !L) return;
  const w = L.dst.cols * L.dst.size, h = L.dst.rows * L.dst.size;
  const fit = fitOf(w, h, side, state.view);
  if (current) {
    drawResultTiles(ctx, fit, Number($("blend").value));
  } else {
    // nothing computed for these tiles yet: the destination, faded
    ctx.globalAlpha = 0.25;
    ctx.drawImage(state.images.dst.el, 0, 0, w, h, fit.x, fit.y, w * fit.k, h * fit.k);
    ctx.globalAlpha = 1;
  }
  drawGrid(ctx, L.dst.boxes, fit);
  drawOutlines(ctx, hoveredDestinations().map((k) => L.dst.boxes[k]), fit);
}

// One of the two images with its tiles. dim: tiles to fade (source tiles
// the result doesn't use)
function drawImageView(view, image, boxes, outlined, dim = []) {
  const { ctx, side } = prepareCanvas(view.querySelector("canvas"));
  view.querySelector(".empty").hidden = Boolean(image);
  if (!image) return;
  const fit = fitOf(image.width, image.height, side);
  ctx.drawImage(image.el, fit.x, fit.y, image.width * fit.k, image.height * fit.k);
  if (!boxes) return;
  ctx.fillStyle = "rgba(255, 255, 255, 0.65)";
  for (const k of dim) ctx.fillRect(...onCanvas(boxes[k], fit));
  drawGrid(ctx, boxes, fit);
  drawOutlines(ctx, outlined.map((k) => boxes[k]), fit);
}

function drawAll() {
  const L = state.layout;
  drawResult();
  drawImageView($("dst-view"), state.images.dst, L && L.dst.boxes, hoveredDestinations());
  const source = hoveredSource();
  const unused = resultIsCurrent() ? state.uses.flatMap((n, j) => (n ? [] : [j])) : [];
  drawImageView($("src-view"), state.images.src, L && L.src && L.src.boxes, source === null ? [] : [source], unused);
}

let frame = 0;
function requestDraw() {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    drawAll();
  });
}

for (const id of ["result-view", "dst-view", "src-view"]) new ResizeObserver(requestDraw).observe($(id));

// ---------------------------------------------------------------------------
// Hover: which tile is where
// ---------------------------------------------------------------------------

const rowColumn = (k, cols) => `row ${Math.floor(k / cols) + 1}, column ${(k % cols) + 1}`;

function setHover(k, j) {
  if (k === state.hovered && j === state.hoveredSrc) return;
  state.hovered = k;
  state.hoveredSrc = j;
  const L = state.layout;
  const current = resultIsCurrent();
  let text = " ";
  if (k !== null) {
    text = `destination tile ${k} (${rowColumn(k, L.dst.cols)})`;
    if (current) {
      const s = state.result.assignment[k];
      text += ` ← source tile ${s} (${rowColumn(s, L.src.cols)}), used ${state.uses[s]}×, cost ${state.result.cost[k]}`;
    }
  } else if (j !== null) {
    text = `source tile ${j} (${rowColumn(j, L.src.cols)})` + (current ? `, used ${state.uses[j]}×` : "");
  }
  $("hover-info").textContent = text;
  requestDraw();
}

// the mouse in image pixels of what a view shows
function mouseIn(view, event, w, h, zoomed) {
  const r = view.getBoundingClientRect();
  const fit = fitOf(w, h, r.width, zoomed ? state.view : null);
  return [(event.clientX - r.left - fit.x) / fit.k, (event.clientY - r.top - fit.y) / fit.k];
}

// destination tiles are a grid of size x size squares from the top left corner
function destinationTileAt(x, y) {
  const D = state.layout.dst;
  const c = Math.floor(x / D.size), r = Math.floor(y / D.size);
  return x >= 0 && y >= 0 && c < D.cols && r < D.rows ? r * D.cols + c : null;
}

$("result-view").addEventListener("mousemove", (event) => {
  if (!state.layout || $("result-view").classList.contains("panning")) return;
  const D = state.layout.dst;
  setHover(destinationTileAt(...mouseIn($("result-view"), event, D.cols * D.size, D.rows * D.size, true)), null);
});
$("dst-view").addEventListener("mousemove", (event) => {
  if (!state.layout) return;
  const image = state.images.dst;
  setHover(destinationTileAt(...mouseIn($("dst-view"), event, image.width, image.height, false)), null);
});
// source tiles may overlap: the one drawn last (on top) is the one under the mouse
$("src-view").addEventListener("mousemove", (event) => {
  if (!state.layout || !state.layout.src) return;
  const image = state.images.src;
  const [x, y] = mouseIn($("src-view"), event, image.width, image.height, false);
  const boxes = state.layout.src.boxes;
  let found = null;
  for (let j = boxes.length - 1; j >= 0 && found === null; j--) {
    const [bx, by, bw, bh] = boxes[j];
    if (x >= bx && x < bx + bw && y >= by && y < by + bh) found = j;
  }
  setHover(null, found);
});
for (const id of ["result-view", "dst-view", "src-view"]) {
  $(id).addEventListener("mouseleave", () => setHover(null, null));
}

// Result: zoom with the wheel (around the mouse), pan by dragging, double-click resets
const resultView = $("result-view");
resultView.addEventListener("wheel", (event) => {
  event.preventDefault();
  const r = resultView.getBoundingClientRect();
  const mx = event.clientX - r.left, my = event.clientY - r.top;
  const v = state.view;
  const scale = Math.min(40, Math.max(0.5, v.scale * Math.exp(-event.deltaY * 0.0015)));
  const f = scale / v.scale;
  state.view = { scale, tx: mx - (mx - v.tx) * f, ty: my - (my - v.ty) * f };
  requestDraw();
}, { passive: false });
let drag = null;
resultView.addEventListener("pointerdown", (event) => {
  drag = { x: event.clientX, y: event.clientY, tx: state.view.tx, ty: state.view.ty };
  resultView.setPointerCapture(event.pointerId);
});
resultView.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
  if (!resultView.classList.contains("panning") && Math.hypot(dx, dy) < 3) return;
  resultView.classList.add("panning");
  state.view = { ...state.view, tx: drag.tx + dx, ty: drag.ty + dy };
  requestDraw();
});
const endDrag = () => { drag = null; resultView.classList.remove("panning"); };
resultView.addEventListener("pointerup", endDrag);
resultView.addEventListener("pointercancel", endDrag);
resultView.addEventListener("dblclick", () => {
  state.view = { scale: 1, tx: 0, ty: 0 };
  requestDraw();
});

// The result at the destination's resolution (at most SAVE_MAX_SIDE), as a PNG
$("save-png").addEventListener("click", () => {
  if (!resultIsCurrent()) return;
  const D = state.layout.dst;
  const w = D.cols * D.size, h = D.rows * D.size;
  const f = Math.min(1, SAVE_MAX_SIDE / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * f);
  canvas.height = Math.round(h * f);
  drawResultTiles(canvas.getContext("2d"), { k: f, x: 0, y: 0 }, Number($("blend").value));
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stem = (image) => image.name.replace(/\.[^.]*$/, "");
    link.href = url;
    link.download = `${stem(state.images.dst)}_from_${stem(state.images.src)}_${D.size}px.png`.replace(/[^\w.-]+/g, "_");
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }, "image/png");
});

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function updateControls() {
  $("overlap-row").hidden = $("mode").value !== "size";
  $("overlap-out").textContent = $("overlap").value;
  $("color-out").textContent = $("color").value;
  $("blend-out").textContent = $("blend").value;
  // no model: colors & layout only
  $("color").disabled = $("model").value === state.options.pixels_model;
  updateAssignmentHint();
}

const KIND_LABELS = { autoencoder: "autoencoders", classifier: "classifiers", pretrained: "pretrained" };

function fillModels(options, value) {
  const select = $("model");
  select.innerHTML = "";
  select.add(new Option("none: colors & layout only", options.pixels_model));
  for (const [kind, title] of Object.entries(KIND_LABELS)) {
    const group = options.models.filter((m) => m.kind === kind);
    if (!group.length) continue;
    const optgroup = document.createElement("optgroup");
    optgroup.label = title;
    for (const m of group) optgroup.appendChild(new Option(m.label, m.id));
    select.appendChild(optgroup);
  }
  if ([...select.options].some((o) => o.value === value)) select.value = value;
}

$("tile-size").addEventListener("change", scheduleLayout);
$("overlap").addEventListener("input", updateControls);
$("overlap").addEventListener("change", scheduleLayout);
// the same grid usually means one to one, the same size a palette
$("mode").addEventListener("change", () => {
  $("max-uses").value = $("mode").value === "grid" ? 1 : 0;
  updateControls();
  scheduleLayout();
});
$("max-uses").addEventListener("input", updateAssignmentHint);
for (const id of ["model", "color"]) $(id).addEventListener("input", updateControls);
for (const id of ["grid", "blend"]) $(id).addEventListener("input", () => { updateControls(); requestDraw(); });

// background of the views (remembered in this browser, shared with the Tile Map)
function setBackground(color) {
  document.documentElement.style.setProperty("--map-bg", color);
  try { localStorage.setItem("map-bg", color); } catch { /* storage blocked: fine */ }
}
$("map-bg").addEventListener("input", () => setBackground($("map-bg").value));
try {
  const saved = localStorage.getItem("map-bg");
  if (saved) {
    $("map-bg").value = saved;
    setBackground(saved);
  }
} catch { /* storage blocked: default background */ }

$("rescan").addEventListener("click", async () => {
  setStatus("scanning models…");
  try {
    state.options = await postJSON("/api/rescan", {});
    fillModels(state.options, $("model").value);
    setStatus(`${state.options.models.length} models`);
  } catch (e) {
    setStatus(e.message, true);
  }
});

async function init() {
  state.options = await api("/api/options");
  fillModels(state.options, state.options.default_model);
  $("mode").innerHTML = "";
  for (const [key, text] of Object.entries(state.options.modes)) $("mode").add(new Option(text, key));
  $("overlap").max = state.options.max_overlap;
  updateControls();
  updateImagesInfo();
  requestDraw();
  setStatus("pick a source and a destination image");
}

init().catch((e) => setStatus(e.message, true));
