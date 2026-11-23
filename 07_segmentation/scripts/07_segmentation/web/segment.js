// Segmentation Explorer front end.
//
// The image and the reference images are uploaded once (POST /api/upload),
// the server keeps them. Every change of the model, the method, the prompts
// or a parameter starts a job (POST /api/run) that the page polls until it is
// done. The result is a list of maps, each an 8 bit gray PNG and the values
// its black (lo) and white (hi) stand for, maybe a segmentation (at every
// pixel the number of the map that wins there, and its value) and RGB
// pictures (PCA, an autoencoder's reconstruction).
//
// The page reads the pixels of the maps and colors them itself: the display
// settings change the view without asking the server again. Every map is
// drawn stretched over the whole image, whatever its resolution (a grid of
// patches, a squashed model input...).
//
// 3D model mode: a Rhino file's meshes (POST /api/scene/model) are rendered
// from N cameras by segment3d.js and the frames uploaded as one set (POST
// /api/scene/frames). The selected frame is the image. A sweep (POST
// /api/sweep) runs the method on every frame: one value per map and frame,
// drawn as a line per map (class, prompt...) under the view.

const $ = (id) => document.getElementById(id);

const MARGIN = 0.03;            // fraction of the view left empty around the image
const SAVE_MAX_SIDE = 4096;
const HERE_ROWS = 12;           // values listed under "Here"

// Segments: the Tile Map's cluster colors, then more hues (golden angle)
const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];

function hexRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function hslRgb(h, s, l) {
  const f = (n) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}

const labelRgb = (i) => (i < PALETTE.length ? hexRgb(PALETTE[i]) : hslRgb((i * 137.508) % 360, 0.6, 0.5));
const labelCss = (i) => `rgb(${labelRgb(i).join(",")})`;

// Color maps: 256 colors from evenly spaced stops
function lut(stops) {
  const out = [];
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (stops.length - 1);
    const a = Math.min(Math.floor(x), stops.length - 2), f = x - a;
    out.push(stops[a].map((c, j) => Math.round(c + f * (stops[a + 1][j] - c))));
  }
  return out;
}
const COLORMAPS = {
  inferno: lut([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21],
    [250, 193, 39], [252, 255, 164]]),
  viridis: lut([[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137],
    [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37]]),
  gray: lut([[0, 0, 0], [255, 255, 255]]),
  red: lut([[40, 0, 0], [255, 40, 30]]),
};
// signed maps: negative blue, 0 light gray, positive red (as the Tile Map's barcode)
const DIVERGING = lut([[42, 120, 214], [236, 236, 234], [227, 73, 72]]);

const METHOD_HELP = {
  similarity: "Every patch's vector compared with every prompt (cosine similarity). No prompt: the whole image is the prompt.",
  classes: "The classifier's head applied to every patch instead of to the whole image: one probability map per class. An approximation, the head never saw single patches (for ResNet it is the classic Class Activation Map).",
  segformer: "A model trained for segmentation: a probability for every class at every pixel (ADE20K: 150 classes of things and stuff). Only the classes that win somewhere are listed.",
  clipseg: "CLIP with a small decoder trained to draw the mask of a prompt, text or example image. Every prompt's mask is independent, 0 … 1.",
  window: "A window slides over the image; every position is scored as an image of its own and the score is spread back over the window, most in its center.",
  occlusion: "A gray square slides over the image: how much does the score drop where it hides something? Red: hiding it lowers the score (it mattered); blue: hiding it raises the score.",
  gradients: "Integrated gradients: how much every pixel adds to the score, summed along a path from a baseline image to this one. Red adds, blue takes away.",
  attention: "Where the [CLS] token of the last layer looks, one map per attention head. Not an explanation of the output, but heads often outline objects.",
  pca: "The 3 directions along which the patch vectors vary most, as red, green and blue: patches of similar color have similar vectors.",
  kmeans: "The patch vectors in k groups, no prompt needed: a segmentation without supervision. Each map: how close every patch is to one group's center.",
};
// what "the score" is, for the methods that explain one score
const SCORE_HELP = {
  classifier: "The score: each class's probability (occlusion) or logit (gradients).",
  autoencoder: "The score: similarity of the latent vector to each prompt. Window also gives the reconstruction error (where the image is unlike the training images) and the reconstruction.",
  dino: "The score: similarity of the image's vector to each prompt.",
  clip: "The score: similarity of the image's vector to each prompt.",
};

const state = {
  options: null,
  mode: "image",        // "image" or "3d"
  image: null,          // the image explored: {token, name, width, height, original, el}
  file: null,           // the file it came from, uploaded again if the server forgot it
  uploaded: null,       // image mode's image and file, kept while in 3D mode: {image, file}
  scene: null,          // 3D mode: {name, notes, frames: [{token, blob, url, camera}]}
  frame: null,          // number of the selected frame
  sweep: null,          // the last sweep: one value per frame and map
  highlight: null,      // key of the map highlighted in the graph
  references: [],       // [{token, name, el, file}]
  points: [],           // [[x, y]] in image pixels
  pointsToken: null,    // the image the points were picked on (another frame: they are looked for in this one)
  result: null,         // the last result from the server
  maps: [],             // its maps, decoded: {name, w, h, data, lo, hi, values, segment}
  segmentation: null,   // {w, h, labels, top, lo, hi}
  pictures: [],         // [{name, el}]
  selected: "image",    // "image", "segmentation", "picture:i", "map:i"
  hover: null,          // [x, y] in image pixels
  view: { scale: 1, tx: 0, ty: 0 },
  layers: new Map(),    // selection -> colored canvas, for the current display settings
  run: 0,               // number of the latest run, older ones are ignored
};

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

async function loadImage(url) {
  const el = new Image();
  el.src = url;
  await el.decode();
  return el;
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const info = await api("/api/upload", { method: "POST", body: form });
  return { ...info, el: await loadImage(`/api/image/${info.token}`), file };
}

// ---------------------------------------------------------------------------
// The image, the prompts
// ---------------------------------------------------------------------------

async function pickImage(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus(`${file.name} is not an image`, true);
    return;
  }
  setStatus(`uploading ${file.name}…`);
  try {
    state.image = await uploadFile(file);
    state.file = file;
    state.uploaded = { image: state.image, file };
    state.points = [];
    showPoints();
    state.view = { scale: 1, tx: 0, ty: 0 };
    clearResult();
    showImageInfo();
    $("window").value = suggestedWindow(state.image);
    updateControls();
    startRun();
  } catch (e) {
    setStatus(e.message, true);
  }
}

function showImageInfo() {
  const image = state.image;
  $("drop-image").src = image.el.src;
  $("drop-image").hidden = false;
  $("drop-text").hidden = true;
  $("drop").classList.add("has-image");
  const [w, h] = image.original;
  $("image-info").textContent = `${image.name}, ${image.width} × ${image.height} px`
    + (w !== image.width ? ` (scaled down from ${w} × ${h})` : "");
}

// a first window: about a fifth of the longer side
function suggestedWindow(image) {
  const size = Math.round(Math.max(image.width, image.height) / 5 / 16) * 16;
  return Math.max(state.options.min_tile, Math.min(size, image.width, image.height));
}

$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", () => {
  pickImage($("file").files[0]);
  $("file").value = "";       // picking the same file again is a change too
});
for (const target of [$("drop"), $("view")]) {
  target.addEventListener("dragover", (event) => {
    event.preventDefault();
    $("drop").classList.add("dragging");
  });
  target.addEventListener("dragleave", () => $("drop").classList.remove("dragging"));
  target.addEventListener("drop", (event) => {
    event.preventDefault();
    $("drop").classList.remove("dragging");
    const file = event.dataTransfer.files[0];
    if (state.mode === "3d") loadModel3d(file);
    else pickImage(file);
  });
}

async function addReferences(files) {
  const images = [...files].filter((f) => f.type.startsWith("image/"));
  if (!images.length) return;
  setStatus(`uploading ${images.length} reference image${images.length > 1 ? "s" : ""}…`);
  try {
    for (const file of images) state.references.push(await uploadFile(file));
    showReferences();
    scheduleRun();
  } catch (e) {
    setStatus(e.message, true);
  }
}

function showReferences() {
  const list = $("references");
  list.innerHTML = "";
  state.references.forEach((r, i) => {
    const figure = document.createElement("figure");
    figure.className = "reference";
    figure.title = `${r.name}, ${r.width} × ${r.height} px`;
    const img = document.createElement("img");
    img.src = r.el.src;
    img.alt = r.name;
    const remove = document.createElement("button");
    remove.className = "remove";
    remove.textContent = "×";
    remove.title = "remove this reference image";
    remove.addEventListener("click", () => {
      state.references.splice(i, 1);
      showReferences();
      scheduleRun();
    });
    figure.append(img, remove);
    list.appendChild(figure);
  });
  updateControls();
}

$("add-reference").addEventListener("click", () => $("reference-file").click());
$("reference-file").addEventListener("change", () => {
  addReferences($("reference-file").files);
  $("reference-file").value = "";
});
$("add-reference").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("add-reference").classList.add("dragging");
});
$("add-reference").addEventListener("dragleave", () => $("add-reference").classList.remove("dragging"));
$("add-reference").addEventListener("drop", (event) => {
  event.preventDefault();
  $("add-reference").classList.remove("dragging");
  addReferences(event.dataTransfer.files);
});

// the points are drawn on the image they were picked on
const pointsHere = () => state.image && state.pointsToken === state.image.token;

function pointsSource() {
  if (!state.scene) return "";
  const f = state.scene.frames.findIndex((fr) => fr.token === state.pointsToken);
  return f >= 0 ? ` on frame ${f + 1}` : "";
}

function showPoints() {
  const list = $("points");
  list.innerHTML = "";
  state.points.forEach(([x, y], i) => {
    const row = document.createElement("div");
    row.className = "point";
    const text = document.createElement("span");
    text.innerHTML = `<span class="marker">${i + 1}</span>`;
    text.append(`point ${i + 1} at ${Math.round(x)}, ${Math.round(y)}${pointsSource()}`);
    const remove = document.createElement("button");
    remove.className = "remove";
    remove.textContent = "×";
    remove.title = "remove this point";
    remove.addEventListener("click", () => {
      state.points.splice(i, 1);
      showPoints();
      requestDraw();
      scheduleRun();
    });
    row.append(text, remove);
    list.appendChild(row);
  });
}

$("pick-points").addEventListener("change", () => $("view").classList.toggle("picking", $("pick-points").checked));

// ---------------------------------------------------------------------------
// Model, method, parameters
// ---------------------------------------------------------------------------

const currentModel = () => state.options.models.find((m) => m.id === $("model").value);
const currentMethod = () => state.options.methods[$("method").value];

// the kinds of prompt the current model and method use
function promptKinds() {
  const model = currentModel(), method = currentMethod();
  if (!model || !method) return [];
  const family = state.options.family_prompts[model.family] || [];
  return method.prompts.filter((p) => family.includes(p));
}

// softmax temperature, from a slider 0..1 on a log scale: 0.003 … 0.3
const temperature = () => 0.003 * 100 ** Number($("temperature").value);

function settings() {
  const kinds = promptKinds();
  return {
    model: $("model").value,
    method: $("method").value,
    token: state.image && state.image.token,
    texts: kinds.includes("text") ? $("texts").value.split("\n") : [],
    template: currentModel().family === "clip" ? $("template").value : "{}",
    references: kinds.includes("image") ? state.references.map((r) => r.token) : [],
    points: kinds.includes("point") ? state.points : [],
    points_token: kinds.includes("point") && state.points.length ? state.pointsToken : null,
    resolution: Number($("resolution").value),
    reference: $("reference").value,
    compete: $("compete").checked,
    temperature: Number(temperature().toPrecision(3)),
    window: Number($("window").value),
    overlap: Number($("overlap").value),
    occluder: Number($("occluder").value),
    steps: Number($("steps").value),
    baseline: $("baseline").value,
    smooth: Number($("smooth").value),
    k: Number($("k").value),
  };
}

const FAMILIES = [
  ["classifier", "classifiers"], ["autoencoder", "autoencoders"], ["dino", "DINOv2"], ["clip", "CLIP"],
  ["segformer", "segmentation models"], ["clipseg", "segmentation models"],
];

function fillModels(value) {
  const select = $("model");
  select.innerHTML = "";
  const groups = new Map();
  for (const [family, title] of FAMILIES) {
    for (const m of state.options.models.filter((m) => m.family === family)) {
      if (!groups.has(title)) {
        const optgroup = document.createElement("optgroup");
        optgroup.label = title;
        groups.set(title, optgroup);
        select.appendChild(optgroup);
      }
      groups.get(title).appendChild(new Option(m.label, m.id));
    }
  }
  if ([...select.options].some((o) => o.value === value)) select.value = value;
}

// the methods of the current model; the same one as before if it has it
function fillMethods() {
  const model = currentModel();
  const before = $("method").value;
  $("method").innerHTML = "";
  for (const id of model.methods) $("method").add(new Option(state.options.methods[id].label, id));
  $("method").value = model.methods.includes(before) ? before : state.options.default_method[model.family];
}

function windowPositions() {
  const image = state.image;
  if (!image) return null;
  const size = Math.max(state.options.min_tile, Math.min(Number($("window").value), image.width, image.height));
  const stride = Math.max(1, Math.floor(size * (1 - Number($("overlap").value) / 100) + 0.5));
  const along = (length) => Math.ceil((length - size) / stride) + 1;
  return along(image.width) * along(image.height);
}

function updateControls() {
  const model = currentModel(), method = currentMethod();
  if (!model || !method) return;
  const kinds = promptKinds();
  $("prompts-panel").hidden = !kinds.length;
  $("text-prompts").hidden = !kinds.includes("text");
  $("template-row").hidden = model.family !== "clip";
  $("image-prompts").hidden = !kinds.includes("image");
  $("point-prompts").hidden = !kinds.includes("point");
  if (!kinds.includes("point") && $("pick-points").checked) {
    $("pick-points").checked = false;
    $("view").classList.remove("picking");
  }
  $("prompts-hint").textContent = model.family === "clipseg"
    ? "CLIPSeg needs at least one prompt."
    : "No prompt: the whole image is the prompt.";

  const params = new Set(method.params);
  if ((params.has("compete") && $("compete").checked) || params.has("k")) params.add("temperature");
  // dense similarity: the reference image setting only matters with reference images
  if (!kinds.includes("image") || !state.references.length) params.delete("reference");
  for (const row of document.querySelectorAll("[data-param]")) row.hidden = !params.has(row.dataset.param);
  $("params-panel").hidden = !params.size;
  $("elevation-row").hidden = $("layout").value !== "orbit";
  for (const id of ["resolution", "overlap", "occluder", "steps", "smooth", "k", "opacity", "gamma", "threshold",
    "views", "elevation", "fov", "plane"]) {
    $(`${id}-out`).textContent = $(id).value;
  }
  $("temperature-out").textContent = temperature().toPrecision(2);

  let hint = "";
  if (params.has("window") && state.image) {
    const n = windowPositions();
    hint = `${n} window positions` + (n > state.options.max_tiles ? ` — too many, at most ${state.options.max_tiles}` : "");
  } else if (method.params.includes("gradients") && model.family === "classifier") {
    hint = `At most ${state.options.max_targets} classes (the most probable ones), each a full pass per step.`;
  }
  $("params-hint").textContent = hint;
  $("method-hint").textContent = METHOD_HELP[$("method").value]
    + (["window", "occlusion", "gradients"].includes($("method").value) ? ` ${SCORE_HELP[model.family]}` : "");
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

let runTimer = null;
function scheduleRun() {
  clearTimeout(runTimer);
  runTimer = setTimeout(startRun, 250);
}

// the server forgot the images (restarted, or too many uploads since): send them again
async function uploadAgain() {
  const renamed = new Map();        // old token -> new
  if (state.mode === "3d" && state.scene) {
    const before = state.scene.frames.map((f) => f.token);
    await uploadFrames(state.scene.frames);
    before.forEach((token, i) => renamed.set(token, state.scene.frames[i].token));
    selectFrame(state.frame, false);
  } else {
    const before = state.image.token;
    state.image = await uploadFile(state.file);
    state.uploaded = { image: state.image, file: state.file };
    renamed.set(before, state.image.token);
  }
  for (let i = 0; i < state.references.length; i++) state.references[i] = await uploadFile(state.references[i].file);
  state.pointsToken = renamed.get(state.pointsToken) || state.pointsToken;
}

// Poll a job until it is done -> its result (null: cancelled or replaced by a newer run)
async function follow(job, run, prefix = "") {
  while (run === state.run) {
    const status = await api(`/api/job/${job}`);
    if (run !== state.run) return null;     // a newer run replaced this one
    if (status.error) throw new Error(status.error);
    if (status.finished) return status.result;
    setStatus(prefix + status.stage);
    setProgress(status.progress);
    await sleep(200);
  }
  return null;
}

// what != null: only the sweep (Compute all frames)
async function startRun(retried = false, what = null) {
  clearTimeout(runTimer);
  if (!state.image || !$("method").value) return;
  const run = ++state.run;
  setStatus("starting…");
  setProgress(0);
  try {
    if (what !== "sweep") {
      const { job } = await postJSON("/api/run", settings());
      const result = await follow(job, run);
      if (!result) return setProgress(null);
      await showResult(result, run);
    }
    // what: null = a change (run, then the sweep if it follows changes), "frame" = another frame only
    if (state.mode === "3d" && state.scene && (what === "sweep" || (what === null && $("auto-sweep").checked))) {
      const { job } = await postJSON("/api/sweep", { ...settings(), frames: state.scene.frames.map((f) => f.token) });
      const result = await follow(job, run, "all frames: ");
      if (result && run === state.run) showSweep(result);
      setProgress(null);
    }
  } catch (e) {
    if (run !== state.run) return;
    if (e.status === 410 && !retried && (state.file || state.scene)) {
      try {
        setStatus("uploading the images again…");
        await uploadAgain();
        return startRun(true, what);
      } catch (again) {
        e = again;
      }
    }
    setStatus(e.message, true);
    setProgress(null);
  }
}

$("compute").addEventListener("click", () => startRun());
$("sweep").addEventListener("click", () => startRun(false, "sweep"));

// the pixels of an 8 bit gray PNG -> {w, h, data: one byte per pixel}
async function grayPixels(url) {
  const el = await loadImage(url);
  const canvas = document.createElement("canvas");
  canvas.width = el.naturalWidth;
  canvas.height = el.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(el, 0, 0);
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const data = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < data.length; i++) data[i] = rgba[4 * i];
  return { w: canvas.width, h: canvas.height, data };
}

// a result from the server, its PNGs read -> {maps, segmentation, pictures}
async function decodeResult(result) {
  const maps = await Promise.all(result.maps.map(async (m) => ({ ...m, ...(await grayPixels(m.png)) })));
  let segmentation = null;
  if (result.segmentation) {
    const s = result.segmentation;
    const [labels, top] = await Promise.all([grayPixels(s.labels), grayPixels(s.top)]);
    segmentation = { w: labels.w, h: labels.h, labels: labels.data, top: top.data, lo: s.lo, hi: s.hi };
  }
  const pictures = await Promise.all(result.images.map(async (p) => ({ name: p.name, el: await loadImage(p.png) })));
  return { maps, segmentation, pictures };
}

async function showResult(result, run) {
  const { maps, segmentation, pictures } = await decodeResult(result);
  if (run !== state.run) return;

  // keep showing the same thing, if the new result has it
  const before = selectionName();
  state.result = result;
  state.maps = maps;
  state.segmentation = segmentation;
  state.pictures = pictures;
  state.selected = defaultSelection(result);
  const same = allSelections().find((key) => selectionName(key) === before);
  if (same && before !== "image") state.selected = same;
  state.layers.clear();

  setProgress(null);
  const model = state.options.models.find((m) => m.id === result.model);
  setStatus(`${model.label}: ${maps.length} map${maps.length === 1 ? "" : "s"}`
    + (segmentation ? " and their segmentation" : "") + ` (${result.seconds}s)`);
  $("result-notes").textContent = result.notes.join(" · ");
  buildCards();
  requestDraw();
  if (state.sweep) drawSweep();       // its colors follow the maps shown
}

function defaultSelection(result) {
  if (result.method === "pca" && state.pictures.length) return "picture:0";
  if (state.segmentation) return "segmentation";
  return state.maps.length ? "map:0" : "image";
}

function clearResult() {
  state.result = null;
  state.maps = [];
  state.segmentation = null;
  state.pictures = [];
  state.selected = "image";
  state.layers.clear();
  $("result-notes").textContent = "";
  buildCards();
  requestDraw();
}

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

const clamp01 = (x) => Math.min(1, Math.max(0, x));

function allSelections() {
  return ["image", ...(state.segmentation ? ["segmentation"] : []),
    ...state.pictures.map((_, i) => `picture:${i}`), ...state.maps.map((_, i) => `map:${i}`)];
}

function selectionName(key = state.selected) {
  if (key === "image") return "image";
  if (key === "segmentation") return "segmentation";
  const [kind, i] = key.split(":");
  const item = kind === "map" ? state.maps[i] : state.pictures[i];
  return item ? item.name : "image";
}

const valueAt = (map, i) => map.lo + (map.data[i] / 255) * (map.hi - map.lo);

// How a map's values become colors: -> {diverging, f: value -> 0..1 (diverging: -1..1)}
function scaleOf(map) {
  let range = $("range").value;
  if (map.values === "signed") {
    const maps = range === "all" ? state.maps.filter((m) => m.values === "signed") : [map];
    const R = Math.max(...maps.map((m) => Math.max(Math.abs(m.lo), Math.abs(m.hi)))) || 1;
    switch ($("signed").value) {
      case "positive": return { f: (v) => Math.max(v, 0) / R };
      case "negative": return { f: (v) => Math.max(-v, 0) / R };
      case "abs": return { f: (v) => Math.abs(v) / R };
      default: return { diverging: true, f: (v) => v / R };
    }
  }
  if (range === "auto") range = map.values === "probability" ? "absolute" : "each";
  let lo = map.lo, hi = map.hi;
  if (range === "absolute") [lo, hi] = [0, 1];
  if (range === "all") {
    const maps = state.maps.filter((m) => m.values !== "signed");
    lo = Math.min(...maps.map((m) => m.lo));
    hi = Math.max(...maps.map((m) => m.hi));
  }
  return { f: (v) => (v - lo) / (hi - lo || 1) };
}

// A map's value at every pixel -> how strong it is (0..1) and which color,
// after contrast and threshold. A threshold above 0 makes the mask hard:
// strength 1 above it, 0 below.
function shaper(map) {
  const { diverging, f } = scaleOf(map);
  const gamma = Number($("gamma").value), threshold = Number($("threshold").value);
  const colors = diverging ? DIVERGING : COLORMAPS[$("colormap").value];
  return (v) => {
    const t = f(v);
    let strength = clamp01(Math.abs(t)) ** gamma;
    if (threshold > 0) strength = strength >= threshold ? 1 : 0;
    const position = diverging ? (Math.sign(t) * strength + 1) / 2 : strength;
    return [strength, colors[Math.round(position * 255)], position];
  };
}

// the colors a map's values are drawn with
const colorsOf = (map) => (scaleOf(map).diverging ? DIVERGING : COLORMAPS[$("colormap").value]);

// the layer drawn over the image, at the map's own resolution
function mapLayer(map, style = $("style").value) {
  const background = hexRgb($("map-bg").value);
  const shape = shaper(map);
  const canvas = document.createElement("canvas");
  canvas.width = map.w;
  canvas.height = map.h;
  const ctx = canvas.getContext("2d");
  const pixels = ctx.createImageData(map.w, map.h);
  const d = pixels.data;
  const hard = Number($("threshold").value) > 0;
  for (let i = 0; i < map.data.length; i++) {
    const [strength, color] = shape(valueAt(map, i));
    let rgb = color, alpha = 255;
    if (style === "cutout") {
      rgb = background;
      alpha = 255 * (1 - strength);
    } else if (hard && strength === 0) {
      alpha = style === "heat" ? 0 : 255;
      if (style === "map") rgb = background;
    }
    d[4 * i] = rgb[0];
    d[4 * i + 1] = rgb[1];
    d[4 * i + 2] = rgb[2];
    d[4 * i + 3] = alpha;
  }
  ctx.putImageData(pixels, 0, 0);
  return canvas;
}

// How sure the segmentation is at every pixel: the winning map's value, 0..1
function confidenceOf(s, maps = state.maps) {
  const probability = maps.some((m) => m.segment && m.values === "probability");
  return (i) => {
    const v = s.lo + (s.top[i] / 255) * (s.hi - s.lo);
    return probability ? v : (v - s.lo) / (s.hi - s.lo || 1);
  };
}

// The segmentation: every pixel in the color of the map that wins there.
// Below the threshold (how sure) nothing wins; contrast fades the colors
// where the winner is not sure (a fuzzy segmentation).
// maps: the maps the segmentation is of (another frame's: colored by key,
// as in the graph)
function segmentationLayer(s, style = $("style").value, maps = state.maps) {
  const background = hexRgb($("map-bg").value);
  const gamma = Number($("gamma").value), threshold = Number($("threshold").value);
  const confidence = confidenceOf(s, maps);
  const colors = maps === state.maps ? maps.map((_, i) => labelRgb(i)) : maps.map((m) => labelRgb(colorIndexOfKey(m.key)));
  const canvas = document.createElement("canvas");
  canvas.width = s.w;
  canvas.height = s.h;
  const ctx = canvas.getContext("2d");
  const pixels = ctx.createImageData(s.w, s.h);
  const d = pixels.data;
  for (let i = 0; i < s.labels.length; i++) {
    const c = confidence(i);
    const on = c >= threshold;
    const strength = gamma > 1 ? clamp01(c) ** (gamma - 1) : 1;
    let rgb = colors[s.labels[i]] || [0, 0, 0], alpha = on ? 255 * strength : 0;
    if (style === "cutout") {
      rgb = background;
      alpha = on ? 0 : 255;
    } else if (style === "map" && !on) {
      rgb = background;
      alpha = 255;
    }
    d[4 * i] = rgb[0];
    d[4 * i + 1] = rgb[1];
    d[4 * i + 2] = rgb[2];
    d[4 * i + 3] = alpha;
  }
  ctx.putImageData(pixels, 0, 0);
  return canvas;
}

// what is drawn over the image for a selection (null: the image alone)
function layerOf(key) {
  if (key === "image") return null;
  if (!state.layers.has(key)) {
    const [kind, i] = key.split(":");
    let layer = null;
    if (key === "segmentation") layer = state.segmentation && segmentationLayer(state.segmentation);
    else if (kind === "picture") layer = state.pictures[i] && state.pictures[i].el;
    else layer = state.maps[i] && mapLayer(state.maps[i]);
    state.layers.set(key, layer);
  }
  return state.layers.get(key);
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

// The image in a square view of `side` pixels: {k: scale, x, y offset}
function fitOf(w, h, side, view = null) {
  const base = side * (1 - 2 * MARGIN) / Math.max(w, h);
  const { scale, tx, ty } = view || { scale: 1, tx: 0, ty: 0 };
  return { k: base * scale, x: (side - w * base) / 2 * scale + tx, y: (side - h * base) / 2 * scale + ty };
}

function prepareCanvas(canvas, width, height) {
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.imageSmoothingQuality = "high";
  return ctx;
}

// the image and a selection's layer, in a box x, y, w, h
function drawComposite(ctx, key, x, y, w, h) {
  const layer = layerOf(key);
  const style = $("style").value;
  if (!layer || style !== "map") ctx.drawImage(state.image.el, x, y, w, h);
  if (!layer) return;
  ctx.imageSmoothingEnabled = $("smooth-display").checked || key.startsWith("picture");
  ctx.globalAlpha = style === "map" ? 1 : Number($("opacity").value);
  ctx.drawImage(layer, x, y, w, h);
  ctx.globalAlpha = 1;
  ctx.imageSmoothingEnabled = true;
}

function drawPoints(ctx, fit) {
  if ($("point-prompts").hidden || !pointsHere()) return;
  ctx.font = "bold 11px Courier New, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  state.points.forEach(([x, y], i) => {
    const cx = fit.x + x * fit.k, cy = fit.y + y * fit.k;
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, 2 * Math.PI);
    ctx.fillStyle = "#000000";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.fillText(String(i + 1), cx, cy + 1);
  });
}

function drawView() {
  const canvas = $("canvas");
  const side = canvas.clientWidth;
  const ctx = prepareCanvas(canvas, side, side);
  $("empty").hidden = Boolean(state.image);
  $("empty").textContent = state.mode === "3d" ? "load a Rhino file" : "upload an image";
  $("view-title").textContent = state.image && state.selected !== "image" ? selectionName() : "View";
  if (!state.image) return;
  const { width: W, height: H } = state.image;
  const fit = fitOf(W, H, side, state.view);
  drawComposite(ctx, state.selected, fit.x, fit.y, W * fit.k, H * fit.k);
  drawPoints(ctx, fit);
}

let frame = 0;
function requestDraw() {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    drawView();
    drawCards();
    showHere();
  });
}

// display settings changed: every layer is colored again
function restyle() {
  state.layers.clear();
  updateControls();
  requestDraw();
}

new ResizeObserver(requestDraw).observe($("view"));
new ResizeObserver(requestDraw).observe($("maps"));

// ---------------------------------------------------------------------------
// Maps: one card per selection
// ---------------------------------------------------------------------------

function cardRange(key) {
  if (!key.startsWith("map")) return "";
  const m = state.maps[key.split(":")[1]];
  const f = (v) => (Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(3));
  return `${f(m.lo)} … ${f(m.hi)}`;
}

function buildCards() {
  const list = $("maps");
  list.innerHTML = "";
  const keys = state.image && state.result ? allSelections() : [];
  $("maps-empty").hidden = keys.length > 0;
  keys.forEach((key) => {
    const card = document.createElement("button");
    card.className = "map-card";
    card.dataset.key = key;
    const canvas = document.createElement("canvas");
    const name = document.createElement("span");
    name.className = "name";
    const [kind, i] = key.split(":");
    if (kind === "map" && state.segmentation && state.maps[i].segment) {
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = labelCss(Number(i));
      name.appendChild(swatch);
    }
    name.append(selectionName(key));
    const range = document.createElement("span");
    range.className = "range";
    range.textContent = cardRange(key);
    card.append(canvas, name, range);
    card.addEventListener("click", () => {
      state.selected = key;
      requestDraw();
    });
    list.appendChild(card);
  });
}

function drawCards() {
  if (!state.image) return;
  const { width: W, height: H } = state.image;
  for (const card of $("maps").children) {
    card.classList.toggle("active", card.dataset.key === state.selected);
    // as high as the strip leaves it, as wide as the image's proportions ask
    const canvas = card.querySelector("canvas");
    const h = Math.max(canvas.clientHeight, 20);
    const w = Math.max(20, Math.round(h * W / H));
    card.style.width = `${Math.max(w + 8, 96)}px`;     // room for the name under a narrow picture
    canvas.style.width = `${w}px`;
    const ctx = prepareCanvas(canvas, w, h);
    drawComposite(ctx, card.dataset.key, 0, 0, w, h);
  }
}

// ---------------------------------------------------------------------------
// Hover: the value of every map here
// ---------------------------------------------------------------------------

// the pixel of a w x h map (or segmentation) under image point x, y
function indexAt(w, h, [x, y]) {
  const { width: W, height: H } = state.image;
  const c = Math.min(w - 1, Math.floor(x / W * w)), r = Math.min(h - 1, Math.floor(y / H * h));
  return r * w + c;
}

const formatValue = (v) => (Math.abs(v) >= 100 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(2) : v.toFixed(3));

// the point under the mouse and the selected map's value there
function showHoverInfo() {
  const p = state.hover;
  let text = " ";
  if (p && state.image) {
    text = `${Math.round(p[0])}, ${Math.round(p[1])}`;
    const [kind, i] = state.selected.split(":");
    const map = kind === "map" && state.maps[i];
    if (map) text += ` · ${map.name}: ${formatValue(valueAt(map, indexAt(map.w, map.h, p)))}`;
  }
  $("hover-info").textContent = text;
}

function showHere() {
  showHoverInfo();
  const bars = $("here-bars");
  const here = state.hover;
  const show = here && state.maps.length;
  $("here-empty").hidden = Boolean(show);
  bars.innerHTML = "";
  if (!show) return;
  const s = state.segmentation;
  if (s) {
    const i = indexAt(s.w, s.h, here);
    const winner = document.createElement("div");
    winner.className = "winner";
    const confidence = confidenceOf(s)(i);
    const swatch = `<span class="swatch" style="display:inline-block;background:${labelCss(s.labels[i])}"></span>`;
    winner.innerHTML = `${swatch} `;
    winner.append(`${state.maps[s.labels[i]].name}`
      + (confidence < Number($("threshold").value) ? " (below the threshold)" : ""));
    bars.appendChild(winner);
  }
  const rows = state.maps.map((m, k) => {
    const v = valueAt(m, indexAt(m.w, m.h, here));
    const { diverging, f } = scaleOf(m);
    return { k, m, v, t: f(v), diverging };
  }).sort((a, b) => b.t - a.t).slice(0, HERE_ROWS);
  for (const { k, m, v, t, diverging } of rows) {
    const row = document.createElement("div");
    row.className = "bar-row" + (s && state.maps[s.labels[indexAt(s.w, s.h, here)]] === m ? " best" : "");
    const label = document.createElement("div");
    label.className = "label";
    const name = document.createElement("span");
    if (s && m.segment) {
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = labelCss(k);
      name.appendChild(swatch);
    }
    name.append(m.name);
    const value = document.createElement("span");
    value.textContent = formatValue(v);
    label.append(name, value);
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill" + (diverging ? (t < 0 ? " negative" : " positive") : "");
    fill.style.width = `${Math.round(clamp01(Math.abs(t)) * 100)}%`;
    track.appendChild(fill);
    row.append(label, track);
    bars.appendChild(row);
  }
}

function mouseInImage(event) {
  const r = $("view").getBoundingClientRect();
  const fit = fitOf(state.image.width, state.image.height, r.width, state.view);
  return [(event.clientX - r.left - fit.x) / fit.k, (event.clientY - r.top - fit.y) / fit.k];
}

const inImage = ([x, y]) => x >= 0 && y >= 0 && x < state.image.width && y < state.image.height;

$("view").addEventListener("mousemove", (event) => {
  if (!state.image || $("view").classList.contains("panning")) return;
  const p = mouseInImage(event);
  state.hover = inImage(p) ? p : null;
  showHere();
});
$("view").addEventListener("mouseleave", () => {
  state.hover = null;
  showHere();
});

// Zoom with the wheel (around the mouse), pan by dragging, double-click resets.
// A click that doesn't drag adds a point, when picking points.
const view = $("view");
view.addEventListener("wheel", (event) => {
  event.preventDefault();
  const r = view.getBoundingClientRect();
  const mx = event.clientX - r.left, my = event.clientY - r.top;
  const v = state.view;
  const scale = Math.min(40, Math.max(0.5, v.scale * Math.exp(-event.deltaY * 0.0015)));
  const f = scale / v.scale;
  state.view = { scale, tx: mx - (mx - v.tx) * f, ty: my - (my - v.ty) * f };
  requestDraw();
}, { passive: false });
let drag = null;
view.addEventListener("pointerdown", (event) => {
  drag = { x: event.clientX, y: event.clientY, tx: state.view.tx, ty: state.view.ty };
  view.setPointerCapture(event.pointerId);
});
view.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
  if (!view.classList.contains("panning") && Math.hypot(dx, dy) < 3) return;
  view.classList.add("panning");
  state.view = { ...state.view, tx: drag.tx + dx, ty: drag.ty + dy };
  requestDraw();
});
view.addEventListener("pointerup", (event) => {
  const panned = view.classList.contains("panning");
  drag = null;
  view.classList.remove("panning");
  if (panned || !state.image || !$("pick-points").checked || $("point-prompts").hidden) return;
  const p = mouseInImage(event);
  if (!inImage(p)) return;
  if (!pointsHere()) state.points = [];      // picked on another image: start over on this one
  state.pointsToken = state.image.token;
  state.points.push(p);
  showPoints();
  requestDraw();
  scheduleRun();
});
view.addEventListener("pointercancel", () => { drag = null; view.classList.remove("panning"); });
view.addEventListener("dblclick", () => {
  state.view = { scale: 1, tx: 0, ty: 0 };
  requestDraw();
});

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

function download(canvas, suffix) {
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stem = state.image.name.replace(/\.[^.]*$/, "");
    link.href = url;
    link.download = `${stem}_${$("method").value}_${suffix}.png`.replace(/[^\w.-]+/g, "_");
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }, "image/png");
}

function imageSizedCanvas() {
  const { width: W, height: H } = state.image;
  const f = Math.min(1, SAVE_MAX_SIDE / Math.max(W, H));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(W * f);
  canvas.height = Math.round(H * f);
  return canvas;
}

$("save-png").addEventListener("click", () => {
  if (!state.image) return;
  const canvas = imageSizedCanvas();
  drawComposite(canvas.getContext("2d"), state.selected, 0, 0, canvas.width, canvas.height);
  download(canvas, selectionName());
});

// the selected map as a gray mask: white = strong (after range, contrast, threshold)
$("save-mask").addEventListener("click", () => {
  const [kind, i] = state.selected.split(":");
  if (!state.image || kind !== "map") {
    setStatus("select a map to save its mask", true);
    return;
  }
  const map = state.maps[i];
  const shape = shaper(map);
  const small = document.createElement("canvas");
  small.width = map.w;
  small.height = map.h;
  const sctx = small.getContext("2d");
  const pixels = sctx.createImageData(map.w, map.h);
  for (let p = 0; p < map.data.length; p++) {
    const g = Math.round(255 * shape(valueAt(map, p))[0]);
    pixels.data.set([g, g, g, 255], 4 * p);
  }
  sctx.putImageData(pixels, 0, 0);
  const canvas = imageSizedCanvas();
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = $("smooth-display").checked;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
  download(canvas, `${map.name}_mask`);
});

// ---------------------------------------------------------------------------
// 3D model mode
// ---------------------------------------------------------------------------

const stemOf = (name) => name.replace(/\.[^.]*$/, "");

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename.replace(/[^\w.-]+/g, "_");
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function setMode(mode) {
  state.mode = mode;
  const three = mode === "3d";
  $("image-source").hidden = three;
  for (const id of ["scene-panel", "frames-panel", "sweep-panel"]) $(id).hidden = !three;
  document.querySelector(".stage.segment").classList.toggle("three", three);     // its rows (segment.css)
  if (three) {
    state.image = null;
    if (state.scene && state.scene.frames.length) selectFrame(state.frame || 0, false);
  } else {
    state.image = state.uploaded && state.uploaded.image;
    state.file = state.uploaded && state.uploaded.file;
  }
  clearResult();
  updateControls();
  startRun();
}

for (const radio of document.querySelectorAll("input[name=mode]")) {
  radio.addEventListener("change", () => setMode(radio.value));
}

async function loadModel3d(file) {
  if (!file) return;
  if (!window.Scene3D) {
    setStatus("three.js didn't load: the 3D mode needs the internet (it comes from cdn.jsdelivr.net)", true);
    return;
  }
  setStatus(`reading ${file.name}…`);
  try {
    const form = new FormData();
    form.append("file", file);
    const data = await api("/api/scene/model", { method: "POST", body: form });
    window.Scene3D.build(data);
    for (const f of state.scene ? state.scene.frames : []) URL.revokeObjectURL(f.url);
    state.scene = { id: data.id, name: data.name, notes: data.notes, frames: [] };
    state.frame = 0;
    $("drop-3dm").textContent = data.name;
    $("drop-3dm").classList.add("has-image");
    $("scene-info").textContent = data.notes.join(", ");
    await renderScene();
  } catch (e) {
    setStatus(e.message, true);
    setProgress(null);
  }
}

// the frames as one set, their tokens written into them
async function uploadFrames(frames) {
  const form = new FormData();
  frames.forEach((f, i) => form.append("frames", f.blob, `frame_${String(i + 1).padStart(3, "0")}.png`));
  const { tokens } = await api("/api/scene/frames", { method: "POST", body: form });
  frames.forEach((f, i) => { f.token = tokens[i]; });
}

async function renderScene() {
  if (!state.scene) return;
  const run = ++state.run;        // stops whatever was computing
  setStatus("rendering…");
  setProgress(0);
  try {
    const frames = await window.Scene3D.render({
      views: Number($("views").value), layout: $("layout").value, elevation: Number($("elevation").value),
      fov: Number($("fov").value), size: Number($("frame-size").value), mode: $("material").value,
      background: $("render-bg").value, onProgress: (p) => setProgress(p * 0.7),
    });
    for (const f of frames) {
      f.url = URL.createObjectURL(f.blob);
      f.el = await loadImage(f.url);
    }
    setStatus(`uploading ${frames.length} frames…`);
    await uploadFrames(frames);
    if (run !== state.run) return;
    for (const f of state.scene.frames) URL.revokeObjectURL(f.url);
    state.scene.frames = frames;
    state.sweep = null;
    state.points = [];
    showPoints();
    buildFrames();
    selectFrame(Math.min(state.frame || 0, frames.length - 1), false);
    clearResult();
    drawSweep();
    startRun();
  } catch (e) {
    setStatus(e.message, true);
    setProgress(null);
  }
}

// frame i becomes the image; run: compute it (not the sweep, which has every frame already)
function selectFrame(i, run = true) {
  const f = state.scene.frames[i];
  if (!f) return;
  state.frame = i;
  const { width, height } = f.camera;
  state.image = { token: f.token, name: `${stemOf(state.scene.name)} frame ${i + 1}`, width, height,
    original: [width, height], el: f.el };
  state.file = new File([f.blob], `frame_${i + 1}.png`, { type: "image/png" });
  state.layers.clear();
  drawFrames();
  drawSweep();
  requestDraw();
  const strip = $("frames").children[i];
  if (strip) strip.scrollIntoView({ block: "nearest", inline: "nearest" });
  if (run) startRun(false, "frame");
}

function buildFrames() {
  const list = $("frames");
  list.innerHTML = "";
  const frames = state.scene ? state.scene.frames : [];
  $("frames-empty").hidden = frames.length > 0;
  frames.forEach((f, i) => {
    const button = document.createElement("button");
    button.className = "frame-thumb";
    button.title = `frame ${i + 1}: camera at ${f.camera.position.map((c) => c.toFixed(2)).join(", ")}`;
    const img = document.createElement("img");
    img.src = f.url;
    img.alt = `frame ${i + 1}`;
    const bar = document.createElement("span");
    bar.className = "bar";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = i + 1;
    button.append(img, bar, label);
    button.addEventListener("click", () => selectFrame(i));
    list.appendChild(button);
  });
}

// the selected frame outlined; with a highlighted map, a bar per frame: its value
function drawFrames() {
  const values = state.highlight !== null && state.sweep ? seriesValues(state.highlight) : null;
  const known = values ? values.filter((v) => v !== null) : [];
  const lo = Math.min(...known), hi = Math.max(...known);
  [...$("frames").children].forEach((button, i) => {
    button.classList.toggle("active", i === state.frame);
    const bar = button.querySelector(".bar");
    bar.hidden = !values || values[i] === null;
    if (!bar.hidden) {
      bar.style.height = `${Math.round(100 * (values[i] - lo) / (hi - lo || 1))}%`;
      bar.style.background = colorOfKey(state.highlight);
    }
  });
}

document.addEventListener("keydown", (event) => {
  if (state.mode !== "3d" || !state.scene || !state.scene.frames.length) return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  const n = state.scene.frames.length;
  if (event.key === "ArrowRight") selectFrame(((state.frame || 0) + 1) % n);
  if (event.key === "ArrowLeft") selectFrame(((state.frame || 0) - 1 + n) % n);
});

$("drop-3dm").addEventListener("click", () => $("file-3dm").click());
$("file-3dm").addEventListener("change", () => {
  loadModel3d($("file-3dm").files[0]);
  $("file-3dm").value = "";
});
$("drop-3dm").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("drop-3dm").classList.add("dragging");
});
$("drop-3dm").addEventListener("dragleave", () => $("drop-3dm").classList.remove("dragging"));
$("drop-3dm").addEventListener("drop", (event) => {
  event.preventDefault();
  $("drop-3dm").classList.remove("dragging");
  loadModel3d(event.dataTransfer.files[0]);
});
$("render").addEventListener("click", renderScene);
for (const id of ["views", "elevation", "fov", "plane"]) $(id).addEventListener("input", updateControls);
for (const id of ["views", "layout", "elevation", "fov", "frame-size", "material", "render-bg"]) {
  $(id).addEventListener("change", () => { updateControls(); renderScene(); });
}

// ---------------------------------------------------------------------------
// Per frame: a line per map, the best frames of each
// ---------------------------------------------------------------------------

// The maps of the sweep are colored as in the segmentation of the frame shown,
// the ones it doesn't have after those
function colorIndexOfKey(key) {
  const shown = state.maps.findIndex((m) => m.key === key);
  if (shown >= 0) return shown;
  const missing = state.sweep ? state.sweep.keys.filter((k) => !state.maps.some((m) => m.key === k)) : [];
  return state.maps.length + Math.max(0, missing.indexOf(key));
}
const colorOfKey = (key) => labelCss(colorIndexOfKey(key));

function aggregate() {
  const what = $("aggregate").value;
  return what === "area" && !state.sweep.area ? "mean" : what;
}

// one value per frame (null where the frame doesn't have the map)
function seriesValues(key) {
  const k = state.sweep.keys.indexOf(key);
  return state.sweep[aggregate()].map((row) => row[k]);
}

function showSweep(result) {
  state.sweep = result;
  if (state.highlight !== null && !result.keys.includes(state.highlight)) state.highlight = null;
  $("aggregate").querySelector("[value=area]").disabled = !result.area;
  const model = state.options.models.find((m) => m.id === result.model);
  $("sweep-info").textContent = `${model.label} · ${state.options.methods[result.method].label} · ${result.seconds}s`;
  setStatus(`${result.frames.length} frames, ${result.keys.length} maps each (${result.seconds}s)`);
  drawSweep();
  drawFrames();
}

const SVG = "http://www.w3.org/2000/svg";
function svgElement(name, attributes, parent) {
  const el = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attributes)) el.setAttribute(k, v);
  parent.appendChild(el);
  return el;
}

const lines = new Map();        // key -> its path in the graph

function drawSweep() {
  const svg = $("graph");
  const s = state.sweep;
  $("sweep-empty").hidden = Boolean(s);
  svg.style.display = s ? "" : "none";
  $("legend").innerHTML = "";
  $("best").innerHTML = "";
  svg.innerHTML = "";
  lines.clear();
  if (!s || !state.scene) return;
  const n = s.frames.length;
  const W = svg.clientWidth || 600, H = Math.max(svg.clientHeight, 60);     // as big as its box
  const m = { l: 56, r: 12, t: 10, b: 24 };
  const all = s[aggregate()].flat().filter((v) => v !== null);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (lo === hi) [lo, hi] = [lo - 1, hi + 1];
  const x = (i) => m.l + (n > 1 ? i / (n - 1) : 0.5) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);

  // axes: value at the bottom, middle, top; frame numbers
  for (const v of [lo, (lo + hi) / 2, hi]) {
    svgElement("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid" }, svg);
    svgElement("text", { x: m.l - 6, y: y(v) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = formatValue(v);
  }
  const step = Math.max(1, Math.ceil(n / 30));
  for (let i = 0; i < n; i += step) {
    svgElement("text", { x: x(i), y: H - 6, class: "tick", "text-anchor": "middle" }, svg).textContent = i + 1;
  }
  if (state.frame !== null) {
    svgElement("line", { x1: x(state.frame), x2: x(state.frame), y1: m.t, y2: H - m.b, class: "selected-frame" }, svg);
  }
  const cursor = svgElement("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, class: "cursor", visibility: "hidden" }, svg);

  for (const key of s.keys) {
    const values = seriesValues(key);
    let d = "", pen = "M";
    values.forEach((v, i) => {
      if (v === null) { pen = "M"; return; }
      d += `${pen}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = "L";
    });
    if (!d) continue;
    const line = svgElement("path", { d, class: "series", stroke: colorOfKey(key) }, svg);
    const hit = svgElement("path", { d, class: "hit" }, svg);
    hit.addEventListener("mouseenter", () => setHighlight(key));
    hit.addEventListener("mouseleave", () => setHighlight(null));
    lines.set(key, line);
  }

  // hover: the values of the frame under the mouse; click: show that frame
  const frameAt = (event) => {
    const r = svg.getBoundingClientRect();
    return Math.max(0, Math.min(n - 1, Math.round((event.clientX - r.left - m.l) / (W - m.l - m.r) * (n - 1))));
  };
  svg.onmousemove = (event) => {
    const i = frameAt(event);
    cursor.setAttribute("x1", x(i));
    cursor.setAttribute("x2", x(i));
    cursor.setAttribute("visibility", "visible");
    const row = s[aggregate()][i];
    const top = s.keys.map((k, j) => [k, row[j]]).filter(([, v]) => v !== null)
      .sort((a, b) => b[1] - a[1]).slice(0, 5);
    $("graph-info").textContent = `frame ${i + 1}: ` + top.map(([k, v]) => `${k} ${formatValue(v)}`).join(" · ");
  };
  svg.onmouseleave = () => {
    cursor.setAttribute("visibility", "hidden");
    $("graph-info").textContent = " ";
  };
  svg.onclick = (event) => selectFrame(frameAt(event));

  // legend: hover to highlight a line, click to show its best frame
  for (const key of s.keys) {
    const item = document.createElement("button");
    item.className = "legend-item";
    item.dataset.key = key;
    item.innerHTML = `<span class="swatch" style="background:${colorOfKey(key)}"></span>`;
    item.append(key);
    item.addEventListener("mouseenter", () => setHighlight(key));
    item.addEventListener("mouseleave", () => setHighlight(null));
    item.addEventListener("click", () => selectFrame(bestFrames(key)[0]));
    $("legend").appendChild(item);
  }
  drawBest();
  setHighlight(state.highlight);
}

// the frames with the highest values of a map, best first
function bestFrames(key, count = 1) {
  return seriesValues(key).map((v, i) => [v, i]).filter(([v]) => v !== null)
    .sort((a, b) => b[0] - a[0]).slice(0, count).map(([, i]) => i);
}

function drawBest() {
  const s = state.sweep;
  for (const key of s.keys) {
    const row = document.createElement("div");
    row.className = "best-row";
    row.dataset.key = key;
    const name = document.createElement("div");
    name.className = "best-name";
    name.innerHTML = `<span class="swatch" style="background:${colorOfKey(key)}"></span>`;
    name.append(key);
    row.appendChild(name);
    const values = seriesValues(key);
    for (const i of bestFrames(key, 3)) {
      const figure = document.createElement("button");
      figure.className = "best-frame" + (i === state.frame ? " active" : "");
      figure.title = `frame ${i + 1}`;
      const img = document.createElement("img");
      img.src = state.scene.frames[i].url;
      img.alt = `frame ${i + 1}`;
      const caption = document.createElement("span");
      caption.textContent = `${i + 1}: ${formatValue(values[i])}`;
      figure.append(img, caption);
      figure.addEventListener("click", () => selectFrame(i));
      row.appendChild(figure);
    }
    row.addEventListener("mouseenter", () => setHighlight(key));
    row.addEventListener("mouseleave", () => setHighlight(null));
    $("best").appendChild(row);
  }
}

// one map's line stands out, the others fade
function setHighlight(key) {
  state.highlight = key;
  for (const [k, line] of lines) {
    line.classList.toggle("highlight", k === key);
    line.classList.toggle("faded", key !== null && k !== key);
  }
  for (const el of document.querySelectorAll("#legend .legend-item, #best .best-row")) {
    el.classList.toggle("active", el.dataset.key === key);
  }
  drawFrames();
}

$("aggregate").addEventListener("change", drawSweep);
new ResizeObserver(() => { if (state.sweep) drawSweep(); }).observe($("sweep-panel"));

// ---------------------------------------------------------------------------
// Rhino files: frustums, the colored model
// ---------------------------------------------------------------------------

const CLAY = [230, 227, 221];       // vertices no view sees: the render's clay color

// what is selected, in a form that finds the same thing in another frame's result
function selectionOf() {
  const [kind, i] = state.selected.split(":");
  if (state.selected === "segmentation") return { kind: "segmentation" };
  if (kind === "map") return { kind: "map", key: state.maps[i].key, name: state.maps[i].name };
  if (kind === "picture") return { kind: "picture", name: state.pictures[i].name };
  return null;
}

const currentFrameData = () => ({ maps: state.maps, segmentation: state.segmentation, pictures: state.pictures });

// The selection drawn alone (style "map") from a frame's result, null if that frame doesn't have it
function layerFor(data, selection) {
  if (selection.kind === "segmentation") {
    return data.segmentation && segmentationLayer(data.segmentation, "map", data.maps);
  }
  if (selection.kind === "picture") {
    const picture = data.pictures.find((p) => p.name === selection.name);
    return picture && picture.el;
  }
  const map = data.maps.find((m) => m.key === selection.key);
  return map && mapLayer(map, "map");
}

// A frame's result from the last sweep (read once, then kept)
const sweepCache = new Map();
async function frameData(i) {
  const key = `${state.sweep.sweep}/${i}`;
  if (!sweepCache.has(key)) {
    if (sweepCache.size > 200) sweepCache.clear();
    sweepCache.set(key, await decodeResult(await api(`/api/sweep/${state.sweep.sweep}/${i}`)));
  }
  return sweepCache.get(key);
}

// All frames need the sweep of what is shown now
function sweepIsCurrent() {
  const s = state.sweep;
  if (s && s.model === $("model").value && s.method === $("method").value) return true;
  setStatus("compute all frames first (with this model and method)", true);
  return false;
}

// a layer at most 160 px along a side, as a PNG data URL: the mask of a frustum
function maskUrl(layer, camera, smooth) {
  const f = Math.min(1, 160 / Math.max(camera.width, camera.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(2, Math.round(camera.width * f));
  canvas.height = Math.max(2, Math.round(camera.height * f));
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = smooth;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(layer, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

async function postFor3dm(path, body, filename) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${path}: ${(await response.text()).replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()}`);
  saveBlob(await response.blob(), filename);
  setStatus(`saved ${filename}`);
}

function exportReady() {
  if (state.mode !== "3d" || state.frame === null || !state.scene || !state.scene.frames.length) return null;
  const selection = selectionOf();
  if (!selection) setStatus("select a map, the segmentation or a picture to save", true);
  return selection;
}

const exportName = (what) => `${stemOf(state.scene.name)} ${what} ${selectionName()}`;

// The selected frame's camera and the selected map, as a Rhino file: the map
// as it is drawn alone (colors, segments, picture) on a mesh across the frustum
$("save-frustum").addEventListener("click", async () => {
  const selection = exportReady();
  if (!selection) return;
  const frame = state.scene.frames[state.frame];
  const name = exportName(`frame ${state.frame + 1}`);
  setStatus("writing the Rhino file…");
  try {
    const smooth = $("smooth-display").checked || selection.kind === "picture";
    await postFor3dm("/api/export/frustum", {
      camera: frame.camera, mask: maskUrl(layerFor(currentFrameData(), selection), frame.camera, smooth),
      token: frame.token, name, plane: Number($("plane").value),
    }, `${name}.3dm`);
  } catch (e) {
    setStatus(e.message, true);
  }
});

// Every frame's frustum and its map (from the last sweep), with the map's
// value in that frame (the graph's) as user text
$("save-frustums").addEventListener("click", async () => {
  const selection = exportReady();
  if (!selection || !sweepIsCurrent()) return;
  const frames = [];
  const smooth = $("smooth-display").checked || selection.kind === "picture";
  const values = selection.kind === "map" ? seriesValues(selection.key) : null;
  try {
    for (let i = 0; i < state.scene.frames.length; i++) {
      setStatus(`frame ${i + 1}/${state.scene.frames.length}…`);
      const frame = state.scene.frames[i];
      const layer = layerFor(await frameData(i), selection);
      if (!layer) continue;
      frames.push({ camera: frame.camera, mask: maskUrl(layer, frame.camera, smooth), token: frame.token,
        name: `frame ${i + 1}`, value: values ? values[i] : null });
    }
    setStatus("writing the Rhino file…");
    const name = exportName("all frames");
    await postFor3dm("/api/export/frustums", { frames, name, plane: Number($("plane").value) }, `${name}.3dm`);
  } catch (e) {
    setStatus(e.message, true);
  }
});

// What a frame's selection is at every pixel: a position in the map's
// colors (maps: averaged before coloring) or a color (segmentation, pictures)
function samplerFor(data, selection) {
  if (selection.kind === "map") {
    const map = data.maps.find((m) => m.key === selection.key);
    if (!map) return null;
    const shape = shaper(map);
    const scalar = new Float32Array(map.data.length);
    for (let i = 0; i < scalar.length; i++) scalar[i] = shape(valueAt(map, i))[2];
    return { w: map.w, h: map.h, scalar, colors: colorsOf(map) };
  }
  const layer = layerFor(data, selection);
  if (!layer) return null;
  const canvas = document.createElement("canvas");
  canvas.width = layer.width || layer.naturalWidth;
  canvas.height = layer.height || layer.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(layer, 0, 0);
  return { w: canvas.width, h: canvas.height, rgb: ctx.getImageData(0, 0, canvas.width, canvas.height).data };
}

// Bytes -> base64, in pieces (String.fromCharCode takes only so many arguments)
function base64Of(bytes) {
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

// The model's meshes with a color per vertex: the selection where the vertex
// lands in a frame, if the camera sees it. One view: the selected frame. All
// views: the average over the frames that see it, each weighted by how
// squarely the surface faces its camera. Vertices no frame sees: clay.
$("save-model").addEventListener("click", async () => {
  const selection = exportReady();
  const all = $("model-views").value === "all";
  if (!selection || (all && !sweepIsCurrent())) return;
  const counts = window.Scene3D.vertexCounts();
  const sums = counts.map((n) => new Float32Array(3 * n));     // scalar in [0], or r, g, b
  const weights = counts.map((n) => new Float32Array(n));
  const frames = all ? state.scene.frames.map((_, i) => i) : [state.frame];
  let colors = null;
  try {
    for (const [step, f] of frames.entries()) {
      setStatus(`projecting onto the model: frame ${f + 1} (${step + 1}/${frames.length})…`);
      setProgress(step / frames.length);
      await sleep(0);       // let the page show it
      const sampler = samplerFor(all ? await frameData(f) : currentFrameData(), selection);
      if (!sampler) continue;
      colors = colors || sampler.colors;
      const { width: W, height: H } = state.scene.frames[f].camera;
      window.Scene3D.project(state.scene.frames[f].camera).forEach(({ uv, facing }, m) => {
        for (let v = 0; v < counts[m]; v++) {
          if (uv[2 * v] < 0) continue;
          const w = all ? facing[v] : 1;
          const px = Math.min(sampler.w - 1, Math.floor(uv[2 * v] / W * sampler.w));
          const py = Math.min(sampler.h - 1, Math.floor(uv[2 * v + 1] / H * sampler.h));
          const p = py * sampler.w + px;
          if (sampler.scalar) {
            sums[m][3 * v] += w * sampler.scalar[p];
          } else {
            for (let c = 0; c < 3; c++) sums[m][3 * v + c] += w * sampler.rgb[4 * p + c];
          }
          weights[m][v] += w;
        }
      });
    }
    let seen = 0, total = 0;
    const encoded = counts.map((n, m) => {
      const rgb = new Uint8Array(3 * n);
      for (let v = 0; v < n; v++) {
        const w = weights[m][v];
        total++;
        let color = CLAY;
        if (w > 0) {
          seen++;
          color = colors ? colors[Math.round(clamp01(sums[m][3 * v] / w) * 255)]
            : [0, 1, 2].map((c) => Math.round(sums[m][3 * v + c] / w));
        }
        rgb.set(color, 3 * v);
      }
      return base64Of(rgb);
    });
    setProgress(null);
    setStatus(`writing the Rhino file (${seen} of ${total} vertices seen)…`);
    const name = exportName(all ? "model all views" : `model frame ${state.frame + 1}`);
    await postFor3dm("/api/export/model", { model: state.scene.id, colors: encoded, name }, `${name}.3dm`);
    setStatus(`saved ${name}.3dm: ${seen} of ${total} vertices seen`);
  } catch (e) {
    setProgress(null);
    setStatus(e.message, true);
  }
});

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

$("model").addEventListener("change", () => {
  fillMethods();
  updateControls();
  clearResult();
  state.sweep = null;         // another model: the lines of the last one would mislead
  state.highlight = null;
  drawSweep();
  drawFrames();
  scheduleRun();
});
$("method").addEventListener("change", () => {
  updateControls();
  scheduleRun();
});
// parameters: the number follows the slider, the run starts on release
for (const id of ["resolution", "overlap", "occluder", "steps", "smooth", "k", "temperature", "window"]) {
  $(id).addEventListener("input", updateControls);
  $(id).addEventListener("change", scheduleRun);
}
for (const id of ["reference", "baseline", "compete", "template"]) {
  $(id).addEventListener("change", () => { updateControls(); scheduleRun(); });
}
$("texts").addEventListener("change", scheduleRun);
$("texts").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    startRun();
  }
});
for (const id of ["style", "colormap", "range", "signed", "opacity", "gamma", "threshold", "smooth-display"]) {
  $(id).addEventListener("input", restyle);
}

// background of the views (remembered in this browser, shared with the other pages)
function setBackground(color) {
  document.documentElement.style.setProperty("--map-bg", color);
  try { localStorage.setItem("map-bg", color); } catch { /* storage blocked: fine */ }
  restyle();
}
$("map-bg").addEventListener("input", () => setBackground($("map-bg").value));
try {
  const saved = localStorage.getItem("map-bg");
  if (saved) {
    $("map-bg").value = saved;
    document.documentElement.style.setProperty("--map-bg", saved);
  }
} catch { /* storage blocked: default background */ }

$("rescan").addEventListener("click", async () => {
  setStatus("scanning models…");
  try {
    state.options = await postJSON("/api/rescan", {});
    fillModels($("model").value);
    fillMethods();
    updateControls();
    setStatus(`${state.options.models.length} models`);
  } catch (e) {
    setStatus(e.message, true);
  }
});

async function init() {
  state.options = await api("/api/options");
  fillModels(state.options.default_model);
  fillMethods();
  $("overlap").max = state.options.max_overlap;
  $("window").min = state.options.min_tile;
  updateControls();
  buildCards();
  requestDraw();
  setStatus("upload an image");
}

init().catch((e) => setStatus(e.message, true));
