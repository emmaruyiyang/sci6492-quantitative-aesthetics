// Classifier Map front end.
//
// The server gives, for every image, its logits (raw class scores). Everything
// else happens here: logits -> softmax or sigmoid scores -> position on the map.
// Positions are applied as a CSS transform, and the .item transition in the
// stylesheet makes the images glide whenever a setting changes.

const $ = (id) => document.getElementById(id);

const state = {
  classNames: [],
  items: new Map(),     // id -> { data: {id, name, thumbnail, logits}, el: <img> }
  probe: null,          // { logits, el } for the live drawing
  hovered: null,        // data of the image shown in the scores panel
};

const settings = () => ({
  softmax: $("softmax").checked,
  biaxial: $("biaxial").checked,
  x: Number($("x-class").value),
  y: Number($("y-class").value),
  thumb: Number($("thumb-size").value),
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

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function countText() {
  return `${state.classNames.length} classes, ${state.items.size} images`;
}

// ---------------------------------------------------------------------------
// Scores and projections
// ---------------------------------------------------------------------------

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

const sigmoid = (logits) => logits.map((v) => 1 / (1 + Math.exp(-v)));

function scoresOf(logits) {
  return settings().softmax ? softmax(logits) : sigmoid(logits);
}

// Radial: unit vector of each class axis, all at equal angles, the first one
// pointing up. (Screen y grows downwards, so "up" is -y.)
function axisDirections(count) {
  return Array.from({ length: count }, (_, k) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * k) / count;
    return [Math.cos(angle), Math.sin(angle)];
  });
}

// How far from the center an image can possibly land, in axis lengths.
// Softmax scores add up to 1, so the point stays inside the polygon of axis
// tips: at most 1. Sigmoid scores are independent, several neighbouring
// classes can all be near 1 and their axes add up to more than 1. The largest
// sum comes from a run of neighbouring axes, so try them all.
function maxReach(count, useSoftmax) {
  if (useSoftmax || count < 2) return 1;
  const dirs = axisDirections(count);
  let best = 1;
  for (let start = 0; start < count; start++) {
    let x = 0, y = 0;
    for (let n = 0; n < count; n++) {
      const [dx, dy] = dirs[(start + n) % count];
      x += dx; y += dy;
      best = Math.max(best, Math.hypot(x, y));
    }
  }
  return best;
}

const PAD = 0.1;            // biaxial: margin around the 0..1 square (fraction of the map)
const RADIUS = 0.4;         // radial: map fraction reachable from the center

// Scores -> position as a fraction of the map, (0, 0) top left, (1, 1) bottom right
function project(scores) {
  const s = settings();
  if (s.biaxial) {
    return [PAD + scores[s.x] * (1 - 2 * PAD), 1 - PAD - scores[s.y] * (1 - 2 * PAD)];
  }
  // center + s_1 * axis_1 + s_2 * axis_2 + ...
  const length = RADIUS / maxReach(scores.length, s.softmax);
  const dirs = axisDirections(scores.length);
  let x = 0.5, y = 0.5;
  scores.forEach((score, k) => {
    x += score * length * dirs[k][0];
    y += score * length * dirs[k][1];
  });
  return [x, y];
}

// ---------------------------------------------------------------------------
// Drawing the map
// ---------------------------------------------------------------------------

function place(el, logits) {
  const size = $("map").clientWidth;
  const [fx, fy] = project(scoresOf(logits));
  el.style.width = `${settings().thumb}px`;
  // translate(-50%, -50%) centers the image on its point
  el.style.transform = `translate(${fx * size}px, ${fy * size}px) translate(-50%, -50%)`;
}

function layout() {
  for (const { data, el } of state.items.values()) place(el, data.logits);
  if (state.probe) place(state.probe.el, state.probe.logits);
  drawAxes();
  if (state.hovered) showScores(state.hovered);
}

const SVG = "http://www.w3.org/2000/svg";

function svg(tag, attributes, text) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attributes)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  $("axes").appendChild(el);
  return el;
}

// The axes svg uses a 0..1000 coordinate system over the whole map
function drawAxes() {
  $("axes").innerHTML = "";
  const s = settings();
  const names = state.classNames;
  if (!names.length) return;
  const P = (f) => f * 1000;

  if (s.biaxial) {
    const lo = P(PAD), hi = P(1 - PAD);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      const v = lo + t * (hi - lo);
      svg("line", { x1: v, y1: lo, x2: v, y2: hi, class: t === 0 ? "axis" : "" });
      svg("line", { x1: lo, y1: hi - (v - lo), x2: hi, y2: hi - (v - lo), class: t === 0 ? "axis" : "" });
      svg("text", { x: v, y: hi + 24, "text-anchor": "middle", class: "tick" }, t);
      svg("text", { x: lo - 10, y: hi - (v - lo) + 5, "text-anchor": "end", class: "tick" }, t);
    }
    if (s.softmax) {
      // softmax scores add up to 1, so x + y <= 1: everything is below this
      // line from (x=0, y=1) to (x=1, y=0)
      svg("line", { x1: lo, y1: lo, x2: hi, y2: hi, class: "dashed" });
    }
    svg("text", { x: hi, y: hi + 56, "text-anchor": "end" }, `${names[s.x]} →`);
    svg("text", { x: lo - 70, y: lo, "text-anchor": "end", transform: `rotate(-90 ${lo - 70} ${lo})` },
      `${names[s.y]} →`);
    return;
  }

  // radial: one axis per class, ending where a score of 1 lands
  const length = RADIUS / maxReach(names.length, s.softmax);
  const dirs = axisDirections(names.length);
  const tips = dirs.map(([dx, dy]) => [P(0.5 + dx * length), P(0.5 + dy * length)]);
  svg("polygon", { points: tips.map((t) => t.join(",")).join(" "), class: "dashed" });
  svg("circle", { cx: 500, cy: 500, r: 4, class: "axis" });
  dirs.forEach(([dx, dy], k) => {
    const [tx, ty] = tips[k];
    svg("line", { x1: 500, y1: 500, x2: tx, y2: ty, class: "axis" });
    // labels of side axes run inwards from the tip so they stay on the map;
    // below the tip for axes pointing down, above it otherwise
    const anchor = Math.abs(dx) < 0.2 ? "middle" : dx > 0 ? "end" : "start";
    svg("text", { x: tx, y: ty + (dy > 0.2 ? 32 : -14), "text-anchor": anchor }, names[k]);
  });
}

function updateHint() {
  const s = settings();
  let text;
  if (s.biaxial) {
    text = s.softmax
      ? "Softmax: x + y can't exceed 1, images stay below the dashed diagonal."
      : "Sigmoid: each class scored on its own, the whole square can be used.";
  } else {
    text = "Each image = center + Σ score × class axis. Strong in one class: near that axis tip. "
      + (s.softmax
        ? "Softmax: inside the dashed polygon, center = undecided."
        : "Sigmoid: strong in all or none of the classes also ends up near the center.");
  }
  $("projection-hint").textContent = text;
}

// ---------------------------------------------------------------------------
// Images on the map
// ---------------------------------------------------------------------------

// New image element in the center of the map. Without a starting position its
// first transition would start from the top left corner.
function newMapImage(className, src, alt) {
  const el = document.createElement("img");
  el.className = className;
  el.src = src;
  el.alt = alt;
  el.style.width = `${settings().thumb}px`;
  const size = $("map").clientWidth;
  el.style.transform = `translate(${size / 2}px, ${size / 2}px) translate(-50%, -50%)`;
  $("items").appendChild(el);
  return el;
}

// Wait until the browser has drawn the element at its start, then move it,
// so it glides out to its place.
const glide = (el, logits) => requestAnimationFrame(() => requestAnimationFrame(() => place(el, logits)));

function addItem(data) {
  const el = newMapImage("item", data.thumbnail, data.name);
  el.addEventListener("mouseenter", () => showScores(data));
  state.items.set(data.id, { data, el });
  glide(el, data.logits);
}

function setItems(list) {
  // keep the elements of images that are still there, so they glide
  const keep = new Set(list.map((d) => d.id));
  for (const [id, { el }] of state.items) {
    if (!keep.has(id)) { el.remove(); state.items.delete(id); }
  }
  for (const data of list) {
    const item = state.items.get(data.id);
    if (item) item.data = data;
    else addItem(data);
  }
}

function showScores(data) {
  state.hovered = data;
  $("info-empty").hidden = true;
  $("info-body").hidden = false;
  $("info-image").src = data.thumbnail;
  $("info-name").textContent = data.name;
  const scores = scoresOf(data.logits);
  const best = scores.indexOf(Math.max(...scores));
  $("info-bars").innerHTML = scores.map((score, k) => `
    <div class="bar-row${k === best ? " best" : ""}">
      <div class="label"><span>${state.classNames[k]}</span><span>${score.toFixed(2)}</span></div>
      <div class="track"><div class="fill" style="width:${(score * 100).toFixed(1)}%"></div></div>
    </div>`).join("");
}

async function uploadFiles(files) {
  const images = [...files].filter((f) => f.type.startsWith("image/"));
  if (!images.length) return;
  const form = new FormData();
  images.forEach((f) => form.append("files", f));
  setStatus(`classifying ${images.length} images…`);
  try {
    const result = await api("/api/images", { method: "POST", body: form });
    result.images.forEach(addItem);
    setStatus(countText());
  } catch (e) {
    setStatus(e.message, true);
  }
}

// ---------------------------------------------------------------------------
// Model and class selection
// ---------------------------------------------------------------------------

function fillClassSelect(select, names, preferred) {
  select.innerHTML = names.map((n, k) => `<option value="${k}">${n}</option>`).join("");
  select.value = String(preferred < names.length ? preferred : 0);
}

async function selectModel(name) {
  setStatus(`loading ${name}…`);
  $("model").disabled = true;
  try {
    const result = await postJSON("/api/model", { name });
    state.classNames = result.class_names;
    // keep the chosen axes when switching model (if the new one has as many classes)
    const s = settings();
    const first = !$("x-class").options.length;
    fillClassSelect($("x-class"), state.classNames, first ? 0 : s.x);
    fillClassSelect($("y-class"), state.classNames, first ? 1 : s.y);
    setItems(result.images);
    await classifyDrawing();     // the drawing needs the new model's scores too
    layout();
    setStatus(countText());
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    $("model").disabled = false;
  }
}

async function loadModels() {
  const result = await api("/api/models");
  $("model").innerHTML = result.models.map((m) => `<option>${m}</option>`).join("");
  if (!result.default) {
    setStatus("no models found", true);
    return;
  }
  $("model").value = result.default;
  await selectModel(result.default);
}

// ---------------------------------------------------------------------------
// Drawing canvas
// ---------------------------------------------------------------------------

const canvas = $("canvas");
const ctx = canvas.getContext("2d");
let drawing = false;

function clearCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function canvasPoint(event) {
  const r = canvas.getBoundingClientRect();
  return [(event.clientX - r.left) * canvas.width / r.width, (event.clientY - r.top) * canvas.height / r.height];
}

canvas.addEventListener("pointerdown", (event) => {
  drawing = true;
  canvas.setPointerCapture(event.pointerId);
  ctx.strokeStyle = $("brush-color").value;
  ctx.fillStyle = $("brush-color").value;
  ctx.lineWidth = Number($("brush-size").value);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const [x, y] = canvasPoint(event);
  ctx.beginPath();
  ctx.arc(x, y, ctx.lineWidth / 2, 0, 2 * Math.PI);     // a click makes a dot
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(x, y);
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  const [x, y] = canvasPoint(event);
  ctx.lineTo(x, y);
  ctx.stroke();
});

const stopDrawing = () => {
  if (!drawing) return;
  drawing = false;
  classifyDrawing();
};
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

async function classifyDrawing() {
  if (!state.classNames.length) return;
  const image = canvas.toDataURL("image/png");
  const { logits } = await postJSON("/api/classify", { image });
  if (!state.probe) {
    const el = newMapImage("item probe", image, "drawing");
    el.addEventListener("mouseenter", () => showScores(probeData()));
    state.probe = { el };
  }
  state.probe.logits = logits;
  state.probe.el.src = image;
  glide(state.probe.el, logits);
  if (state.hovered && state.hovered.id === "drawing") showScores(probeData());
}

function probeData() {
  return { id: "drawing", name: "drawing (not embedded)", thumbnail: state.probe.el.src, logits: state.probe.logits };
}

$("clear-canvas").addEventListener("click", () => {
  clearCanvas();
  classifyDrawing();
});

$("embed").addEventListener("click", async () => {
  try {
    const result = await postJSON("/api/embed", { image: canvas.toDataURL("image/png") });
    result.images.forEach(addItem);
    setStatus(countText());
  } catch (e) {
    setStatus(e.message, true);
  }
});

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function onSettingsChange() {
  const biaxial = $("biaxial").checked;
  $("x-class").disabled = !biaxial;
  $("y-class").disabled = !biaxial;
  updateHint();
  layout();
}

for (const id of ["softmax", "biaxial", "x-class", "y-class"]) {
  $(id).addEventListener("change", onSettingsChange);
}
$("thumb-size").addEventListener("input", layout);
$("model").addEventListener("change", () => selectModel($("model").value));

$("upload").addEventListener("click", () => $("files").click());
$("files").addEventListener("change", () => {
  uploadFiles($("files").files);
  $("files").value = "";      // so picking the same files again still triggers change
});

$("clear-map").addEventListener("click", async () => {
  await postJSON("/api/clear", {});
  setItems([]);
  setStatus(countText());
});

// drag and drop files on the map
const map = $("map");
map.addEventListener("dragover", (event) => { event.preventDefault(); map.classList.add("dragging"); });
map.addEventListener("dragleave", () => map.classList.remove("dragging"));
map.addEventListener("drop", (event) => {
  event.preventDefault();
  map.classList.remove("dragging");
  uploadFiles(event.dataTransfer.files);
});

// the map is sized by the window, positions are in pixels: redo them on resize
new ResizeObserver(() => layout()).observe(map);

clearCanvas();
onSettingsChange();
loadModels().catch((e) => setStatus(e.message, true));
