// Curve fitter front end.
//
// The page keeps the model as a list of layer descriptions:
//   {id, type: "linear",   out: 16, params: {weight, bias} | null}
//   {id, type: "relu"}
//   {id, type: "function", expr: "a*cos(x)", params: {a: {value, min, max}}}
// Every change is sent to /api/evaluate, which builds the torch model, reports
// the shape after each layer and returns the curve it draws. /api/optimize runs
// Adam in the background and /api/status is polled for the loss and parameters.

const $ = (sel) => document.querySelector(sel);

const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
const lossCanvas = $("#loss-chart");
const lossCtx = lossCanvas.getContext("2d");

const VIEW = 1.2;              // the canvas shows [-VIEW, VIEW] in x and y
const DEFAULT_RANGE = 5;       // slider range for a new formula parameter

let layers = [];
let selectedId = null;
let nextId = 1;

let stroke = [];               // raw drawn points (world coordinates)
let points = [];               // stroke resampled to N points: the target
let curve = null;              // model curve, finely sampled
let pred = null;               // model output at the N target t values
let info = [];                 // per-layer report from the server
let modelError = "";
let currentLoss = null;

let losses = [];               // loss history of the current optimization
let running = false;


// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

function param(value, min = -DEFAULT_RANGE, max = DEFAULT_RANGE) {
  return { value, min, max };
}

const PRESETS = {
  fourier: () => [{
    type: "function",
    expr: "a1*cos(2*pi*f1*x + p1) + a2*cos(2*pi*f2*x + p2) + cx;\nb1*sin(2*pi*f1*x + p1) + b2*sin(2*pi*f2*x + p2) + cy",
    params: {
      a1: param(0.6, -2, 2), f1: param(1), p1: param(0, -4, 4), a2: param(0.2, -2, 2), f2: param(2), p2: param(0, -4, 4),
      cx: param(0, -1, 1), b1: param(0.6, -2, 2), b2: param(0.2, -2, 2), cy: param(0, -1, 1),
    },
  }],
  mlp: () => [
    { type: "linear", out: 32 }, { type: "relu" },
    { type: "linear", out: 32 }, { type: "relu" },
    { type: "linear", out: 2 },
  ],
  mixed: () => [
    {
      type: "function",
      expr: "cos(2*pi*f1*x); sin(2*pi*f1*x)\ncos(2*pi*f2*x); sin(2*pi*f2*x)\ncos(2*pi*f3*x); sin(2*pi*f3*x)",
      params: { f1: param(1, 0, 8), f2: param(2, 0, 8), f3: param(3, 0, 8) },
    },
    { type: "linear", out: 2 },
  ],
};

function loadPreset(name) {
  layers = PRESETS[name]().map((l) => ({ params: null, ...l, id: nextId++ }));
  selectedId = layers[0].id;
  renderLayers();
  renderProps();
  evaluate();
}


// ---------------------------------------------------------------------------
// Talking to the server
// ---------------------------------------------------------------------------

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function payloadLayers() {
  return layers.map(({ id, type, out, expr, params }) => ({ id, type, out, expr, params }));
}

let evalTimer = null;
let evalSeq = 0;

function evaluate(delay = 0) {
  clearTimeout(evalTimer);
  evalTimer = setTimeout(async () => {
    const seq = ++evalSeq;
    const res = await post("/api/evaluate", { layers: payloadLayers(), points, loss: $("#loss").value });
    if (seq !== evalSeq || running) return;      // a newer request is on its way
    applyLayerInfo(res.layers, false);
    modelError = res.layers.find((l) => l.error)?.error || res.error || "";
    curve = res.ok ? res.curve : null;
    pred = res.ok ? res.pred : null;
    currentLoss = res.ok && res.loss !== undefined ? res.loss : null;
    renderLayers();
    renderDrawing();
    renderLossValue();
  }, delay);
}

// Copy what the server reports (shapes, parameter values) into the layers.
// fromOptimizer: the server's values win; otherwise the page's values win and
// only new parameters are taken over.
function applyLayerInfo(report, fromOptimizer) {
  info = report;
  let namesChanged = false;
  for (const entry of report) {
    const layer = layers.find((l) => l.id === entry.id);
    if (!layer) continue;
    if (layer.type === "linear" && entry.weight) {
      layer.params = { weight: entry.weight, bias: entry.bias };
    }
    if (layer.type === "function" && entry.names) {
      const old = layer.params || {};
      const params = {};
      for (const name of entry.names) {
        const v = entry.values[name];
        if (old[name]) {
          params[name] = old[name];
          if (fromOptimizer) {
            params[name].value = v;
            params[name].min = Math.min(params[name].min, v);
            params[name].max = Math.max(params[name].max, v);
          }
        } else {
          params[name] = param(v, Math.min(-DEFAULT_RANGE, v), Math.max(DEFAULT_RANGE, v));
        }
      }
      if (Object.keys(old).join() !== entry.names.join()) namesChanged = true;
      layer.params = params;
    }
  }
  if (namesChanged) renderParamList();
  else updateParamValues();
  if (selectedLayer()?.type === "linear") renderWeights();
}


// ---------------------------------------------------------------------------
// Layer list
// ---------------------------------------------------------------------------

const TYPE_NAMES = { linear: "Linear", relu: "ReLU", function: "Function" };

function selectedLayer() {
  return layers.find((l) => l.id === selectedId);
}

function renderLayers() {
  const list = $("#layers");
  list.innerHTML = "";

  const input = document.createElement("li");
  input.className = "io";
  input.textContent = "input t  [N, 1]";
  list.append(input);

  let broken = false;
  for (const layer of layers) {
    const entry = info.find((e) => e.id === layer.id);
    const li = document.createElement("li");
    li.className = "layer";
    if (layer.id === selectedId) li.classList.add("selected");
    if (entry?.error) li.classList.add("broken");

    let detail = "";
    if (layer.type === "linear") detail = `out ${layer.out}`;
    if (layer.type === "function") detail = layer.expr.replace(/\s*\n\s*/g, " ; ");

    let shape = "";
    if (entry?.error) { shape = "error"; broken = true; }
    else if (entry?.out !== undefined && !broken) shape = `→ [N, ${entry.out}]`;

    li.innerHTML = `<span class="name"></span><span class="detail"></span><span class="shape"></span>
      <button class="del" title="Delete layer">×</button>`;
    li.querySelector(".name").textContent = TYPE_NAMES[layer.type];
    li.querySelector(".detail").textContent = detail;
    li.querySelector(".shape").textContent = shape;
    li.title = entry?.error || "";
    li.addEventListener("click", () => {
      selectedId = layer.id;
      renderLayers();
      renderProps();
    });
    li.querySelector(".del").addEventListener("click", (ev) => {
      ev.stopPropagation();
      layers = layers.filter((l) => l !== layer);
      if (selectedId === layer.id) selectedId = null;
      renderLayers();
      renderProps();
      evaluate();
    });
    list.append(li);
  }

  const last = info.length ? info[info.length - 1] : null;
  const status = $("#model-status");
  if (!layers.length) {
    status.textContent = "add layers that end with [N, 2]";
    status.className = "status error";
  } else if (modelError) {
    status.textContent = modelError;
    status.className = "status error";
  } else if (last?.out === 2) {
    status.textContent = "output [N, 2] ✓";
    status.className = "status";
  } else {
    status.textContent = "";
  }
  $("#optimize").disabled = !running && (!!modelError || !layers.length || points.length < 2);
}

function addLayer(type) {
  const layer = { id: nextId++, type, params: null };
  if (type === "linear") layer.out = 2;
  if (type === "function") layer.expr = "a1*cos(x*f1) + a2*sin(x*f2)";
  // Insert after the selected layer, or at the end.
  const at = layers.indexOf(selectedLayer());
  layers.splice(at < 0 ? layers.length : at + 1, 0, layer);
  selectedId = layer.id;
  renderLayers();
  renderProps();
  evaluate();
}


// ---------------------------------------------------------------------------
// Properties panel
// ---------------------------------------------------------------------------

function renderProps() {
  const props = $("#props");
  const layer = selectedLayer();
  props.innerHTML = "";
  if (!layer) {
    props.innerHTML = `<p class="hint">Select a layer to edit it.</p>`;
    return;
  }

  if (layer.type === "relu") {
    props.innerHTML = `<p class="note">ReLU: max(0, x) on every value. No parameters, keeps the shape.</p>`;
  }

  if (layer.type === "linear") {
    props.innerHTML = `
      <label>out features <input id="out-features" type="number" min="1" max="512"></label>
      <p class="note">y = x·Wᵀ + b. In features come from the previous layer.
        Changing the size starts from new random weights.</p>
      <div id="weights" class="weights"></div>`;
    const input = $("#out-features");
    input.value = layer.out;
    input.addEventListener("change", () => {
      layer.out = Math.max(1, parseInt(input.value) || 1);
      input.value = layer.out;
      renderLayers();
      evaluate();
    });
    renderWeights();
  }

  if (layer.type === "function") {
    props.innerHTML = `
      <textarea id="expr" spellcheck="false"></textarea>
      <p class="note">x is the layer input [N, D], x0, x1, … its columns. Separate formulas with ; or a new
        line, their outputs are placed side by side. Known: sin cos tan exp log sqrt abs tanh sigmoid relu
        sinh cosh atan2 minimum maximum pow sq softplus, pi, e. Any other name is a parameter.</p>
      <div id="param-list"></div>`;
    const expr = $("#expr");
    expr.value = layer.expr;
    expr.addEventListener("input", () => {
      layer.expr = expr.value;
      renderLayers();
      evaluate(250);
    });
    renderParamList();
  }
}

function renderWeights() {
  const box = $("#weights");
  const layer = selectedLayer();
  if (!box || !layer?.params?.weight) return;
  const fmt = (v) => v.toFixed(3);
  const rows = layer.params.weight.map((row, i) =>
    `<tr>${row.map((v) => `<td>${fmt(v)}</td>`).join("")}<td>│ ${fmt(layer.params.bias[i])}</td></tr>`);
  box.innerHTML = `<p class="note">W [${layer.params.weight.length} × ${layer.params.weight[0].length}] │ b</p>
    <table>${rows.join("")}</table>`;
}

// The sliders of a formula layer; rebuilt only when the parameter names change
// so that dragging a slider is never interrupted.
function renderParamList() {
  const box = $("#param-list");
  const layer = selectedLayer();
  if (!box || layer?.type !== "function") return;
  box.innerHTML = "";

  const entry = info.find((e) => e.id === layer.id);
  if (entry?.error) {
    box.innerHTML = `<p class="error"></p>`;
    box.querySelector(".error").textContent = entry.error;
  }

  for (const [name, p] of Object.entries(layer.params || {})) {
    const row = document.createElement("div");
    row.className = "param";
    row.dataset.name = name;
    row.innerHTML = `
      <span class="pname"></span>
      <input class="slider" type="range">
      <input class="value" type="number" step="any">
      <span class="range">
        min <input class="min" type="number" step="any">
        max <input class="max" type="number" step="any">
      </span>`;
    row.querySelector(".pname").textContent = name;
    const slider = row.querySelector(".slider");
    const value = row.querySelector(".value");
    const min = row.querySelector(".min");
    const max = row.querySelector(".max");

    const sync = () => {
      slider.min = p.min;
      slider.max = p.max;
      slider.step = (p.max - p.min) / 1000 || 0.001;
      slider.value = p.value;
      value.value = round(p.value);
      min.value = round(p.min);
      max.value = round(p.max);
    };
    row.sync = sync;
    sync();

    slider.addEventListener("input", () => {
      p.value = parseFloat(slider.value);
      value.value = round(p.value);
      evaluate();
    });
    value.addEventListener("change", () => {
      const v = parseFloat(value.value);
      if (Number.isFinite(v)) {
        p.value = v;
        p.min = Math.min(p.min, v);
        p.max = Math.max(p.max, v);
      }
      sync();
      evaluate();
    });
    const setRange = () => {
      const lo = parseFloat(min.value), hi = parseFloat(max.value);
      if (Number.isFinite(lo) && Number.isFinite(hi) && lo < hi) {
        p.min = lo;
        p.max = hi;
        p.value = Math.min(hi, Math.max(lo, p.value));
      }
      sync();
      evaluate();
    };
    min.addEventListener("change", setRange);
    max.addEventListener("change", setRange);
    box.append(row);
  }
}

function updateParamValues() {
  const box = $("#param-list");
  if (!box) return;
  // Refresh the error line, then every slider.
  const layer = selectedLayer();
  const entry = info.find((e) => e.id === layer?.id);
  const err = box.querySelector(".error");
  if (entry?.error && !err) return renderParamList();
  if (err && !entry?.error) err.remove();
  if (err && entry?.error) err.textContent = entry.error;
  for (const row of box.querySelectorAll(".param")) {
    if (row.contains(document.activeElement) && !running) continue;
    row.sync?.();
  }
}

function round(v) {
  return Math.round(v * 10000) / 10000;
}


// ---------------------------------------------------------------------------
// Drawing canvas
// ---------------------------------------------------------------------------

function resizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = Math.round(rect.width * dpr);
  c.height = Math.round(rect.height * dpr);
  c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

function toScreen([x, y]) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  return [(x / VIEW + 1) * w / 2, (1 - y / VIEW) * h / 2];
}

function toWorld(ev) {
  const rect = canvas.getBoundingClientRect();
  const sx = (ev.clientX - rect.left) / rect.width;
  const sy = (ev.clientY - rect.top) / rect.height;
  return [(sx * 2 - 1) * VIEW, (1 - sy * 2) * VIEW];
}

// N points evenly spaced along the length of the stroke.
function resample(path, n) {
  if (path.length < 2) return [];
  const cum = [0];
  for (let i = 1; i < path.length; i++) {
    cum.push(cum[i - 1] + Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]));
  }
  const total = cum[cum.length - 1];
  if (total === 0) return [];
  const out = [];
  let j = 0;
  for (let i = 0; i < n; i++) {
    const d = total * i / (n - 1);
    while (j < path.length - 2 && cum[j + 1] < d) j++;
    const seg = cum[j + 1] - cum[j];
    const u = seg > 0 ? (d - cum[j]) / seg : 0;
    out.push([
      path[j][0] + u * (path[j + 1][0] - path[j][0]),
      path[j][1] + u * (path[j + 1][1] - path[j][1]),
    ]);
  }
  return out;
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function polyline(c, pts) {
  c.beginPath();
  pts.forEach((p, i) => {
    const [sx, sy] = toScreen(p);
    i ? c.lineTo(sx, sy) : c.moveTo(sx, sy);
  });
  c.stroke();
}

function renderDrawing() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  // grid at every 0.5, axes a little stronger
  ctx.lineWidth = 1;
  for (let v = -1; v <= 1.001; v += 0.5) {
    ctx.strokeStyle = Math.abs(v) < 1e-9 ? css("--canvas-axis") : css("--canvas-grid");
    polyline(ctx, [[v, -VIEW], [v, VIEW]]);
    polyline(ctx, [[-VIEW, v], [VIEW, v]]);
  }

  if (drawing && stroke.length > 1) {
    ctx.strokeStyle = css("--target");
    ctx.lineWidth = 2;
    polyline(ctx, stroke);
    return;
  }

  // matches: target point i  <->  model output at t_i
  if ($("#show-links").checked && pred && pred.length === points.length) {
    ctx.strokeStyle = css("--link");
    ctx.lineWidth = 1;
    ctx.beginPath();
    points.forEach((p, i) => {
      const [ax, ay] = toScreen(p), [bx, by] = toScreen(pred[i]);
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    });
    ctx.stroke();
  }

  // target points, the first one larger (t = 0)
  ctx.fillStyle = css("--target");
  points.forEach((p, i) => {
    const [sx, sy] = toScreen(p);
    ctx.beginPath();
    ctx.arc(sx, sy, i === 0 ? 5 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  });

  if (curve) {
    ctx.strokeStyle = css("--model");
    ctx.lineWidth = 2;
    polyline(ctx, curve);
  }
  if (pred && $("#show-links").checked) {
    ctx.fillStyle = css("--model");
    for (const p of pred) {
      const [sx, sy] = toScreen(p);
      ctx.beginPath();
      ctx.arc(sx, sy, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  $("#draw-hint").textContent = points.length
    ? `${points.length} target points (white), the big dot is t = 0. Model curve in cyan.`
    : "Drag on the canvas to draw a curve.";
}

let drawing = false;

canvas.addEventListener("pointerdown", (ev) => {
  if (running) return;
  drawing = true;
  canvas.setPointerCapture(ev.pointerId);
  stroke = [toWorld(ev)];
  renderDrawing();
});
canvas.addEventListener("pointermove", (ev) => {
  if (!drawing) return;
  stroke.push(toWorld(ev));
  renderDrawing();
});
canvas.addEventListener("pointerup", () => {
  if (!drawing) return;
  drawing = false;
  updatePoints();
});

function updatePoints() {
  const n = Math.max(2, parseInt($("#n-points").value) || 100);
  points = resample(stroke, n);
  pred = null;
  renderDrawing();
  renderLayers();
  evaluate();
}


// ---------------------------------------------------------------------------
// Optimization
// ---------------------------------------------------------------------------

async function startOptimize() {
  const res = await post("/api/optimize", {
    layers: payloadLayers(),
    points,
    steps: parseInt($("#steps").value) || 1000,
    lr: parseFloat($("#lr").value) || 0.01,
    clamp: $("#clamp").checked,
    optimizer: $("#optimizer").value,
    loss: $("#loss").value,
    settings: optimizerSettings[$("#optimizer").value],
  });
  if (res.error) {
    $("#train-status").textContent = res.error;
    return;
  }
  losses = [];
  setRunning(true);
  poll();
}

async function poll() {
  const res = await (await fetch(`/api/status?since=${losses.length}`)).json();
  losses.push(...res.losses);
  if (res.snapshot) {
    applyLayerInfo(res.snapshot.layers, true);
    curve = res.snapshot.curve;
    pred = res.snapshot.pred;
    renderDrawing();
  }
  currentLoss = losses.length ? losses[losses.length - 1] : null;
  renderLossValue();
  renderLossChart();
  $("#train-status").textContent = res.error ? res.error : `step ${res.step} / ${$("#steps").value}`;

  if (res.running) {
    setTimeout(poll, 100);
  } else {
    setRunning(false);
    renderLayers();
    evaluate();            // the exact loss of the final parameters
  }
}

function setRunning(on) {
  running = on;
  const btn = $("#optimize");
  btn.textContent = on ? "Stop" : "Optimize";
  btn.classList.toggle("stop", on);
  btn.disabled = false;
  $(".model").inert = on;
  for (const id of ["#clear", "#n-points", "#reinit", "#optimizer", "#loss", "#steps", "#lr", "#clamp"]) $(id).disabled = on;
  for (const input of document.querySelectorAll("#opt-settings input")) input.disabled = on;
  canvas.style.cursor = on ? "default" : "crosshair";
}

$("#optimize").addEventListener("click", () => {
  if (running) post("/api/stop", {});
  else startOptimize();
});

$("#reinit").addEventListener("click", () => {
  for (const layer of layers) {
    if (layer.type === "linear") layer.params = null;
    if (layer.type === "function") for (const p of Object.values(layer.params || {})) {
      p.value = Math.min(p.max, Math.max(p.min, 1));
    }
  }
  updateParamValues();
  evaluate();
});


// ---------------------------------------------------------------------------
// Loss chart: log scale, one line, hover for the exact value
// ---------------------------------------------------------------------------

const CHART_PAD = { left: 44, right: 10, top: 8, bottom: 22 };
let chartScale = null;

function renderLossValue() {
  $("#loss-value").textContent = currentLoss === null ? "–" : formatLoss(currentLoss);
}

function formatLoss(v) {
  return v >= 0.01 ? v.toFixed(4) : v.toExponential(2);
}

function renderLossChart(hoverIndex = null) {
  const rect = resizeCanvas(lossCanvas);
  const w = rect.width, h = rect.height;
  const c = lossCtx;
  c.clearRect(0, 0, w, h);
  c.font = `11px ${css("--font")}`;

  const data = losses.filter((v) => v > 0 && Number.isFinite(v));
  if (data.length < 2) {
    c.fillStyle = css("--ink-3");
    c.fillText("press Optimize to see the loss", CHART_PAD.left, h / 2);
    chartScale = null;
    return;
  }

  let lo = Math.floor(Math.log10(Math.min(...data)));
  let hi = Math.ceil(Math.log10(Math.max(...data)));
  if (hi === lo) hi = lo + 1;
  const pw = w - CHART_PAD.left - CHART_PAD.right;
  const ph = h - CHART_PAD.top - CHART_PAD.bottom;
  const X = (i) => CHART_PAD.left + (i / (losses.length - 1)) * pw;
  const Y = (v) => CHART_PAD.top + (1 - (Math.log10(v) - lo) / (hi - lo)) * ph;
  chartScale = { X, Y, pw };

  // grid: one line per decade
  c.strokeStyle = css("--grid");
  c.fillStyle = css("--ink-3");
  c.lineWidth = 1;
  c.textAlign = "right";
  c.textBaseline = "middle";
  for (let d = lo; d <= hi; d++) {
    const y = Y(10 ** d);
    c.beginPath();
    c.moveTo(CHART_PAD.left, y);
    c.lineTo(w - CHART_PAD.right, y);
    c.stroke();
    c.fillText(`1e${d}`, CHART_PAD.left - 6, y);
  }
  c.textAlign = "left";
  c.textBaseline = "alphabetic";
  c.fillText("step 0", CHART_PAD.left, h - 6);
  c.textAlign = "right";
  c.fillText(`${losses.length}`, w - CHART_PAD.right, h - 6);

  // the loss, skipping points closer than half a pixel
  c.strokeStyle = css("--accent");
  c.lineWidth = 2;
  c.lineJoin = "round";
  c.beginPath();
  const every = Math.max(1, Math.floor(losses.length / (pw * 2)));
  let started = false;
  for (let i = 0; i < losses.length; i += every) {
    const v = losses[i];
    if (!(v > 0)) continue;
    started ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v));
    started = true;
  }
  const lastI = losses.length - 1;
  if (losses[lastI] > 0) c.lineTo(X(lastI), Y(losses[lastI]));
  c.stroke();

  // hover crosshair
  if (hoverIndex !== null && losses[hoverIndex] > 0) {
    const x = X(hoverIndex), y = Y(losses[hoverIndex]);
    c.strokeStyle = css("--ink-3");
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(x, CHART_PAD.top);
    c.lineTo(x, h - CHART_PAD.bottom);
    c.stroke();
    c.fillStyle = css("--accent");
    c.strokeStyle = css("--surface");
    c.lineWidth = 2;
    c.beginPath();
    c.arc(x, y, 4, 0, Math.PI * 2);
    c.fill();
    c.stroke();
  }
}

lossCanvas.addEventListener("pointermove", (ev) => {
  const tip = $("#tooltip");
  if (!chartScale || losses.length < 2) return;
  const rect = lossCanvas.getBoundingClientRect();
  const u = (ev.clientX - rect.left - CHART_PAD.left) / chartScale.pw;
  const i = Math.round(Math.min(1, Math.max(0, u)) * (losses.length - 1));
  renderLossChart(i);
  tip.hidden = false;
  tip.textContent = `step ${i + 1}  ·  ${formatLoss(losses[i])}`;
  const x = chartScale.X(i);
  tip.style.left = `${Math.min(x + 8, rect.width - tip.offsetWidth)}px`;
  tip.style.top = "4px";
});
lossCanvas.addEventListener("pointerleave", () => {
  $("#tooltip").hidden = true;
  renderLossChart();
});


// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

$("#add-btn").addEventListener("click", (ev) => {
  ev.stopPropagation();
  $("#add-menu").hidden = !$("#add-menu").hidden;
});
for (const btn of document.querySelectorAll("#add-menu button")) {
  btn.addEventListener("click", () => {
    $("#add-menu").hidden = true;
    addLayer(btn.dataset.type);
  });
}
document.addEventListener("click", () => { $("#add-menu").hidden = true; });

$("#preset").addEventListener("change", (ev) => {
  if (ev.target.value) loadPreset(ev.target.value);
  ev.target.value = "";
});

$("#clear").addEventListener("click", () => {
  stroke = [];
  points = [];
  pred = null;
  currentLoss = null;
  renderDrawing();
  renderLayers();
  renderLossValue();
});
$("#n-points").addEventListener("change", updatePoints);
$("#show-links").addEventListener("change", renderDrawing);
$("#optimizer").addEventListener("change", renderOptimizerSettings);
$("#loss").addEventListener("change", () => {
  $("#loss-label").textContent = `${$("#loss").value} loss`;
  losses = [];             // the old curve measured something else
  renderLossChart();
  evaluate();
});

window.addEventListener("resize", () => {
  resizeCanvas(canvas);
  renderDrawing();
  renderLossChart();
});

// Momentum-like settings of each optimizer (beta1/beta2, momentum, ...),
// remembered per optimizer while switching between them.
let optimizerSettings = {};

async function loadOptimizers() {
  const list = await (await fetch("/api/optimizers")).json();
  optimizerSettings = Object.fromEntries(list.map((o) => [o.name, { ...o.settings }]));
  $("#optimizer").innerHTML = list.map((o) => `<option>${o.name}</option>`).join("");
  renderOptimizerSettings();
}

function renderOptimizerSettings() {
  const box = $("#opt-settings");
  box.innerHTML = "";
  const settings = optimizerSettings[$("#optimizer").value] || {};
  for (const [key, value] of Object.entries(settings)) {
    const label = document.createElement("label");
    label.innerHTML = `<span></span><input type="number" step="0.01" min="0" max="0.9999">`;
    label.querySelector("span").textContent = key;
    const input = label.querySelector("input");
    input.value = value;
    input.disabled = running;
    input.addEventListener("change", () => {
      const v = parseFloat(input.value);
      if (Number.isFinite(v)) settings[key] = v;
      input.value = settings[key];
    });
    box.append(label);
  }
}

resizeCanvas(canvas);
renderLossChart();
loadOptimizers();
loadPreset("fourier");
