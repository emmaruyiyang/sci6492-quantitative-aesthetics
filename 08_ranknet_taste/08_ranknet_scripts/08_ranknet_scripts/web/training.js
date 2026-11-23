// Training tab: a form to start training an autoencoder, a classifier or a
// RankNet (on a ranking file of a dataset, see ranking.js), and a
// monitor that follows the run (GET /api/train/status about every second):
// progress, loss / accuracy curves, a preview of a few images after every
// epoch, and buttons to stop it, with or without saving.
//
// Uses the helpers of embedding_map.js ($, api, postJSON,
// diverging, applyOptions...), loaded before this file.

const SERIES = ["#2a78d6", "#eb6834"];      // train, evaluation (categorical slots 1, 2)

const train = {
  options: null,            // /api/train/options
  kind: "autoencoder",
  nameInfo: null,           // /api/train/model for the name in the field
  previewVersion: -1,
  lastState: null,
  timer: null,
};

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

const TAB_HINTS = {
  map: "Images placed by the vectors a model gives them, squeezed to 2D and grouped into clusters.",
  train: "Train a new autoencoder, classifier or RankNet, or keep training one.",
  data: "Make a new dataset from one (every image converted, or spectrograms of a sound), or rank its images in pairs.",
};
const VIEWS = ["map", "train", "data"];

function showView(view) {
  document.querySelectorAll(".tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === view);
    b.setAttribute("aria-selected", b.dataset.view === view);
  });
  for (const v of VIEWS) $(`view-${v}`).hidden = v !== view;
  $("tab-hint").textContent = TAB_HINTS[view];
  // reloaded every time: rankings saved in the Dataset tab appear in the RankNet datasets
  if (view === "train") loadTrainOptions();
  if (view === "data") dataViewShown();      // dataset.js
  pollSoon(0);
}

function updateTrainHint() {
  if (train.options && !$("view-train").hidden) {
    $("tab-hint").innerHTML = `${TAB_HINTS.train} Models are saved in `
      + `<span class="path">${escapeHTML(train.options.models_folder)}</span>.`;
  }
}

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));

const escapeHTML = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------------------------------------------------------------------------
// Form
// ---------------------------------------------------------------------------

async function loadTrainOptions() {
  try {
    train.options = await api("/api/train/options");
  } catch (e) {
    $("t-form-status").textContent = e.message;
    return;
  }
  const o = train.options;
  // reloaded (new datasets): keep what was picked
  if (!$("t-ae-type").options.length) fillSelect($("t-ae-type"), Object.keys(o.autoencoder_types).map((t) => [t, t]), "cnn");
  if (!$("t-backbone").options.length) {
    fillSelect($("t-backbone"), o.backbones.map((b) => [b.id, b.label]), "dinov2|dinov2_vitb14|None");
  }
  if (!$("t-rn-backbones").children.length) {
    for (const b of o.ranknet_backbones) {
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox" value="${escapeHTML(b.id)}"${b.id === "dino~dinov2_vitb14" ? " checked" : ""}> ${escapeHTML(b.label)}`;
      $("t-rn-backbones").appendChild(label);
    }
  }
  setKind(train.kind);
  if (!$("t-ae-options").children.length) aeTypeChanged();
  updateTrainHint();
}

function setKind(kind) {
  const previous = train.kind;
  train.kind = kind;
  document.querySelectorAll("#t-kind button").forEach((b) => b.classList.toggle("active", b.dataset.kind === kind));
  $("t-ae").hidden = kind !== "autoencoder";
  $("t-clf").hidden = kind !== "classifier";
  $("t-rn").hidden = kind !== "ranknet";
  // a RankNet trains on embeddings computed once: nothing to augment
  $("t-aug-panel").hidden = kind === "ranknet";
  $("t-ranking-row").hidden = $("t-ranking-info").hidden = kind !== "ranknet";
  const o = train.options;
  if (o) {
    if (kind === "classifier") {
      fillSelect($("t-dataset"), o.classifier_datasets.map((d) => [d.id, `${d.label} (${d.count})`]), $("t-dataset").value);
    } else if (kind === "ranknet") {
      fillDatasetSelect($("t-dataset"), o.ranknet_datasets, $("t-dataset").value);
    } else {
      fillDatasetSelect($("t-dataset"), o.autoencoder_datasets, $("t-dataset").value);
    }
  }
  if (previous !== kind) {
    $("t-batch").value = kind === "classifier" ? 32 : 64;
    // a RankNet epoch is a fraction of a second: many more of them
    if (kind === "ranknet") $("t-epochs").value = 200;
    else if (previous === "ranknet") $("t-epochs").value = 50;
    suggestLearningRate();
  }
  datasetChanged();
  checkName();
  updateFormState();
}

document.querySelectorAll("#t-kind button").forEach((b) => b.addEventListener("click", () => setKind(b.dataset.kind)));

// The learning rate that usually works, set whenever the choice it depends on changes
function suggestLearningRate() {
  if (train.kind === "autoencoder") {
    const type = train.options && train.options.autoencoder_types[$("t-ae-type").value];
    $("t-lr").value = type ? type.learning_rate : 0.001;
  } else if (train.kind === "ranknet") {
    $("t-lr").value = 0.001;
  } else {
    // a frozen backbone: only the small head learns and can take bigger steps
    $("t-lr").value = $("t-freeze").checked ? 0.001 : 0.0001;
  }
}

// One input per option of the chosen autoencoder kind, with its default value
function aeTypeChanged() {
  const type = train.options.autoencoder_types[$("t-ae-type").value];
  const box = $("t-ae-options");
  box.innerHTML = "";
  for (const [key, value] of Object.entries(type.defaults)) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.dataset.option = key;
    if (typeof value === "boolean") {
      input.type = "checkbox";
      input.checked = value;
      input.dataset.kind = "boolean";
      label.append(input, ` ${key.replaceAll("_", " ")}`);
    } else {
      label.className = "pick";
      input.type = Array.isArray(value) ? "text" : "number";
      input.step = "any";
      input.value = Array.isArray(value) ? value.join(", ") : value;
      input.dataset.kind = Array.isArray(value) ? "list" : "number";
      label.append(key.replaceAll("_", " "), " ", input);
    }
    box.appendChild(label);
  }
  $("t-ae-hint").textContent = AE_HINTS[$("t-ae-type").value] || "";
  suggestLearningRate();
}

const AE_HINTS = {
  mlp: "Fully connected layers on the flattened pixels.",
  cnn: "Convolutions that halve the image 'steps' times (the image size must be a multiple of 2^steps).",
  resnet: "The CNN with residual blocks after every halving and doubling.",
  vgg: "Encoder = pretrained VGG-16 (halves 5 times: image size a multiple of 32).",
  attention: "Patches + transformers. Needs a small learning rate.",
  vae: "Variational: the codes are pulled towards a standard normal (beta = weight of that pull).",
  sparse: "Codes >= 0, pushed towards 0: few active numbers per image.",
  vae_gan: "VAE + a discriminator that pushes for sharper images. Watch d_real / d_fake in the log.",
};

function aeOptions() {
  const options = {};
  for (const input of $("t-ae-options").querySelectorAll("input")) {
    const k = input.dataset.option;
    if (input.dataset.kind === "boolean") options[k] = input.checked;
    else if (input.dataset.kind === "list") options[k] = input.value.split(",").map((v) => Number(v.trim())).filter((v) => v > 0);
    else options[k] = Number(input.value);
  }
  return options;
}

function datasetChanged() {
  const o = train.options;
  if (!o) return;
  const id = $("t-dataset").value;
  if (train.kind === "classifier") {
    const d = o.classifier_datasets.find((x) => x.id === id);
    $("t-dataset-info").textContent = d ? `${d.classes.length} classes: ${d.classes.join(", ")}` : "no folders with class subfolders found";
  } else if (train.kind === "ranknet") {
    const d = o.ranknet_datasets.find((x) => x.id === id);
    $("t-dataset-info").textContent = d ? `${d.count} images${d.recursive ? " (subfolders included)" : ""}`
      : "no dataset has a ranking file yet: make one in the Dataset tab (Ranking)";
    const files = d ? d.rankings.filter((r) => !r.error) : [];
    fillSelect($("t-ranking"), files.map((r) => [r.file, `${r.label} (${r.pairs} pairs)`]), $("t-ranking").value);
    rankingChanged();
  } else {
    const d = o.autoencoder_datasets.find((x) => x.id === id);
    $("t-dataset-info").textContent = d ? `${d.count} images${d.recursive ? " (subfolders included)" : ""}` : "no image folders found";
  }
}

function rankingChanged() {
  const d = train.options.ranknet_datasets.find((x) => x.id === $("t-dataset").value);
  const r = d && d.rankings.find((x) => x.file === $("t-ranking").value);
  $("t-ranking-info").textContent = r ? rankingSummary(r) : "";
}

// "240 pairs (180 with a preference), self-consistency 85%: 'which one ...?'" (ranking.js uses it too)
function rankingSummary(r) {
  if (r.error) return r.error;
  const c = r.consistency;
  return `${r.pairs} pairs (${r.strict} with a preference)`
    + (r.dropped ? `, ${r.dropped} with images not in this dataset` : "")
    + (c ? `, self-consistency ${(c.rate * 100).toFixed(0)}% over ${c.total} repeats` : "")
    + (r.question ? `. "${r.question}"` : "");
}

// Does a model with this name exist already? Warn before it gets overwritten.
let nameTimer = null;
function checkName() {
  clearTimeout(nameTimer);
  nameTimer = setTimeout(async () => {
    const name = $("t-name").value.trim();
    train.nameInfo = null;
    if (!name) {
      $("t-name-info").textContent = "";
      updateFormState();
      return;
    }
    try {
      train.nameInfo = await api(`/api/train/model?${query({ name })}`);
    } catch (e) {
      $("t-name-info").textContent = e.message.replace(/^.*?: \d+ /, "");
      $("t-name-info").classList.add("warn");
      return;
    }
    updateFormState();
  }, 250);
}

function updateFormState() {
  const info = train.nameInfo;
  const keepGoing = $("t-continue").checked;
  const el = $("t-name-info");
  el.classList.remove("warn");
  let lockArchitecture = false;
  if (!info) {
    el.textContent = $("t-name").value.trim() ? "" : "the model is saved in a folder with this name";
  } else if (!info.exists) {
    el.textContent = "new model";
  } else {
    const epochs = info.epochs ? `, ${info.epochs} epochs trained` : "";
    if (info.kind !== train.kind) {
      el.textContent = `exists: ${info.summary}${epochs}. It is not a ${train.kind}: training would overwrite it.`;
      el.classList.add("warn");
    } else if (keepGoing) {
      el.textContent = `exists: ${info.summary}${epochs}. Training continues from it, with its own architecture.`;
      lockArchitecture = true;
    } else {
      el.textContent = `exists: ${info.summary}${epochs}. Without 'continue' it will be overwritten.`;
      el.classList.add("warn");
    }
  }
  // continuing: the saved model decides the architecture
  $("t-ae-fields").disabled = lockArchitecture;
  $("t-clf-fields").disabled = lockArchitecture;
  $("t-rn-fields").disabled = lockArchitecture;
  const mlp = $("t-head").value === "mlp";
  $("t-hidden-row").hidden = !mlp;
  $("t-dropout-row").hidden = !mlp;
  $("t-bblr-row").hidden = $("t-freeze").checked;
  $("t-clf-hint").textContent = $("t-freeze").checked
    ? "Only the head learns. Without augmentation the backbone's features are computed once: very fast."
    : "The whole network learns (fine-tuning), the backbone with a smaller learning rate. Slow, needs more memory.";

  const augment = $("t-augment").checked;
  $("t-aug-fields").disabled = !augment;
  $("t-aug-hint").textContent = !augment
    ? "Off: every epoch the model sees exactly the same images."
    : train.kind === "autoencoder"
      ? "A different random version of every training image each epoch; the autoencoder has to reconstruct "
        + "that version, so it can't memorize exact pixels. Evaluation images are never changed."
      : "A different random version of every training image each epoch, so the classifier can't memorize exact "
        + "pixels. Evaluation images are never changed. With a frozen backbone, augmentation means running the "
        + "backbone every epoch (slower).";
}

$("t-name").addEventListener("input", checkName);
$("t-continue").addEventListener("change", updateFormState);
$("t-dataset").addEventListener("change", datasetChanged);
$("t-ranking").addEventListener("change", rankingChanged);
$("t-ae-type").addEventListener("change", aeTypeChanged);
$("t-head").addEventListener("change", updateFormState);
$("t-freeze").addEventListener("change", () => { suggestLearningRate(); updateFormState(); });
$("t-augment").addEventListener("change", updateFormState);

function trainSettings() {
  const backbone = train.options.backbones.find((b) => b.id === $("t-backbone").value);
  return {
    kind: train.kind,
    name: $("t-name").value.trim(),
    continue: $("t-continue").checked,
    dataset: $("t-dataset").value,
    epochs: Number($("t-epochs").value),
    learning_rate: Number($("t-lr").value),
    batch_size: Number($("t-batch").value),
    eval_percent: Number($("t-eval").value),
    save_every: Number($("t-save-every").value),
    augment: {
      enabled: $("t-augment").checked,
      flip: $("t-flip").checked,
      rotation: Number($("t-rotation").value),
      jitter: Number($("t-jitter").value),
      crop: Number($("t-crop").value),
    },
    autoencoder: {
      type: $("t-ae-type").value,
      image_size: Number($("t-image-size").value),
      latent_dim: Number($("t-latent").value),
      options: aeOptions(),
    },
    ranking: $("t-ranking").value,
    ranknet: {
      backbones: [...$("t-rn-backbones").querySelectorAll("input:checked")].map((input) => input.value),
      hidden_dim: Number($("t-rn-hidden").value),
      dropout: Number($("t-rn-dropout").value),
      weight_decay: Number($("t-rn-wd").value),
    },
    classifier: {
      backbone: backbone && backbone.value,
      head: $("t-head").value,
      hidden_size: Number($("t-hidden").value),
      dropout: Number($("t-dropout").value),
      freeze_backbone: $("t-freeze").checked,
      backbone_lr_factor: Number($("t-bblr").value),
    },
  };
}

async function startTraining(overwrite = false) {
  const s = { ...trainSettings(), overwrite };
  if (!s.name) {
    $("t-form-status").textContent = "give the model a name";
    $("t-name").focus();
    return;
  }
  $("t-form-status").textContent = "starting…";
  const response = await fetch("/api/train/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(s),
  });
  if (response.status === 409) {
    const body = await response.text();
    let data = {};
    try { data = JSON.parse(body); } catch { /* plain text error */ }
    if (data.exists) {
      if (confirm(`A model named "${s.name}" exists already and 'continue' is off.\n\nOverwrite it?`)) {
        return startTraining(true);
      }
      $("t-form-status").textContent = "not started";
      return;
    }
    $("t-form-status").textContent = body;
    return;
  }
  if (!response.ok) {
    $("t-form-status").textContent = await response.text();
    return;
  }
  $("t-form-status").textContent = "";
  train.previewVersion = -1;
  pollSoon(0);
}

$("t-start").addEventListener("click", () => startTraining());

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

const ACTIVE = ["starting", "running", "stopping"];

// Poll fast while the tab is open, slowly while a run is going on in the
// background (to notice when it ends), not at all otherwise.
function pollSoon(delay) {
  clearTimeout(train.timer);
  train.timer = setTimeout(poll, delay);
}

async function poll() {
  const visible = !$("view-train").hidden;
  let status = null;
  try {
    status = await api(`/api/train/status?${query({ preview: train.previewVersion })}`);
    if (visible) renderStatus(status);
    await noticeSaved(status);
  } catch (e) {
    if (visible) $("t-form-status").textContent = e.message;
  }
  const active = status && ACTIVE.includes(status.state);
  if (visible) pollSoon(1000);
  else if (active) pollSoon(4000);
}

// When a run saves or ends, the Embedding Map's model list may have a new model
let lastSavedState = null;
async function noticeSaved(status) {
  const key = `${status.name}|${status.state}|${status.saved}|${status.history ? status.history.length : 0}`;
  if (status.saved && key !== lastSavedState && lastSavedState !== null && !ACTIVE.includes(status.state)) {
    try { applyOptions(await api("/api/options"), settings()); } catch { /* the map will catch up on rescan */ }
  }
  lastSavedState = key;
}

const formatSeconds = (s) => {
  if (s === null || s === undefined) return "";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
};

function renderStatus(st) {
  $("m-empty").hidden = st.state !== "idle";
  $("m-body").hidden = st.state === "idle";
  const active = ACTIVE.includes(st.state);
  $("t-start").disabled = active;
  if (st.state === "idle") return;

  $("m-title").textContent = `${st.name}  ·  ${st.kind}`;
  $("m-description").textContent = st.description;
  $("m-state").textContent = st.state;
  $("m-state").className = `badge ${st.state}`;
  $("m-stop-save").disabled = !active || st.state === "stopping";
  $("m-stop").disabled = !active || st.state === "stopping";
  $("m-message").hidden = !st.message;
  $("m-message").textContent = st.message;

  $("m-epoch-label").textContent = `epoch ${st.epoch} / ${st.total_epochs}`
    + (st.history.length > st.epoch ? `  (${st.history.length} in total)` : "");
  $("m-time").textContent = `${formatSeconds(st.elapsed)}${st.eta ? `, about ${formatSeconds(st.eta)} left` : ""}`;
  const epochsDone = active ? Math.max(0, st.epoch - 1) + (st.batches ? st.batch / st.batches : 0) : st.epoch;
  $("m-epoch-fill").style.width = `${(100 * epochsDone / Math.max(1, st.total_epochs)).toFixed(1)}%`;
  $("m-stage").textContent = active ? st.stage : (st.saved ? "model saved" : "not saved");
  $("m-batch-label").textContent = st.batches ? `batch ${st.batch} / ${st.batches}` : "";
  $("m-batch-fill").style.width = `${st.batches ? (100 * st.batch / st.batches).toFixed(1) : 0}%`;

  renderCharts(st);
  if (st.preview) {
    train.previewVersion = st.preview_version;
    renderPreview(st.preview);
  }
  const log = $("m-log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = st.log.join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
}

$("m-stop-save").addEventListener("click", () => postJSON("/api/train/stop", { save: true }).then(() => pollSoon(0)));
$("m-stop").addEventListener("click", () => {
  if (confirm("Stop without saving? Whatever was saved before (every 'save every' epochs) stays.")) {
    postJSON("/api/train/stop", { save: false }).then(() => pollSoon(0));
  }
});

// history entries -> {train: {...}, eval: {...}} for every kind (classifiers
// are saved in the format of 05_b_classifiers: train_loss, eval_accuracy...,
// RankNets too, accuracy = share of the pairs the scores order as answered)
const SCORED_KINDS = ["classifier", "ranknet"];
function normalizedHistory(st) {
  return st.history.map((h) => (SCORED_KINDS.includes(st.kind)
    ? { train: { loss: h.train_loss, accuracy: h.train_accuracy },
        eval: h.eval_loss === null ? null : { loss: h.eval_loss, accuracy: h.eval_accuracy } }
    : h));
}

function renderCharts(st) {
  const history = normalizedHistory(st);
  const series = (key) => [
    { name: "train", color: SERIES[0], values: history.map((h) => h.train[key]) },
    { name: "evaluation", color: SERIES[1], values: history.map((h) => (h.eval ? h.eval[key] : null)) },
  ].filter((s) => s.values.some((v) => v !== null && v !== undefined));
  lineChart($("m-loss"), series("loss"), { log: true });
  $("m-acc-figure").hidden = !SCORED_KINDS.includes(st.kind);
  $("m-acc-caption").textContent = st.kind === "ranknet" ? "agreement with the answers (pairs with a preference)" : "accuracy";
  if (SCORED_KINDS.includes(st.kind)) lineChart($("m-acc"), series("accuracy"), { min: 0, max: 1, percent: true });

  // every part of the loss of the last epoch (autoencoders: reconstruction, kl...)
  const last = history[history.length - 1];
  $("m-parts-figure").hidden = !last;
  if (last) {
    const keys = Object.keys(last.train);
    const fmt = (v) => (v === undefined || v === null ? "" : v.toFixed(4));
    $("m-parts").innerHTML = `<tr><th></th><th>train</th>${last.eval ? "<th>evaluation</th>" : ""}</tr>`
      + keys.map((k) => `<tr><td>${k}</td><td>${fmt(last.train[k])}</td>${last.eval ? `<td>${fmt(last.eval[k])}</td>` : ""}</tr>`).join("");
  }
}

// A line chart in an svg with a 600 x 230 viewBox: one line per series, x =
// epoch, a legend, and a crosshair with the values under the mouse.
function lineChart(svg, series, { log = false, min = null, max = null, percent = false } = {}) {
  const W = 600, H = 230, L = 62, R = 14, T = 26, B = 30;
  const n = Math.max(0, ...series.map((s) => s.values.length));
  const values = series.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined && (!log || v > 0));
  svg.innerHTML = "";
  const el = (tag, attrs, text) => {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    svg.appendChild(e);
    return e;
  };
  if (!n || !values.length) {
    el("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, "waiting for the first epoch…");
    return;
  }
  const f = log ? Math.log10 : (v) => v;
  let lo = min ?? Math.min(...values), hi = max ?? Math.max(...values);
  if (lo === hi) { lo -= log ? lo * 0.5 : 0.5; hi += log ? hi * 0.5 : 0.5; }
  const [flo, fhi] = [f(lo), f(hi)];
  const x = (i) => L + (n === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (n - 1));
  const y = (v) => T + (H - T - B) * (1 - (f(v) - flo) / (fhi - flo));
  const label = (v) => (percent ? `${(v * 100).toFixed(0)}%` : v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v.toPrecision(2));

  // horizontal grid: 5 values evenly spaced (in log space for a log chart)
  for (let k = 0; k <= 4; k++) {
    const v = log ? 10 ** (flo + (k / 4) * (fhi - flo)) : lo + (k / 4) * (hi - lo);
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid" });
    el("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, label(v));
  }
  el("line", { x1: L, x2: L, y1: T, y2: H - B, class: "axis" });
  el("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" });
  // epoch ticks: about 6 whole numbers
  const step = Math.max(1, Math.ceil(n / 6));
  for (let i = 0; i < n; i += step) el("text", { x: x(i), y: H - 10, "text-anchor": "middle" }, i + 1);
  el("text", { x: W - R, y: H - 10, "text-anchor": "end" }, "epoch");

  series.forEach((s, k) => {
    const points = s.values.map((v, i) => (v === null || v === undefined || (log && v <= 0) ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`))
      .filter(Boolean).join(" ");
    el("polyline", { points, class: "series", stroke: s.color });
    // legend, top right
    const lx = W - R - 110 * (series.length - k);
    el("line", { x1: lx, x2: lx + 18, y1: 12, y2: 12, stroke: s.color, "stroke-width": 3 });
    el("text", { x: lx + 24, y: 16 }, s.name);
  });

  const cross = el("line", { x1: 0, x2: 0, y1: T, y2: H - B, class: "crosshair", visibility: "hidden" });
  const readout = el("text", { x: L + 4, y: 16, visibility: "hidden" });
  const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" });
  hit.addEventListener("mousemove", (event) => {
    const r = svg.getBoundingClientRect();
    const px = ((event.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - L) / (W - L - R)) * (n - 1))));
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    readout.textContent = `epoch ${i + 1}: ` + series.map((s) => {
      const v = s.values[i];
      return `${s.name} ${v === null || v === undefined ? "–" : percent ? `${(v * 100).toFixed(1)}%` : v.toPrecision(4)}`;
    }).join("  ");
    readout.setAttribute("visibility", "visible");
  });
  hit.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden");
    readout.setAttribute("visibility", "hidden");
  });
}

function renderPreview(p) {
  const box = $("m-preview");
  box.className = p.kind;
  box.innerHTML = "";
  $("m-confusion-block").hidden = !(p.kind === "classifier" && p.confusion);
  $("m-misclassified-block").hidden = !(p.kind === "classifier" && p.misclassified && p.misclassified.length);

  if (p.kind === "ranknet") {
    renderRankNetPreview(p, box);
    return;
  }

  if (p.kind === "autoencoder") {
    $("m-preview-title").textContent = "Preview  (input · reconstruction · latent: blue −, red +)";
    const scale = Math.max(1e-6, ...p.rows.flatMap((r) => r.latent.map(Math.abs)));
    for (const row of p.rows) {
      const div = document.createElement("div");
      div.className = "ae-row";
      div.innerHTML = `<img src="${row.input}" alt="input"><img src="${row.output}" alt="reconstruction"><canvas width="${row.latent.length}" height="1"></canvas>`;
      const ctx = div.querySelector("canvas").getContext("2d");
      row.latent.forEach((v, d) => {
        ctx.fillStyle = diverging(v / scale);
        ctx.fillRect(d, 0, 1, 1);
      });
      box.appendChild(div);
    }
    return;
  }

  const augmented = p.samples.some((s) => s.augmented && s.augmented.length);
  $("m-preview-title").textContent = "Preview  (softmax scores, true class in green"
    + (augmented ? "; below each image: augmented versions, as the network receives them while training" : "") + ")";
  for (const s of p.samples) {
    const best = s.scores.indexOf(Math.max(...s.scores));
    const card = document.createElement("div");
    card.className = "clf-card";
    card.innerHTML = `<img src="${s.image}" alt="">`
      + (s.augmented && s.augmented.length
        ? `<div class="augmented">${s.augmented.map((url) => `<img src="${url}" alt="augmented">`).join("")}</div>` : "")
      + `<div class="true">${best === s.label ? "✓" : "✗"} ${escapeHTML(p.class_names[s.label])}</div>`
      + s.scores.map((score, c) => `
        <div class="bar-row${c === s.label ? " true-class" : ""}">
          <div class="label"><span>${escapeHTML(p.class_names[c])}</span><span>${score.toFixed(2)}</span></div>
          <div class="track"><div class="fill" style="width:${(score * 100).toFixed(1)}%"></div></div>
        </div>`).join("");
    box.appendChild(card);
  }
  if (p.confusion) renderConfusion(p.class_names, p.confusion);
  if (p.misclassified) renderMisclassified(p.class_names, p.misclassified);
}

// The best and worst images by the current scores, and the evaluation pairs
// the model disagrees with most (the answer outlined)
function renderRankNetPreview(p, box) {
  const c = p.consistency;
  $("m-preview-title").textContent = `Preview  (${p.pairs} pairs on ${p.images} images`
    + (c ? `; you agreed with yourself ${(c.rate * 100).toFixed(0)}% of the time over ${c.total} repeated pairs: `
      + "about the best agreement a model can reach" : "") + ")";
  const strip = (title, items) => `<h2>${title}</h2><div class="rank-strip">${items.map((m) => `
    <figure><img src="${m.image}" alt=""><figcaption>${m.score.toFixed(2)}  ${escapeHTML(m.name)}</figcaption></figure>`).join("")}</div>`;
  box.innerHTML = strip("Best", p.best) + strip("Worst", p.worst)
    + `<h2>Most disagreed <span class="note">(evaluation pairs, outlined: the one that was chosen)</span></h2>`
    + (p.disagreements.length ? `<div class="rank-pairs">${p.disagreements.map((d) => `
      <div class="rank-pair">
        <div class="images"><img src="${d.a}" alt=""${d.answer > 0 ? ' class="preferred"' : ""}><img src="${d.b}" alt=""${d.answer < 0 ? ' class="preferred"' : ""}></div>
        <div class="note">the model gives the chosen one ${(d.model * 100).toFixed(0)}%</div>
      </div>`).join("")}</div>`
      : `<p class="hint">${p.disagreements ? "none: the model orders every evaluation pair as answered" : ""}</p>`);
}

// one row per true class: how many are wrong, and the worst of them
function renderMisclassified(names, rows) {
  $("m-misclassified").innerHTML = rows.map((row, c) => `
    <div class="mis-row">
      <div class="mis-class"><strong>${escapeHTML(names[c])}</strong><br>
        <span class="note">${row.wrong} / ${row.total} wrong</span></div>
      <div class="mis-images">${row.images.length ? row.images.map((m) => `
        <figure>
          <img src="${m.image}" alt="">
          <figcaption>as ${escapeHTML(names[m.predicted])} ${m.score.toFixed(2)}<br>true class ${m.true_score.toFixed(2)}</figcaption>
        </figure>`).join("") : `<span class="note">${row.total ? "none wrong" : "no evaluation images"}</span>`}
      </div>
    </div>`).join("");
}

// rows: true class, columns: predicted. Cells shaded by the share of their row.
function renderConfusion(names, matrix) {
  const head = `<tr><th></th>${names.map((n) => `<th>${escapeHTML(n)}</th>`).join("")}</tr>`;
  const rows = matrix.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return `<tr><th>${escapeHTML(names[i])}</th>${row.map((count, j) => {
      const share = count / total;
      const [r, g, b] = SCORE.map((c) => Math.round(255 + share * (c - 255)));     // white -> dark blue
      const ink = share > 0.5 ? "#ffffff" : "#000000";
      return `<td class="${i === j ? "diagonal" : ""}" style="background:rgb(${r},${g},${b});color:${ink}">${count}</td>`;
    }).join("")}</tr>`;
  }).join("");
  $("m-confusion").innerHTML = `<table class="confusion">${head}${rows}</table>`;
}

// a run may already be going on (started before the page was opened)
pollSoon(500);
