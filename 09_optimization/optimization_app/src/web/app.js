// CLIP optimization front end: config form <-> /api/start, polls /api/state, draws the target map.

const $ = (sel) => document.querySelector(sel);
const STORAGE_KEY = "clip-optimization-config";
const TAB_KEY = "clip-optimization-tab";
const EMBED_KEY = "clip-optimization-embed";
const POLL_MS = 200;
const COLORS = { plus: "#000", minus: "#c0392b", views: "#2c6fbb", axis: "#bbb", label: "#000" };

let options = null;
let runId = "";
let traces = { plus: [], minus: [] };
let termTraces = { plus: [], minus: [] };  // per step, the values of lastState.terms
let lastState = null;
let frameVersion = -1;
let matchTargets = {};  // match term -> override image path (absent = the start image)
let patchImages = [];  // PIXEL_PATCHES image sources: [{path, count}]

// ── Config form ────────────────────────────────────────────────────────────────

function fieldInputs() {
  return document.querySelectorAll("#sidebar [data-key], #inspector [data-key]");
}

function writeForm(cfg) {
  // Keys missing from an older saved config fall back to the script defaults.
  cfg = { ...(options?.defaults || {}), ...migrateConfig(migrateShapeMode(cfg)) };
  for (const el of fieldInputs()) {
    const key = el.dataset.key;
    let value = cfg[key];
    if (el.dataset.idx !== undefined) value = value?.[+el.dataset.idx];
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value ?? "";
  }
  const stop = cfg.stop_at_similarity;
  $("#budget-on").checked = stop !== null && stop !== undefined && stop !== "";
  if ($("#budget-on").checked) $("#budget-value").value = stop;
  writeSurface(cfg.surface);
  writeLevelset(cfg.levelset);
  writeFunctionStack(cfg.function_stack);
  writeRevolution(cfg.revolution);
  writeMeshTexture(cfg.mesh_texture);
  writeGaussians3D(cfg.gaussians_3d);
  writeVoxels(cfg.voxels);
  writeSolids(cfg);
  // a config saved when the templates were global: they go to each CLIP model that has none
  const legacyTemplates = { use_templates: cfg.use_text_templates ?? true, templates: cfg.text_templates };
  models = (cfg.models || []).map((m) => ({ ...newModel(m.family, m.model),
    ...(m.family === "clip" && !m.templates && legacyTemplates.templates ? legacyTemplates : {}),
    ...m, family: entryFamily(m) }));
  selected = models.length ? 0 : -1;
  renderModels();
  matchTargets = { ...(cfg.match_targets || {}) };
  refreshMatchTargets();
  patchImages = (cfg.patch_images || []).map((e) => ({ ...e }));
  renderPatchImages();
  syncBackgroundImage();
  updateFamilyVisibility();
  syncDistortionControls();
  syncPatchControls();
  syncShapeControls();
  syncConvControls();
  syncSequentialControls();
  syncPatternSelect();
  updateFieldVisibility();
  updateStartThumb();
  updatePatternThumb();
}

function readForm() {
  const cfg = {};
  for (const el of fieldInputs()) {
    const key = el.dataset.key;
    let value;
    if (el.type === "checkbox") value = el.checked;
    else if (el.type === "number") value = el.value === "" ? null : Number(el.value);
    else value = el.value;
    if (el.dataset.idx !== undefined) {
      cfg[key] = cfg[key] || [];
      cfg[key][+el.dataset.idx] = value;
    } else {
      cfg[key] = value;
    }
  }
  cfg.stop_at_similarity = $("#budget-on").checked ? Number($("#budget-value").value) : null;
  commitSheet();
  cfg.models = models.map((m) => ({ ...m }));
  cfg.match_targets = { ...matchTargets };
  cfg.patch_images = patchImages.map((e) => ({ ...e }));
  cfg.surface = structuredClone(surface);
  cfg.levelset = structuredClone(levelset);
  cfg.function_stack = structuredClone(functionStack);
  cfg.revolution = structuredClone(revolution);
  cfg.mesh_texture = structuredClone(meshTexture);
  cfg.gaussians_3d = structuredClone(gaussians3d);
  cfg.voxels = structuredClone(voxels);
  cfg.assembly = structuredClone(assembly);
  cfg.solids_on_off = structuredClone(onOff);
  cfg.extrusions = structuredClone(extrusions);
  return cfg;
}

function saveLocal() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(readForm())); } catch { /* storage unavailable */ }
}

function loadLocal() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch { return null; }
}

// The sheet's template list shows only while it is in use (it is kept either way).
function syncTemplatesEnabled() {
  $("#sheet-templates-label").hidden = !$("#sheet-use-templates").checked;
}

// ── Models ─────────────────────────────────────────────────────────────────────
// The config's `models`: every model the image is optimized for, all at once. An entry is
// {family, model (a ModelID string; a custom model's id), targets, directions, and the
// options of its kind}. The list shows them; the property sheet below edits models[selected].

let models = [];
let selected = -1;
let sheetFor = -1;  // the entry the sheet's inputs currently show (-1: none)

function newModel(family, model) {
  const vlm = family === "vlm";
  return { family, model, targets: [], directions: [], score: vlm ? "logprob" : "logits", tokens: "cls",
           prompt: "an image", goal: "maximize", value: 0, weight: 1, precision: vlm ? "bf16" : "fp32",
           use_templates: true, templates: [...(options?.text_templates || [])] };
}

// An entry's family; an aesthetic score saved under "custom" (before it had its own family) -> "aesthetic"
function entryFamily(entry) {
  return entry.family === "custom" && String(entry.model).startsWith("aesthetic/") ? "aesthetic" : entry.family;
}

// A config saved before the model list (one model_family, with its own keys) -> {models: [that model]}
function migrateConfig(cfg) {
  if (!cfg || Array.isArray(cfg.models) || !cfg.model_family) return cfg;
  const custom = options.custom_models.find((m) => m.id === cfg.custom_model);
  const classifier = custom?.kind === "classifier";
  const entry = {
    clip: { model: cfg.model, targets: cfg.text_targets, directions: cfg.direction_targets },
    dino: { model: cfg.dino_model, tokens: cfg.dino_tokens, targets: cfg.dino_targets, directions: cfg.dino_direction_targets },
    imagenet: { model: cfg.imagenet_model, score: cfg.imagenet_score, targets: cfg.class_targets,
                directions: cfg.class_direction_targets },
    custom: custom && { model: custom.id, prompt: cfg.custom_prompt, precision: cfg.custom_precision,
                        score: cfg.custom_class_score, goal: cfg.score_goal, value: cfg.score_value, weight: cfg.score_weight,
                        targets: classifier ? cfg.custom_class_targets : cfg.custom_targets,
                        directions: classifier ? cfg.custom_class_direction_targets : cfg.custom_direction_targets },
  }[cfg.model_family];
  const given = entry ? Object.fromEntries(Object.entries(entry).filter(([, v]) => v !== undefined && v !== null)) : {};
  return { ...cfg, models: given.model ? [{ ...newModel(cfg.model_family, given.model), ...given }] : [] };
}

// A config saved with the Monochrome toggle -> shape_mode. That mix ran from the background to the
// fg colour, so those are its two colours now.
function migrateShapeMode(cfg) {
  if (!cfg || cfg.shape_mode || cfg.shape_monochrome === undefined) return cfg;
  const { shape_monochrome: mono, shape_fg_color: fg, shape_fg_optimize: optimize, ...rest } = cfg;
  return { ...rest, shape_mode: mono ? "monochrome" : "color",
           ...(mono && fg ? { shape_color_1: fg, shape_color_1_optimize: !!optimize } : {}),
           ...(mono && cfg.background_color ? { shape_color_2: cfg.background_color } : {}) };
}

// What an entry is: kind (its family, or a custom model's kind), the mode of its target ends
// ("text" | "class" | "image" | "query": a VLM's question and answer; null = a score, no
// targets), class names, label, sheet tags.
function modelInfo(entry) {
  const isCustom = entry.family === "custom" || entry.family === "aesthetic";
  const custom = isCustom ? options.custom_models.find((m) => m.id === entry.model) : null;
  const kind = isCustom ? (custom?.kind || "missing") : entry.family;
  const mode = { clip: "text", imagenet: "class", classifier: "class", dino: "image", autoencoder: "image",
                 vlm: "query" }[kind] || null;
  const classes = kind === "imagenet" ? options.imagenet_classes : kind === "classifier" ? custom.classes : null;
  const [, arch, pretrained] = String(entry.model).split("~");
  const label = { clip: `CLIP ${arch} · ${pretrained}`, dino: `DINOv2 ${arch}`, imagenet: `ImageNet ${arch}`,
                  vlm: `VLM ${String(arch).split("/").pop()}` }[entry.family]
    || (custom ? custom.label : `missing: ${entry.model}`);
  return { kind, mode, classes, label, tags: [entry.family, kind, ...(custom?.prompt ? ["prompt"] : [])] };
}

function modelSummary(entry, info) {
  if (info.kind === "missing") return "not in the models folder";
  if (!info.mode) {
    const goal = entry.goal === "value" ? `to ${entry.value}` : entry.goal;
    return `${goal}, weight ${entry.weight}` + (info.tags.includes("prompt") ? ` · "${entry.prompt}"` : "");
  }
  const n = entry.targets.length, d = entry.directions.length;
  return [n && `${n} target${n > 1 ? "s" : ""}`, d && `${d} Δ`].filter(Boolean).join(" · ") || "no targets yet";
}

function renderModelList() {
  const list = $("#model-list");
  if (!models.length) {
    const empty = document.createElement("div");
    empty.className = "muted small";
    empty.textContent = "no models: only the image matching terms (Targets tab)";
    list.replaceChildren(empty);
    return;
  }
  list.replaceChildren(...models.map((entry, i) => {
    const info = modelInfo(entry);
    const row = document.createElement("div");
    row.className = "model-item" + (i === selected ? " selected" : "");
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "model-pick";
    pick.title = i === 0 ? "the first model also measures the edit size (sim → start)" : "click to edit";
    const name = document.createElement("span");
    name.textContent = info.label;
    const summary = document.createElement("span");
    summary.className = "small";
    summary.textContent = (i === 0 && models.length > 1 ? "edit size · " : "") + modelSummary(entry, info);
    pick.append(name, summary);
    pick.onclick = () => selectModel(i);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "remove";
    remove.className = "remove";
    remove.onclick = () => removeModel(i);
    row.append(pick, remove);
    return row;
  }));
}

const KIND_INFO = {
  clip: "text prompts or images", dino: "no text encoder: targets are images",
  imagenet: "type to filter the 1000 ImageNet classes", autoencoder: "autoencoder: image targets in its latent space",
  vlm: "a question and an answer: maximizes log p(answer | image, question). Δ: log p(to) - log p(from), e.g. No → Yes",
  ranknet: "RankNet: one score per image", aesthetic: "aesthetic score: one number per image",
  missing: "this model is not in the models folder (env_config.json): remove it or restore the model",
};

function renderSheet() {
  const sheet = $("#model-sheet");
  sheetFor = selected;
  sheet.hidden = selected < 0;
  if (selected < 0) return;
  const entry = models[selected];
  const info = modelInfo(entry);
  $("#sheet-title").textContent = info.label;
  $("#sheet-info").textContent = (info.kind === "classifier"
    ? `classifier, ${info.classes.length} classes: ${info.classes.join(", ")}` : KIND_INFO[info.kind])
    + (selected === 0 && models.length > 1 ? " · also measures the edit size (sim → start)" : "");
  for (const el of sheet.querySelectorAll(".sheet-only")) {
    el.hidden = !el.dataset.for.split(" ").some((tag) => info.tags.includes(tag));
  }
  fillSheetModels(entry);
  // the score choices of the kind: a classifier's logits / softmax, a VLM's log-probability
  const vlm = info.kind === "vlm";
  $("#sheet-score").replaceChildren(...(vlm ? options.vlm_scores : options.imagenet_scores)
    .map((s) => new Option({ logprob: "answer log-probability", mean: "per-token log-probability" }[s] || s, s)));
  $("#sheet-score-label").firstChild.textContent = vlm ? "Score " : "Maximize ";
  $("#sheet-score-label").title = vlm
    ? "answer log-probability: log p(answer), summed over its tokens. per token: their mean -- long answers are not penalized for their length"
    : "logits: raw class scores (unbounded). softmax: class probabilities (0..1, saturating -- the gradient vanishes once a class wins or is hopeless)";
  for (const el of sheet.querySelectorAll("[data-prop]")) {
    const value = entry[el.dataset.prop];
    if (el.type === "checkbox") el.checked = !!value;
    else if (el.dataset.lines !== undefined) el.value = (value || []).join("\n");
    else el.value = value ?? "";
  }
  syncTemplatesEnabled();
  $("#sheet-value-label").hidden = entry.goal !== "value";
  const word = { class: "class", query: "question" }[info.mode] || "text";
  for (const b of sheet.querySelectorAll("button[data-word]")) b.textContent = `${b.dataset.add === "direction" ? "Δ" : "+"} ${word}`;
  $("#sheet-hint").textContent = info.mode === "text" ? "Paraphrases: separate with |"
    : info.mode === "query" ? "A direction's second question: blank = the same question" : "";
  $("#sheet-classes").replaceChildren(...(info.classes || []).map((_, i) => new Option(classLabel(i, info.classes))));
  $("#sheet-targets").replaceChildren(...entry.targets.map((values) => targetRow("target", values, info)),
                                      ...entry.directions.map((values) => targetRow("direction", values, info)));
}

// The sheet's model selector: the other models of the entry's family (CLIP / DINOv2 / ImageNet:
// their targets are interchangeable), as in the add-model selector. Always holds the entry's own
// model, so committing the sheet never changes a custom entry or one no longer on offer.
function fillSheetModels(entry) {
  const source = { clip: "#model", dino: "#dino-model", imagenet: "#imagenet-model", vlm: "#vlm-model" }[entry.family];
  const opts = source ? [...$(source).options].map((o) => new Option(o.text, o.value)) : [];
  if (!opts.some((o) => o.value === entry.model)) opts.unshift(new Option(entry.model, entry.model));
  $("#sheet-model").replaceChildren(...opts);
}

// The sheet's inputs -> models[selected]
function commitSheet() {
  if (sheetFor < 0 || sheetFor !== selected || !models[selected]) return;
  const entry = models[selected];
  const info = modelInfo(entry);
  for (const el of $("#model-sheet").querySelectorAll("[data-prop]")) {
    entry[el.dataset.prop] = el.type === "checkbox" ? el.checked
      : el.dataset.lines !== undefined ? el.value.split("\n").map((s) => s.trim()).filter(Boolean)
      : el.type === "number" ? (el.value === "" ? null : Number(el.value)) : el.value;
  }
  entry.targets = readTargets("target", info);
  entry.directions = readTargets("direction", info);
}

function renderModels() {
  renderModelList();
  renderSheet();
}

function selectModel(i) {
  commitSheet();
  selected = i;
  renderModels();
}

function removeModel(i) {
  commitSheet();
  models.splice(i, 1);
  if (i < selected || selected >= models.length) selected -= 1;
  if (selected < 0 && models.length) selected = 0;
  sheetFor = -1;  // its inputs show the removed (or a shifted) entry
  renderModels();
  saveLocal();
}

function addModel() {
  const family = $("#model-family").value;
  const model = $({ clip: "#model", dino: "#dino-model", imagenet: "#imagenet-model", vlm: "#vlm-model",
                     aesthetic: "#aesthetic-model", custom: "#custom-model" }[family]).value;
  if (!model) { setStatus(`no ${family} model to add`, true); return; }
  commitSheet();
  models.push(newModel(family, model));
  selected = models.length - 1;
  renderModels();
  saveLocal();
}

// ── Target rows ────────────────────────────────────────────────────────────────
// A target row ("target") or a direction row ("direction", source -> target) of the
// selected model. Its ends are text prompts (CLIP), classes (ImageNet / custom classifiers:
// indices, picked by name from #sheet-classes), images ({image: path}: a thumbnail that
// uploads a new image on click, whose embedding is the target) -- any of those modes may use
// images -- or a VLM's queries ({question, answer}: two inputs; no images).

// ImageNet names repeat (two "crane"s, two "maillot"s), so the index rides along in the label.
function classLabel(index, names = options.imagenet_classes) {
  return `${names[index]} (${index})`;
}

function classIndex(text, names = options.imagenet_classes) {
  const m = text.match(/\((\d+)\)\s*$/);
  if (m && +m[1] < names.length) return +m[1];
  const i = names.findIndex((n) => n.toLowerCase() === text.trim().toLowerCase());
  return i < 0 ? null : i;
}

// Square letterboxed preview of a target image (the whole image visible).
function thumbUrl(path, size = 128) {
  return `/api/preview/image?${new URLSearchParams({ path, size, fit: "letterbox", w: size, h: size })}`;
}

const isImage = (v) => v !== null && typeof v === "object" && "image" in v;

// A VLM end: the question above the answer. A direction's second question may stay blank
// (placeholder: the same question).
function queryEnd(value, sameQuestion) {
  const end = document.createElement("div");
  end.className = "prompt query";
  const [question, answer] = ["question", "answer"].map((key) => {
    const input = document.createElement("input");
    input.type = "text";
    input.className = key;
    input.placeholder = key === "question" && sameQuestion ? "same question" : key;
    input.value = value?.[key] ?? "";
    end.appendChild(input);
    return input;
  });
  if (sameQuestion && value?.question === sameQuestion) question.value = "";
  return end;
}

function imagePicker(path) {
  const pick = document.createElement("button");
  pick.type = "button";
  pick.className = "prompt image-pick";
  pick.title = "click to upload an image";
  const set = (p) => {
    pick.dataset.path = p || "";
    if (p) {
      const img = document.createElement("img");
      img.src = thumbUrl(p);
      img.alt = p.split(/[\\/]/).pop();
      pick.replaceChildren(img);
      pick.title = `${p}\nclick to replace`;
    } else {
      pick.textContent = "+ image";
    }
  };
  set(path);
  pick.onclick = () => {
    const file = document.createElement("input");
    file.type = "file";
    file.accept = "image/*";
    file.onchange = async () => {
      if (!file.files[0]) return;
      try { set(await uploadFile(file.files[0])); saveLocal(); } catch (err) { setStatus(err.message, true); }
    };
    file.click();
  };
  return pick;
}

function targetRow(kind, values, info) {
  const isClass = info.mode === "class";
  const names = info.classes;
  const isDirection = kind === "direction";
  const row = document.createElement("div");
  row.className = `target ${isDirection ? "direction" : "text"}`;
  row.dataset.kind = kind;
  const texts = isDirection ? [values[0], values[1]] : [values[0]];
  const placeholders = isDirection ? ["source", "target"] : [isClass ? "class" : "prompt"];
  // Directions stack vertically: a "from" row, a "to" row, then weight / remove.
  const tag = (text) => {
    if (!isDirection) return;
    const span = document.createElement("span");
    span.className = "tag";
    span.textContent = text;
    row.appendChild(span);
  };
  texts.forEach((text, i) => {
    tag(i === 0 ? "from" : "to");
    if (isImage(text)) { row.appendChild(imagePicker(text.image)); return; }
    if (info.mode === "query") { row.appendChild(queryEnd(text, i === 1 ? texts[0]?.question || " " : null)); return; }
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = placeholders[i];
    input.className = "prompt";
    if (isClass) {
      input.setAttribute("list", "sheet-classes");
      input.value = text === "" || text == null || !(text < names.length) ? "" : classLabel(text, names);
      const check = () => input.classList.toggle("invalid", !!input.value.trim() && classIndex(input.value, names) === null);
      input.addEventListener("input", check);
      // The browser filters the class list by the input's text, so a chosen class would
      // hide all the others: while focused the input is empty (the full list; typing
      // filters it) with the chosen class held as its placeholder, restored on leaving
      // without a new choice.
      input.addEventListener("focus", () => {
        if (!input.value) return;
        input.dataset.held = input.value;
        input.placeholder = input.value;
        input.value = "";
      });
      input.addEventListener("blur", () => {
        if (!input.value && input.dataset.held) input.value = input.dataset.held;
        delete input.dataset.held;
        input.placeholder = placeholders[i];
        check();
      });
      // Normalize an unambiguous typed name to its "name (index)" label.
      input.addEventListener("change", () => {
        const idx = classIndex(input.value, names);
        if (idx !== null) input.value = classLabel(idx, names);
        check();
      });
    } else {
      input.value = text ?? "";
    }
    row.appendChild(input);
  });
  if (isDirection) row.firstChild.textContent = "Δ from";
  tag("weight");
  const weight = document.createElement("input");
  weight.type = "number";
  weight.step = "0.1";
  weight.value = values[values.length - 1] ?? 1;
  weight.className = "weight";
  weight.title = "weight";
  row.appendChild(weight);
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "×";
  remove.title = "remove";
    remove.className = "remove";
  remove.onclick = () => { row.remove(); saveLocal(); };
  row.appendChild(remove);
  row.addEventListener("input", saveLocal);
  return row;
}

// Rows with an empty prompt, an unknown class, no image yet or a query without its question
// or answer are left out.
function readTargets(kind, info) {
  const parse = info.mode === "class" ? (text) => classIndex(text, info.classes) : (text) => text.trim() || null;
  return [...$("#sheet-targets").children].filter((row) => row.dataset.kind === kind).map((row) => {
    let question = null;  // a VLM direction's second question defaults to the first
    const prompts = [...row.querySelectorAll(".prompt")].map((el) => {
      if (el.classList.contains("query")) {
        question = el.querySelector(".question").value.trim() || question;
        const answer = el.querySelector(".answer").value.trim();
        return question && answer ? { question, answer } : null;
      }
      // a focused class input holds its class aside (see targetRow)
      return el.classList.contains("image-pick") ? (el.dataset.path ? { image: el.dataset.path } : null)
        : parse(el.value || el.dataset.held || "");
    });
    return [...prompts, Number(row.querySelector("input.weight").value)];
  }).filter((values) => values.slice(0, -1).every((v) => v !== null));
}

// ── Sidebar tabs ───────────────────────────────────────────────────────────────

function showTab(name) {
  const panels = [...document.querySelectorAll(".tab-panel")];
  if (!panels.some((p) => p.dataset.tab === name)) name = panels[0].dataset.tab;
  for (const p of panels) p.hidden = p.dataset.tab !== name;
  for (const b of document.querySelectorAll(".tabs [data-tab]")) {
    b.setAttribute("aria-selected", String(b.dataset.tab === name));
  }
  try { localStorage.setItem(TAB_KEY, name); } catch { /* storage unavailable */ }
  if (name === "status") $("#status-log").scrollTop = $("#status-log").scrollHeight;
}

// ── Status tab ─────────────────────────────────────────────────────────────────
// The server's warnings and errors since the last run started (state.status_log). The entries
// come only when they changed; the log stays at its end unless it was scrolled up.

let statusLogVersion = -1;

function showStatusLog(log) {
  statusLogVersion = log.version;
  document.querySelector('.tabs [data-tab="status"]').dataset.level = log.level;
  if (!log.entries) return;
  const box = $("#status-log");
  const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 20;
  box.replaceChildren(...log.entries.map((entry) => {
    const row = document.createElement("div");
    row.className = "status-entry";
    row.dataset.level = entry.level;
    const when = document.createElement("span");
    when.className = "when";
    when.textContent = `${entry.time}  ${entry.level}${entry.count > 1 ? `  x${entry.count}` : ""}\n`;
    row.append(when, entry.text);
    return row;
  }));
  if (!log.entries.length) {
    const empty = document.createElement("div");
    empty.className = "muted small";
    empty.textContent = "No warnings or errors since the last run started.";
    box.append(empty);
  }
  if (atEnd) box.scrollTop = box.scrollHeight;
}

// ── Field-specific controls / previews ─────────────────────────────────────────

// The Max field's unit follows the offset mode; the fold weight is live only with Prevent folds on.
function syncDistortionControls() {
  const resolution = $("#distortion-offset-mode").value === "resolution";
  $("#distortion-max-label").firstChild.textContent = resolution ? "Max (× grid step) " : "Max (% of size) ";
  $("#fold-weight").disabled = !$("#prevent-folds").checked;
}

// Sequential mode sets the steps per pole itself (steps / target x targets x cycles).
function syncSequentialControls() {
  const on = $("#sequential-targets").checked;
  $("#sequential-steps").disabled = !on;
  $("#sequential-cycles").disabled = !on;
  document.querySelector('[data-key="epochs"]').disabled = on;
}

// Randomized patch transforms are not optimized, so their LR does not apply.
// Image sources: the pixel-only controls go, the image list comes.
function syncPatchControls() {
  const images = $("#patch-source").value === "images";
  for (const el of document.querySelectorAll(".patch-pixels")) el.hidden = images;
  for (const el of document.querySelectorAll(".patch-images")) el.hidden = !images;
  $("#patch-transform-lr").disabled = !images && $("#patch-randomize").checked;
}

// Each shape mode's controls; masking has one colour, so it is not "1".
function syncShapeControls() {
  const mode = $("#shape-mode").value;
  for (const el of document.querySelectorAll(".shape-mode")) el.hidden = !el.dataset.modes.split(" ").includes(mode);
  $("#shape-color-1-label").firstChild.textContent = mode === "masking" ? "Color " : "Color 1 ";
}

// Opens a file dialog; the uploaded image's path goes to onPath.
function pickImageFile(onPath) {
  const file = document.createElement("input");
  file.type = "file";
  file.accept = "image/*";
  file.onchange = async () => {
    if (!file.files[0]) return;
    try { onPath(await uploadFile(file.files[0])); } catch (err) { setStatus(err.message, true); }
  };
  file.click();
}

// One tile per image source: thumbnail (click to replace), its patch count, x to remove.
function renderPatchImages() {
  const list = $("#patch-image-list");
  list.replaceChildren(...patchImages.map((entry, i) => {
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "match-pick";
    pick.title = `${entry.path}\nclick to replace`;
    const img = document.createElement("img");
    img.src = thumbUrl(entry.path, 112);
    img.alt = "";
    pick.append(img);
    pick.onclick = () => pickImageFile((path) => { entry.path = path; renderPatchImages(); saveLocal(); });
    const count = document.createElement("input");
    count.type = "number";
    count.min = "0";
    count.value = entry.count;
    count.title = "How many patches (transformed copies) of this image";
    count.oninput = () => { entry.count = Math.max(0, Number(count.value) || 0); saveLocal(); };
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "patch-image-remove";
    remove.textContent = "×";
    remove.title = "remove this image";
    remove.onclick = () => { patchImages.splice(i, 1); renderPatchImages(); saveLocal(); };
    const tile = document.createElement("div");
    tile.className = "patch-image";
    tile.append(pick, count, remove);
    return tile;
  }));
}

// The background image control: one for every field with a background, moved next to the active
// field's colour. Its thumbnail (click to upload) shows the image framed on the canvas.
function syncBackgroundImage() {
  const path = $("#bg-image-path").value;
  const pick = $("#bg-image-pick");
  pick.title = path ? `${path}\nclick to replace` : "Background image: click to upload";
  $("#bg-image-clear").hidden = !path;
  $("#bg-image-options").hidden = !path;
  if (!path) { pick.textContent = "+"; return; }
  const [w, h] = [...document.querySelectorAll('[data-key="canvas"]')].map((el) => Number(el.value) || 512);
  const img = document.createElement("img");
  img.alt = "";
  img.src = `/api/preview/image?${new URLSearchParams({ path, size: 112, fit: $("#bg-image-fit").value, w, h })}`;
  pick.replaceChildren(img);
}

function setBackgroundImage(path) {
  $("#bg-image-path").value = path;
  syncBackgroundImage();
  saveLocal();
  schedulePreview();
}

function setupBackgroundImage() {
  $("#bg-image-pick").onclick = () => pickImageFile(setBackgroundImage);
  $("#bg-image-clear").onclick = () => setBackgroundImage("");
  $("#bg-image-fit").addEventListener("change", () => { syncBackgroundImage(); schedulePreview(); });
  $("#bg-image-multiply").addEventListener("change", schedulePreview);
  for (const el of document.querySelectorAll('[data-key="canvas"]')) el.addEventListener("change", syncBackgroundImage);
}

// Inplace convolutions have no input tensor (nor a fit); the generative net's output size.
function syncConvControls() {
  const inplace = $("#conv-inplace").checked;
  for (const el of document.querySelectorAll(".conv-generative")) el.hidden = inplace;
  for (const el of document.querySelectorAll(".conv-inplace")) el.hidden = !inplace;
  const value = (sel) => Number(document.querySelector(sel).value) || 0;
  const layers = value('[data-key="conv_layers"]');
  const scale = 2 ** layers;
  $("#conv-note").textContent = inplace ? "Output: the canvas size (the start image's)."
    : `Output: ${value('[data-key="conv_input_res"][data-idx="0"]') * scale} x ${value('[data-key="conv_input_res"][data-idx="1"]') * scale}, resized to the canvas.`;
}

// The model selector of the family chosen to add
function updateFamilyVisibility() {
  const family = $("#model-family").value;
  for (const el of document.querySelectorAll(".family")) el.hidden = el.dataset.family !== family;
}

function updateFieldVisibility() {
  const type = $("#field-type").value;
  for (const el of document.querySelectorAll(".only")) {
    el.hidden = !el.dataset.fields.split(" ").includes(type);
  }
  document.querySelector(`.bg-image-slot[data-bg-slot~="${type}"]`)?.append($("#bg-image"));
  updatePatternThumb();
  schedulePreview();
}

// ── Surface ────────────────────────────────────────────────────────────────────
// The config's `surface` (parametric_surface.normalize_config), edited in place. Plain values
// are inputs with data-surf="path.to.key" (data-idx: an array element); the base's parameters
// and the deformation / camera lists are built from options.surface (the server's registries),
// so a new base or deformation type needs no page changes. Every edit re-renders the preview.

let surface = null;

const getPath = (obj, path) => path.split(".").reduce((o, k) => o?.[k], obj);

function setPath(obj, path, value) {
  const keys = path.split(".");
  const last = keys.pop();
  keys.reduce((o, k) => (o[k] ??= {}), obj)[last] = value;
}

// Nested defaults under a (possibly partial or older) saved surface; arrays are replaced whole.
function withDefaults(defaults, given) {
  if (given === undefined || given === null) return structuredClone(defaults);
  if (typeof defaults !== "object" || Array.isArray(defaults) || typeof given !== "object" || Array.isArray(given)) return given;
  const out = structuredClone(defaults);
  for (const [k, v] of Object.entries(given)) out[k] = k in defaults ? withDefaults(defaults[k], v) : v;
  return out;
}

const inputValue = (el) => (el.type === "checkbox" ? el.checked
  : el.type === "number" ? (el.value === "" ? null : Number(el.value)) : el.value);

function writeSurface(cfg) {
  surface = withDefaults(options.defaults.surface, cfg);
  for (const el of document.querySelectorAll("[data-surf]")) {
    let value = getPath(surface, el.dataset.surf);
    if (el.dataset.idx !== undefined) value = value?.[+el.dataset.idx];
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value ?? "";
  }
  renderSurfaceControls();
}

function readSurfaceInput(el) {
  const value = inputValue(el);
  if (el.dataset.idx !== undefined) getPath(surface, el.dataset.surf)[+el.dataset.idx] = value;
  else setPath(surface, el.dataset.surf, value);
}

const baseSchema = () => options.surface.bases.find((b) => b.id === surface.base.type);
const deformationSchema = () => options.surface.deformations.find((d) => d.id === surface.deformation.type);
const round3 = (x) => Math.round(x * 1000) / 1000;

function surfaceChanged() {
  saveLocal();
  schedulePreview();
}

function renderSurfaceControls() {
  renderBaseParams();
  renderDeformationList();
  renderCameraList();
  $("#surface-light-mode").closest("label").nextElementSibling.querySelectorAll("input").forEach((el, i) => {
    el.disabled = i < 2 && surface.light.mode === "random";  // random lights have no direction
  });
}

function optimizeCheck(key) {
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = !!surface.optimize[key];
  box.title = "optimize";
  box.onchange = () => { surface.optimize[key] = box.checked; surfaceChanged(); };
  return box;
}

function numberInput(value, param, onInput) {
  const input = document.createElement("input");
  input.type = "number";
  input.step = param.step ?? "any";
  if (param.min !== null && param.min !== undefined) input.min = param.min;
  input.value = value ?? "";
  input.oninput = () => { onInput(input.value === "" ? null : Number(input.value)); schedulePreview(); };
  input.onchange = saveLocal;
  return input;
}

// The selected base's parameters, each with an "optimize" checkbox.
function renderBaseParams() {
  const box = $("#surface-base-params");
  const params = baseSchema()?.params || [];
  if (!params.length) {
    const none = document.createElement("div");
    none.className = "muted small";
    none.textContent = "no parameters";
    box.replaceChildren(none);
    return;
  }
  box.replaceChildren(...params.map((p) => {
    const row = document.createElement("div");
    row.className = "param-row";
    const head = document.createElement("label");
    head.className = "check small";
    head.append(optimizeCheck(p.key), p.label);
    head.title = "optimize";
    const values = document.createElement("div");
    values.className = "row";
    const current = surface.base[p.key];
    if (p.size === 1) values.append(numberInput(current, p, (v) => { surface.base[p.key] = v; }));
    else for (let i = 0; i < p.size; i++) values.append(numberInput(current?.[i], p, (v) => { surface.base[p.key][i] = v; }));
    row.append(head, values);
    return row;
  }));
}

function randomValue(range) {
  const [lo, hi] = range;
  return round3(lo + Math.random() * (hi - lo));
}

// A grid editor for a list of items with numeric params: a header (labels, optional optimize
// checkboxes), optional random range rows (min / max) and one row per item, each with
// `buttons` ({text, title, onclick(i)}).
function paramList(container, { params, items, optimize = false, random = null, buttons = [] }) {
  container.style.gridTemplateColumns = `1.8em repeat(${params.length}, minmax(0, 1fr)) repeat(${buttons.length}, 1.7em)`;
  const cells = [];
  const text = (t, cls = "idx") => { const el = document.createElement("span"); el.className = cls; el.textContent = t; return el; };
  const pad = () => buttons.map(() => text(""));
  cells.push(text(""), ...params.map((p) => {
    const head = document.createElement("label");
    head.className = "head";
    if (optimize) { head.append(optimizeCheck(p.key)); head.title = `optimize ${p.label}`; }
    head.append(p.label);
    return head;
  }), ...pad());
  if (random) {
    for (const [i, name] of [[0, "min"], [1, "max"]]) {
      cells.push(text(name, "idx muted"), ...params.map((p) => (random[p.key]
        ? numberInput(random[p.key][i], { step: p.step }, (v) => { random[p.key][i] = v; })
        : text(""))), ...pad());
    }
  }
  items.forEach((item, i) => {
    cells.push(text(String(i + 1)), ...params.map((p) => numberInput(item[p.key], p, (v) => { item[p.key] = v; })),
      ...buttons.map((b) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = b.text;
        button.title = b.title;
        if (b.text === "×") button.className = "remove";
        button.onclick = () => b.onclick(i);
        return button;
      }));
  });
  container.replaceChildren(...cells);
}

function randomItem(schema, ranges) {
  return Object.fromEntries(schema.params.map((p) => [p.key, ranges[p.key] ? randomValue(ranges[p.key]) : p.default]));
}

function renderDeformationList() {
  const schema = deformationSchema();
  const d = surface.deformation;
  $("#deformation-count-label").textContent = `${schema.item_label[0].toUpperCase()}${schema.item_label.slice(1)}s`;
  $("#deformation-count").value = d.items.length;
  paramList($("#deformation-list"), {
    params: schema.params, items: d.items, optimize: true, random: d.random,
    buttons: [
      { text: "⚄", title: `random ${schema.item_label} (within min / max)`, onclick: (i) => { d.items[i] = randomItem(schema, d.random); renderDeformationList(); surfaceChanged(); } },
      { text: "×", title: "remove", onclick: (i) => { d.items.splice(i, 1); renderDeformationList(); surfaceChanged(); } },
    ],
  });
}

function setDeformationCount(n) {
  const schema = deformationSchema();
  const d = surface.deformation;
  n = Math.max(0, Math.min(256, Math.round(n) || 0));
  while (d.items.length < n) d.items.push(randomItem(schema, d.random));
  d.items.length = n;
  renderDeformationList();
  surfaceChanged();
}

function renderCameraList() {
  $("#camera-count").value = surface.cameras.length;
  paramList($("#camera-list"), {
    params: options.surface.camera_params, items: surface.cameras,
    buttons: [{ text: "×", title: "remove", onclick: (i) => {
      if (surface.cameras.length > 1) surface.cameras.splice(i, 1);
      renderCameraList();
      surfaceChanged();
    } }],
  });
}

// New cameras copy the last one, turned on by 360° / count.
function setCameraCount(n) {
  const cams = surface.cameras;
  n = Math.max(1, Math.min(32, Math.round(n) || 1));
  while (cams.length < n) {
    const last = cams[cams.length - 1];
    cams.push({ ...last, azimuth: round3(((last.azimuth ?? 0) + 360 / n) % 360) });
  }
  cams.length = n;
  renderCameraList();
  surfaceChanged();
}

function spreadCameras() {
  const cams = surface.cameras;
  const start = cams[0].azimuth ?? 0;
  cams.forEach((c, i) => { c.azimuth = round3((start + 360 * i / cams.length) % 360); });
  renderCameraList();
  surfaceChanged();
}

// The preview: one request at a time; edits during a request ask for one more after it.
let previewTimer = 0;
let previewBusy = false;
let previewAgain = false;

// The fields with a preview: its endpoint, image and note.
const PREVIEWS = {
  SURFACE: { url: "/api/preview/surface", img: "#surface-preview", note: "#surface-preview-note" },
  REVOLUTION: { url: "/api/preview/revolution", img: "#revolution-preview", note: "#revolution-preview-note" },
  MESH_TEXTURE: { url: "/api/preview/mesh", img: "#mesh-preview", note: "#mesh-preview-note" },
  GAUSSIANS_3D: { url: "/api/preview/gaussians3d", img: "#g3d-preview", note: "#g3d-preview-note" },
  VOXELS: { url: "/api/preview/voxels", img: "#vol-preview", note: "#vol-preview-note" },
  ASSEMBLY: { url: "/api/preview/solid/assembly", img: "#vol-preview", note: "#vol-preview-note" },
  SOLIDS_ON_OFF: { url: "/api/preview/solid/solids_on_off", img: "#vol-preview", note: "#vol-preview-note" },
  EXTRUSIONS: { url: "/api/preview/solid/extrusions", img: "#vol-preview", note: "#vol-preview-note" },
};

// The field types listed under "3D" in the field type menu.
const FIELDS_3D = ["SURFACE", "REVOLUTION", "MESH_TEXTURE", "GAUSSIANS_3D", "VOXELS", "ASSEMBLY", "SOLIDS_ON_OFF",
                   "EXTRUSIONS"];

function schedulePreview() {
  const type = $("#field-type").value;
  if (!{ SURFACE: surface, REVOLUTION: revolution, MESH_TEXTURE: meshTexture, GAUSSIANS_3D: gaussians3d,
         VOXELS: voxels, ASSEMBLY: assembly, SOLIDS_ON_OFF: onOff, EXTRUSIONS: extrusions }[type]) return;
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPreview, 120);
}

async function refreshPreview() {
  if (previewBusy) { previewAgain = true; return; }
  const type = $("#field-type").value;
  const target = PREVIEWS[type];
  if (!target) return;
  previewBusy = true;
  const note = $(target.note);
  try {
    if (type === "MESH_TEXTURE" && !meshTexture.path) throw new Error("Choose or upload a .3dm mesh");
    const canvas = [...document.querySelectorAll('[data-key="canvas"]')].map((el) => Number(el.value) || 512);
    const t0 = performance.now();
    const background = { background_image: $("#bg-image-path").value, background_image_fit: $("#bg-image-fit").value,
                         background_image_multiply: $("#bg-image-multiply").checked };
    // What the textured fields start from
    const scene = { background_color: document.querySelector('[data-key="background_color"]').value,
                    start_image: $("#start-image").value, start_image_fit: $("#start-image-fit").value,
                    canvas_init: $("#canvas-init").value };
    const body = type === "SURFACE" ? { surface, canvas, device: $("#device").value, ...background }
      : type === "REVOLUTION" ? { revolution, canvas, ...background, ...scene }
      : type === "GAUSSIANS_3D" || type in VOLUME_CONFIGS
        ? { ...(type === "GAUSSIANS_3D" ? { gaussians_3d: gaussians3d } : { [type.toLowerCase()]: VOLUME_CONFIGS[type]() }),
            canvas, ...background, ...scene,
            seed: Number(document.querySelector('[data-key="seed"]').value) || 0,
            canvas_noise_std: document.querySelector('[data-key="canvas_noise_std"]').value }
      : { mesh_texture: meshTexture, canvas, ...background, ...scene };
    const res = await fetch(target.url, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const img = $(target.img);
    const old = img.src;
    img.src = URL.createObjectURL(await res.blob());
    if (old.startsWith("blob:")) URL.revokeObjectURL(old);
    const what = type === "REVOLUTION" ? revolution.mode
      : type === "MESH_TEXTURE" ? res.headers.get("X-Mesh")
      : type === "GAUSSIANS_3D" ? `${gaussians3d.count} splats, ${gaussians3d.views.count} cameras`
      : type in VOLUME_CONFIGS ? `${volumeConfig().resolution}³ voxels, ${volumeConfig().views.count} cameras`
      : `${surface.cameras.length} camera${surface.cameras.length > 1 ? "s" : ""}`;
    note.textContent = `${what} · ${Math.round(performance.now() - t0)} ms`;
    note.classList.remove("error");
  } catch (err) {
    note.textContent = err.message;
    note.classList.add("error");
  } finally {
    previewBusy = false;
    if (previewAgain) { previewAgain = false; refreshPreview(); }
  }
}

// The run's current parameters (state.field_values), and adopting them as the start.
let fieldValues = null;

function showFieldValues(values) {
  fieldValues = values;
  $("#surface-current-block").hidden = !values;
  showRevolutionValues(values);
  showMeshValues(values);
  $("#g3d-current").textContent = values && "mean opacity" in values
    ? Object.entries(values).map(([k, v]) => `${k}  ${v}`).join("\n") : "";
  $("#vol-current").textContent = values && !values.base && lastState?.field_type in VOLUME_CONFIGS
    ? Object.entries(values).map(([k, v]) => `${k}  ${v}`).join("\n") : "";
  if (!values) return;
  const lines = Object.entries(values.base || {}).map(([k, v]) => `${k}  ${[].concat(v).map((x) => x.toFixed(3)).join("  ")}`);
  const items = values.deformation?.items || [];
  const schema = deformationSchema();
  if (items.length && schema) {
    const cell = (s) => String(s).padStart(8);
    lines.push("  #" + schema.params.map((p) => cell(p.label)).join(""));
    items.forEach((item, i) => lines.push(String(i + 1).padStart(3) + schema.params.map((p) => cell((item[p.key] ?? 0).toFixed(3))).join("")));
  }
  $("#surface-current").textContent = lines.join("\n");
}

function adoptFieldValues() {
  if (!fieldValues) return;
  Object.assign(surface.base, structuredClone(fieldValues.base || {}));
  if (fieldValues.deformation?.items) surface.deformation.items = structuredClone(fieldValues.deformation.items);
  writeSurface(surface);
  surfaceChanged();
}

function setupSurface() {
  const s = options.surface;
  $("#surface-base").replaceChildren(...s.bases.map((b) => new Option(b.label, b.id)));
  $("#surface-deformation").replaceChildren(...s.deformations.map((d) => new Option(d.label, d.id)));
  $("#surface-shading").replaceChildren(...s.shadings.map((m) => new Option(m, m)));
  $("#surface-light-mode").replaceChildren(...s.light_modes.map((m) => new Option(m.label, m.id)));
  $("#surface-renderer").replaceChildren(...s.renderers.map((r) => new Option(r.label, r.id)));
  $("#surface-panel").addEventListener("input", (e) => {
    const el = e.target;
    if (!el.dataset.surf) return;
    readSurfaceInput(el);
    if (["base.type", "deformation.type", "light.mode"].includes(el.dataset.surf)) renderSurfaceControls();
    schedulePreview();
  });
  $("#deformation-count").onchange = (e) => setDeformationCount(Number(e.target.value));
  $("#deformation-randomize").onclick = () => {
    const schema = deformationSchema();
    surface.deformation.items = surface.deformation.items.map(() => randomItem(schema, surface.deformation.random));
    renderDeformationList();
    surfaceChanged();
  };
  $("#camera-count").onchange = (e) => setCameraCount(Number(e.target.value));
  $("#cameras-spread").onclick = spreadCameras;
  $("#surface-adopt").onclick = adoptFieldValues;
  for (const el of document.querySelectorAll('[data-key="canvas"]')) el.addEventListener("change", schedulePreview);
}

// ── Level set ──────────────────────────────────────────────────────────────────
// The config's `levelset` (optimization.normalize_levelset): a background and a list of layers
// {resolution [w, h], color (hex), alpha, optimize_color, contrast, optimize_contrast}.

let levelset = null;

// The blocks' controls, shared with the function stack.
const layerText = (t, cls = "head") => { const el = document.createElement("span"); el.className = cls; el.textContent = t; return el; };
const layerCheck = (item, key, title) => {
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = !!item[key];
  box.title = title;
  box.onchange = () => { item[key] = box.checked; };
  return box;
};
// a small label before a control
const layerField = (label, control, title) => {
  const el = document.createElement("label");
  el.className = "field";
  if (title) el.title = title;
  el.append(layerText(label, "head"), control);
  return el;
};
const layerLine = (...children) => { const el = document.createElement("div"); el.className = "layer-line"; el.append(...children); return el; };
// A checkbox with its label
const layerOpt = (item, key, title, label = "opt") => {
  const el = document.createElement("label");
  el.className = "check field opt";
  el.title = title;
  el.append(layerCheck(item, key, title), layerText(label, "head"));
  return el;
};
const layerRemove = (onclick) => {
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "×";
  remove.title = "remove";
  remove.className = "remove";
  remove.onclick = onclick;
  return remove;
};

function writeLevelset(cfg) {
  levelset = withDefaults(options.defaults.levelset, cfg);
  const layerDefault = options.defaults.levelset.layers[0];
  levelset.layers = levelset.layers.map((l) => ({ ...structuredClone(layerDefault), ...l }));
  $("#levelset-background").value = levelset.background;
  $("#levelset-optimize-background").checked = !!levelset.optimize_background;
  renderLevelsetList();
}

function renderLevelsetList() {
  const layers = levelset.layers;
  $("#levelset-count").value = layers.length;
  const [text, field, line, opt] = [layerText, layerField, layerLine, layerOpt];
  // Four lines per layer: resolution / colour + alpha / contrast / coverage budget and its weight.
  const blocks = layers.map((layer, i) => {
    const color = document.createElement("input");
    color.type = "color";
    color.value = layer.color;
    color.oninput = () => { layer.color = color.value; };
    const remove = layerRemove(() => { if (layers.length > 1) layers.splice(i, 1); renderLevelsetList(); saveLocal(); });
    const coverage = numberInput(layer.coverage, { step: 0.05, min: 0 }, (v) => { layer.coverage = v; });
    coverage.max = 1;
    coverage.placeholder = "free";
    const indent = () => text("", "idx");
    const block = document.createElement("div");
    block.className = "layer";
    block.append(
      line(text(String(i + 1), "idx"),
        field("W", numberInput(layer.resolution[0], { step: 1, min: 1 }, (v) => { layer.resolution[0] = v; })),
        field("H", numberInput(layer.resolution[1], { step: 1, min: 1 }, (v) => { layer.resolution[1] = v; })),
        remove),
      line(indent(),
        field("color", color),
        field("α", numberInput(layer.alpha, { step: 0.05, min: 0 }, (v) => { layer.alpha = v; })),
        opt(layer, "optimize_color", "optimize the colour (RGBA); unchecked = frozen")),
      line(indent(),
        field("contrast", numberInput(layer.contrast, { step: 1, min: 0 }, (v) => { layer.contrast = v; }),
              "contrast sigmoid factor (0 = linear)"),
        opt(layer, "optimize_contrast", "optimize the contrast (in log space, so it must start > 0); unchecked = frozen")),
      line(indent(),
        field("cover", coverage, "coverage budget: the share of the canvas the mask should cover, 0..1 (empty = free)"),
        field("w", numberInput(layer.coverage_weight, { step: 1, min: 0 }, (v) => { layer.coverage_weight = v; }),
              "weight of the coverage budget: w x (mean mask - cover)^2")));
    return block;
  });
  $("#levelset-list").replaceChildren(...blocks);
}

// New layers copy the last one.
function setLevelsetCount(n) {
  const layers = levelset.layers;
  n = Math.max(1, Math.min(64, Math.round(n) || 1));
  while (layers.length < n) layers.push(structuredClone(layers[layers.length - 1]));
  layers.length = n;
  renderLevelsetList();
  saveLocal();
}

function setupLevelset() {
  $("#levelset-background").oninput = (e) => { levelset.background = e.target.value; };
  $("#levelset-optimize-background").onchange = (e) => { levelset.optimize_background = e.target.checked; };
  $("#levelset-count").onchange = (e) => setLevelsetCount(Number(e.target.value));
}

// ── Function stack ─────────────────────────────────────────────────────────────
// The config's `function_stack` (optimization.normalize_function_stack): a base shape and a list of
// functions {wave, offset, optimize_offset, components, frequency, optimize_frequency, amplitude, color, alpha,
// optimize_color, filled, thickness}. The background is the shared background_color.

let functionStack = null;

function writeFunctionStack(cfg) {
  functionStack = withDefaults(options.defaults.function_stack, cfg);
  const functionDefault = options.defaults.function_stack.functions[0];
  functionStack.functions = functionStack.functions.map((f) => ({ ...structuredClone(functionDefault), ...f }));
  $("#function-kind").value = functionStack.kind;
  $("#function-softness").value = functionStack.softness;
  $("#function-shape-lr").value = functionStack.shape_lr;
  renderFunctionList();
}

function renderFunctionList() {
  const functions = functionStack.functions;
  const blocks = functions.map((fn, i) => {
    const wave = document.createElement("select");
    wave.replaceChildren(...options.function_waves.map((w) => new Option(w, w)));
    wave.value = fn.wave;
    wave.onchange = () => { fn.wave = wave.value; };
    const color = document.createElement("input");
    color.type = "color";
    color.value = fn.color;
    color.oninput = () => { fn.color = color.value; };
    const remove = layerRemove(() => { if (functions.length > 1) functions.splice(i, 1); renderFunctionList(); saveLocal(); });
    remove.disabled = functions.length === 1;
    const thickness = numberInput(fn.thickness, { step: 0.005, min: 0 }, (v) => { fn.thickness = v; });
    thickness.disabled = fn.filled;
    const filled = layerOpt(fn, "filled", "filled: below the line / inside the circle; else a stroke", "filled");
    filled.querySelector("input").addEventListener("change", () => { thickness.disabled = fn.filled; });
    const indent = () => layerText("", "idx");
    const block = document.createElement("div");
    block.className = "layer";
    block.append(
      layerLine(layerText(String(i + 1), "idx"), wave,
        layerField("waves", numberInput(fn.components, { step: 1, min: 1 }, (v) => { fn.components = v; }),
                   "components: how many waves are superimposed, starting at 1, 2, 3 ... x the frequency"),
        remove),
      layerLine(indent(),
        layerField("offset", numberInput(fn.offset, { step: 0.05 }, (v) => { fn.offset = v; }),
                   "line: moved up from the canvas centre. circle: added to the base radius"),
        layerOpt(fn, "optimize_offset", "optimize the offset; unchecked = frozen")),
      layerLine(indent(),
        layerField("amp", numberInput(fn.amplitude, { step: 0.01, min: 0 }, (v) => { fn.amplitude = v; }),
                   "starting amplitude of the first wave (wave k starts at amp / k); they are optimized"),
        layerField("freq", numberInput(fn.frequency, { step: 1, min: 0 }, (v) => { fn.frequency = v; }),
                   "starting cycles of the first wave across the canvas' width, or around the circle (only whole numbers close it)"),
        layerOpt(fn, "optimize_frequency", "optimize each wave's frequency (a circle then gets a seam at its left); unchecked = frozen")),
      layerLine(indent(),
        layerField("color", color),
        layerField("α", numberInput(fn.alpha, { step: 0.05, min: 0 }, (v) => { fn.alpha = v; })),
        layerOpt(fn, "optimize_color", "optimize the colour (RGBA); unchecked = frozen")),
      layerLine(indent(), filled,
        layerField("thickness", thickness, "stroke width, when not filled")));
    return block;
  });
  $("#function-list").replaceChildren(...blocks);
}

function setupFunctionStack() {
  $("#function-kind").replaceChildren(...options.function_kinds.map((k) => new Option(k, k)));
  $("#function-kind").onchange = (e) => { functionStack.kind = e.target.value; };
  $("#function-softness").oninput = (e) => { functionStack.softness = e.target.value === "" ? null : Number(e.target.value); };
  $("#function-shape-lr").oninput = (e) => { functionStack.shape_lr = e.target.value === "" ? null : Number(e.target.value); };
  $("#function-add").onclick = () => {
    const functions = functionStack.functions;
    functions.push(structuredClone(functions[functions.length - 1]));
    renderFunctionList();
    saveLocal();
  };
}

// ── Surface of revolution ──────────────────────────────────────────────────────
// The config's `revolution` (surface_of_revolution.normalize_config), edited in place. Plain values
// are inputs with data-rev="path.to.key" (data-idx: an array element); each mode's list is built here.

let revolution = null;
let formulaRequest = 0;  // the latest formula check; an older answer is dropped

function writeRevolution(cfg) {
  revolution = withDefaults(options.defaults.revolution, cfg);
  // A formula's own parameters, not merged with those of the default formula.
  if (cfg?.parametric?.parameters) revolution.parametric.parameters = structuredClone(cfg.parametric.parameters);
  for (const el of document.querySelectorAll("[data-rev]")) {
    let value = getPath(revolution, el.dataset.rev);
    if (el.dataset.idx !== undefined) value = value?.[+el.dataset.idx];
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value ?? "";
  }
  $("#revolution-formula").value = revolution.parametric.formula;
  renderRevolutionControls();
}

function syncRevolutionControls() {
  for (const el of document.querySelectorAll(".rev-mode")) el.hidden = el.dataset.mode !== revolution.mode;
  $("#revolution-texture-options").hidden = !revolution.texture.enabled;
  $("#revolution-color-row").hidden = revolution.texture.enabled;
  for (const el of $("#revolution-panel").querySelectorAll('[data-rev^="shading.a"], [data-rev="shading.elevation"]')) {
    el.disabled = revolution.shading.mode === "flat";
  }
  $('[data-rev="optimize_radius"]').disabled = !revolution.optimize_profile;
}

function renderRevolutionControls() {
  syncRevolutionControls();
  renderRevolutionPoints();
  renderRevolutionWaves();
  renderRevolutionParameters();
}

function selectInput(choices, value, onChange) {
  const select = document.createElement("select");
  select.replaceChildren(...choices.map((c) => new Option(c, c)));
  select.value = value;
  select.onchange = () => onChange(select.value);
  return select;
}

function renderRevolutionPoints() {
  const points = revolution.points;
  $("#revolution-point-count").value = points.length;
  $("#revolution-points").replaceChildren(...points.map((point, i) => {
    const method = selectInput(options.revolution.interpolations, point.interpolation, (v) => { point.interpolation = v; });
    method.disabled = i === points.length - 1;  // no span follows the last point
    const block = document.createElement("div");
    block.className = "layer";
    block.append(layerLine(layerText(String(i + 1), "idx"),
      layerField("offset", numberInput(point.offset, { step: 0.01 }, (v) => { point.offset = v; })), method));
    return block;
  }));
}

function setRevolutionPointCount(n) {
  const points = revolution.points;
  n = Math.max(2, Math.min(options.revolution.max_points, Math.round(n) || 2));
  while (points.length < n) points.push(structuredClone(points[points.length - 1]));
  points.length = n;
  renderRevolutionPoints();
  saveLocal();
  schedulePreview();
}

function renderRevolutionWaves() {
  const waves = revolution.waves;
  $("#revolution-waves").replaceChildren(...waves.map((wave, i) => {
    const remove = layerRemove(() => {
      if (waves.length > 1) waves.splice(i, 1);
      renderRevolutionWaves();
      saveLocal();
      schedulePreview();
    });
    remove.disabled = waves.length === 1;
    const block = document.createElement("div");
    block.className = "layer";
    block.append(
      layerLine(layerText(String(i + 1), "idx"),
        selectInput(options.revolution.waves, wave.wave, (v) => { wave.wave = v; }),
        layerField("waves", numberInput(wave.components, { step: 1, min: 1 }, (v) => { wave.components = v; }),
                   "components: how many waves are superimposed, at 1, 2, 3 ... x the frequency"),
        remove),
      layerLine(layerText("", "idx"),
        layerField("amp", numberInput(wave.amplitude, { step: 0.01, min: 0 }, (v) => { wave.amplitude = v; }),
                   "starting amplitude of the first wave (wave k starts at amp / k)"),
        layerField("phase", numberInput(wave.phase, { step: 0.1 }, (v) => { wave.phase = v; }), "starting phase, radians")),
      layerLine(layerText("", "idx"),
        layerField("freq", numberInput(wave.frequency, { step: 0.5, min: 0 }, (v) => { wave.frequency = v; }),
                   "starting cycles of the first wave along the height"),
        layerOpt(wave, "optimize_frequency", "optimize each wave's frequency; unchecked = frozen")));
    return block;
  }));
}

function renderRevolutionParameters() {
  const parameters = revolution.parametric.parameters;
  const blocks = Object.entries(parameters).map(([name, parameter]) => {
    const block = document.createElement("div");
    block.className = "layer";
    block.append(layerLine(layerText(name, "idx"),
      layerField("value", numberInput(parameter.value, { step: 0.01 }, (v) => { parameter.value = v; })),
      layerOpt(parameter, "optimize", "optimize this parameter; unchecked = frozen")));
    return block;
  });
  if (!blocks.length) {
    const none = document.createElement("div");
    none.className = "muted small";
    none.textContent = "no free parameters";
    blocks.push(none);
  }
  $("#revolution-parameters").replaceChildren(...blocks);
}

// The server reads the formula: its free parameters, or what is wrong with it.
async function checkFormula() {
  const request = ++formulaRequest;
  const note = $("#revolution-formula-note");
  try {
    const res = await fetch("/api/revolution/formula", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ formula: revolution.parametric.formula, parameters: revolution.parametric.parameters }),
    });
    const answer = await res.json();
    if (request !== formulaRequest) return;
    if (!res.ok) throw new Error(answer.error || res.statusText);
    revolution.parametric.parameters = answer.parameters;
    note.textContent = "";
    note.classList.remove("error");
    renderRevolutionParameters();
    saveLocal();
    schedulePreview();
  } catch (err) {
    if (request !== formulaRequest) return;
    note.textContent = err.message;
    note.classList.add("error");
  }
}

// The current parameters of a run (state.field_values). SURFACE's have a `base`.
function showRevolutionValues(values) {
  const number = (v) => (typeof v === "number" ? v.toFixed(3) : String(v));
  $("#revolution-current").textContent = values && !values.base
    ? Object.entries(values).map(([k, v]) => `${k}  ${[].concat(v).map(number).join("  ")}`).join("\n")
    : "";
}

function setupRevolution() {
  const r = options.revolution;
  $("#revolution-mode").replaceChildren(...r.modes.map((m) => new Option(m, m)));
  $("#revolution-shading").replaceChildren(...r.shadings.map((m) => new Option(m, m)));
  $("#revolution-point-all").replaceChildren(new Option("(set)", ""), ...r.interpolations.map((m) => new Option(m, m)));
  $("#revolution-formula-label").title =
    `Python syntax, ^ or ** for powers. Functions: ${r.functions.join(", ")}. t runs 0..1 up the surface, pi is 3.14..; any other name is a free parameter.`;
  const panel = $("#revolution-panel");
  panel.addEventListener("input", (e) => {
    const el = e.target;
    if (!el.dataset.rev) return;
    if (el.dataset.idx !== undefined) getPath(revolution, el.dataset.rev)[+el.dataset.idx] = inputValue(el);
    else setPath(revolution, el.dataset.rev, inputValue(el));
    syncRevolutionControls();
    if (el.dataset.rev === "mode" && revolution.mode === "parametric") checkFormula();
    schedulePreview();
  });
  panel.addEventListener("change", () => { saveLocal(); schedulePreview(); });
  $("#revolution-point-count").onchange = (e) => setRevolutionPointCount(Number(e.target.value));
  $("#revolution-point-all").onchange = (e) => {
    if (!e.target.value) return;
    for (const point of revolution.points) point.interpolation = e.target.value;
    e.target.value = "";
    renderRevolutionPoints();
    saveLocal();
    schedulePreview();
  };
  $("#revolution-wave-add").onclick = () => {
    revolution.waves.push(structuredClone(revolution.waves[revolution.waves.length - 1]));
    renderRevolutionWaves();
    saveLocal();
    schedulePreview();
  };
  let formulaTimer = 0;
  $("#revolution-formula").oninput = (e) => {
    revolution.parametric.formula = e.target.value;
    clearTimeout(formulaTimer);
    formulaTimer = setTimeout(checkFormula, 200);
  };
  // What the preview also shows, set outside the panel.
  for (const sel of ["#start-image", "#start-image-fit", "#canvas-init", '[data-key="background_color"]']) {
    $(sel).addEventListener("change", schedulePreview);
  }
}

// ── Mesh texture ───────────────────────────────────────────────────────────────
// The config's `mesh_texture` (mesh_texture.normalize_config), edited in place: inputs with
// data-mesh="path.to.key" (data-idx: an array element). The mesh list is the .3dm files of the
// patterns and uploads folders, as for the shape patterns.

let meshTexture = null;

function writeMeshTexture(cfg) {
  meshTexture = withDefaults(options.defaults.mesh_texture, cfg);
  for (const el of document.querySelectorAll("[data-mesh]")) {
    let value = getPath(meshTexture, el.dataset.mesh);
    if (el.dataset.idx !== undefined) value = value?.[+el.dataset.idx];
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value ?? "";
  }
  syncMeshControls();
}

function syncMeshControls() {
  const select = $("#mesh-select");
  select.value = [...select.options].some((o) => o.value === meshTexture.path) ? meshTexture.path : "";
  $("#mesh-bump-options").hidden = meshTexture.mode !== "bump";
  for (const el of document.querySelectorAll("#mesh-panel .mesh-lit input")) el.disabled = meshTexture.shading.mode === "flat";
  const count = Math.max(1, meshTexture.views.count || 1);
  $('[data-mesh="views.per_step"]').max = count;
  $('[data-mesh="views.canvas_view"]').max = count - 1;
}

function setMeshPath(path) {
  meshTexture.path = path;
  $("#mesh-path").value = path;
  syncMeshControls();
  saveLocal();
  schedulePreview();
}

// The current colours of a run (state.field_values), when they are optimized. SURFACE's have a `base`.
function showMeshValues(values) {
  $("#mesh-current").textContent = values && !values.base
    ? Object.entries(values).map(([k, v]) => `${k}  ${v}`).join("\n") : "";
}

function setupMeshTexture() {
  const m = options.mesh_texture;
  const fill = (sel, items) => $(sel).replaceChildren(...items.map((x) => new Option(x, x)));
  fill("#mesh-mode", m.modes);
  fill("#mesh-uv", m.uv_sources);
  fill("#mesh-normals", m.normals);
  fill("#mesh-shading", m.shadings);
  fill("#mesh-light-space", m.light_spaces);
  $("#mesh-select").replaceChildren(new Option("(path below)", ""),
    ...options.patterns.map((p) => new Option(p.name, p.path)));
  const panel = $("#mesh-panel");
  panel.addEventListener("input", (e) => {
    const el = e.target;
    if (!el.dataset.mesh) return;
    if (el.dataset.idx !== undefined) getPath(meshTexture, el.dataset.mesh)[+el.dataset.idx] = inputValue(el);
    else setPath(meshTexture, el.dataset.mesh, inputValue(el));
    syncMeshControls();
    if (el.dataset.mesh !== "path") schedulePreview();  // a path being typed: once it is changed
  });
  panel.addEventListener("change", () => { saveLocal(); schedulePreview(); });
  $("#mesh-select").addEventListener("change", (e) => { if (e.target.value) setMeshPath(e.target.value); });
  $("#mesh-upload").onclick = () => $("#mesh-file").click();
  $("#mesh-file").onchange = async (e) => {
    if (!e.target.files[0]) return;
    try {
      const path = await uploadFile(e.target.files[0]);
      for (const select of [$("#mesh-select"), $("#pattern-select")]) {
        if (![...select.options].some((o) => o.value === path)) select.appendChild(new Option(path.split(/[\\/]/).pop(), path));
      }
      setMeshPath(path);
    } catch (err) { setStatus(err.message, true); }
    e.target.value = "";
  };
}

// ── 3D Gaussian splats ─────────────────────────────────────────────────────────
// The config's `gaussians_3d` (gaussian_splats_3d.normalize_config), edited in place: inputs with
// data-g3d="path.to.key".

let gaussians3d = null;

function writeGaussians3D(cfg) {
  gaussians3d = withDefaults(options.defaults.gaussians_3d, cfg);
  for (const el of document.querySelectorAll("[data-g3d]")) {
    const value = getPath(gaussians3d, el.dataset.g3d);
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value ?? "";
  }
  syncGaussians3DControls();
}

function syncGaussians3DControls() {
  const count = Math.max(1, gaussians3d.views.count || 1);
  $('[data-g3d="views.per_step"]').max = count;
  $('[data-g3d="views.canvas_view"]').max = count - 1;
}

function setupGaussians3D() {
  const panel = $("#g3d-panel");
  panel.addEventListener("input", (e) => {
    const el = e.target;
    if (!el.dataset.g3d) return;
    setPath(gaussians3d, el.dataset.g3d, inputValue(el));
    syncGaussians3DControls();
    schedulePreview();
  });
  panel.addEventListener("change", () => { saveLocal(); schedulePreview(); });
  // The preview's start also depends on these, set outside the panel.
  for (const sel of ['[data-key="seed"]', '[data-key="canvas_noise_std"]']) $(sel).addEventListener("change", schedulePreview);
}

// ── Volume fields: VOXELS, ASSEMBLY, SOLIDS_ON_OFF, EXTRUSIONS ─────────────────
// Each config is edited in place: its own panel's inputs carry data-vox / data-asm / data-onoff /
// data-ext="path.to.key"; the rendering settings they all share (voxel_field.normalize_render) are one
// block, data-vr="path.to.key", bound to the config of the field type shown. ASSEMBLY's groups and
// EXTRUSIONS' planes are lists built here.

let voxels = null;
let assembly = null;
let onOff = null;
let extrusions = null;
const VOLUME_CONFIGS = { VOXELS: () => voxels, ASSEMBLY: () => assembly, SOLIDS_ON_OFF: () => onOff,
                         EXTRUSIONS: () => extrusions };
const volumeConfig = () => VOLUME_CONFIGS[$("#field-type").value]?.();
const fileName = (path) => String(path).split(/[\\/]/).pop();

function writeInputs(attr, cfg) {
  for (const el of document.querySelectorAll(`[data-${attr}]`)) {
    const value = getPath(cfg, el.dataset[attr]);
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value ?? "";
  }
}

// A panel's inputs (data-<attr>) into its config; get() -> the config, after() resyncs. Paths are
// previewed once changed, not while typed.
function bindPanel(panel, attr, get, after) {
  panel.addEventListener("input", (e) => {
    const el = e.target;
    const key = el.dataset[attr];
    if (!key) return;
    setPath(get(), key, inputValue(el));
    after?.();
    if (el.type !== "text") schedulePreview();
  });
  panel.addEventListener("change", () => { saveLocal(); schedulePreview(); });
}

function writeVoxels(cfg) {
  voxels = withDefaults(options.defaults.voxels, cfg);
  writeInputs("vox", voxels);
  syncVoxelControls();
  writeVolumeRender();
}

function writeSolids(cfg) {
  assembly = withDefaults(options.defaults.assembly, cfg.assembly);
  onOff = withDefaults(options.defaults.solids_on_off, cfg.solids_on_off);
  extrusions = withDefaults(options.defaults.extrusions, cfg.extrusions);
  writeInputs("asm", assembly);
  writeInputs("onoff", onOff);
  writeInputs("ext", extrusions);
  renderAssemblyGroups();
  renderExtrusionPlanes();
  syncOnOffSelect();
  writeVolumeRender();
}

// The shared rendering block shows the shown field type's settings.
function writeVolumeRender() {
  const cfg = volumeConfig();
  if (!cfg) return;
  writeInputs("vr", cfg);
  syncVolumeRender();
}

function syncVolumeRender() {
  const cfg = volumeConfig();
  if (!cfg) return;
  const type = $("#field-type").value;
  // The material is the colour of a volume without colours of its own.
  $("#vol-material-row").hidden = type === "ASSEMBLY" || (type === "VOXELS" && voxels.mode !== "density");
  for (const el of document.querySelectorAll("#vol-render-panel .vol-lit input")) el.disabled = cfg.shading.mode === "flat";
  const count = Math.max(1, cfg.views.count || 1);
  $('[data-vr="views.per_step"]').max = count;
  $('[data-vr="views.canvas_view"]').max = count - 1;
}

function syncVoxelControls() {
  for (const el of document.querySelectorAll(".vox-init-ball")) el.hidden = voxels.init !== "ball";
  for (const el of document.querySelectorAll(".vox-init-mesh")) el.hidden = voxels.init !== "mesh";
  for (const el of document.querySelectorAll("#vox-panel .vox-blur input")) el.disabled = !voxels.blur.every;
  $('#vox-panel [data-vox="blur.color"]').disabled ||= voxels.mode === "density";  // no colour volume
  const select = $("#vox-mesh-select");
  select.value = [...select.options].some((o) => o.value === voxels.init_mesh) ? voxels.init_mesh : "";
  if ($("#field-type").value === "VOXELS") syncVolumeRender();
}

function setVoxelMesh(path) {
  voxels.init_mesh = path;
  $("#vox-mesh-path").value = path;
  syncVoxelControls();
  saveLocal();
  schedulePreview();
}

// A <select> of files ({name, path}), `path` chosen (added when the list lacks it); `none`: a first
// option with value "" and this label.
function fileSelect(files, path, onChange, none = null) {
  const select = document.createElement("select");
  const entries = [...files];
  if (path && !entries.some((f) => f.path === path)) entries.push({ name: fileName(path), path });
  select.replaceChildren(...(none !== null ? [new Option(none, "")] : []), ...entries.map((f) => new Option(f.name, f.path)));
  select.value = path || "";
  select.onchange = () => onChange(select.value);
  return select;
}

// Uploads a file and lists it with the other files of its kind (lists: options' arrays to add it to).
async function uploadInto(file, lists) {
  const path = await uploadFile(file);
  for (const list of lists) if (!list.some((f) => f.path === path)) list.push({ name: fileName(path), path });
  return path;
}

function renderAssemblyGroups() {
  const groups = assembly.groups;
  $("#asm-groups").replaceChildren(...groups.map((group, i) => {
    const color = document.createElement("input");
    color.type = "color";
    color.value = group.color;
    color.oninput = () => { group.color = color.value; schedulePreview(); };
    const remove = layerRemove(() => {
      if (groups.length > 1) groups.splice(i, 1);
      renderAssemblyGroups();
      saveLocal();
      schedulePreview();
    });
    remove.disabled = groups.length === 1;
    const block = document.createElement("div");
    block.className = "layer";
    block.append(
      layerLine(layerText(String(i + 1), "idx"),
        fileSelect(options.volumes, group.path, (v) => { group.path = v; schedulePreview(); }, "(choose a solid)"),
        remove),
      layerLine(layerText("", "idx"),
        layerField("count", numberInput(group.count, { step: 1, min: 0 }, (v) => { group.count = v; }),
                   "how many clones of this solid"),
        layerField("scale", numberInput(group.scale, { step: 0.05, min: 0.02 }, (v) => { group.scale = v; }),
                   "the clones' starting scale: their shape's bounding cube, as a share of the scene's"),
        layerField("color", color, "the clones' colour")));
    return block;
  }));
}

function renderExtrusionPlanes() {
  $("#ext-planes").replaceChildren(...options.solids.planes.map((plane) => {
    const block = document.createElement("div");
    block.className = "layer";
    const label = { xy: "XY", yz: "YZ", zx: "ZX" }[plane];
    const select = fileSelect(options.patterns, extrusions.patterns[plane], (v) => {
      extrusions.patterns[plane] = v;
      schedulePreview();
    }, plane === "xy" ? "(choose a pattern)" : "(as XY)");
    block.append(layerLine(layerOpt(extrusions.planes, plane, `use the ${label} plane`, label), select));
    return block;
  }));
}

function syncOnOffSelect() {
  const select = $("#onoff-select");
  select.replaceChildren(new Option("(path below)", ""), ...options.patterns.map((p) => new Option(p.name, p.path)));
  select.value = options.patterns.some((p) => p.path === onOff.path) ? onOff.path : "";
}

function setupVolumes() {
  const v = options.voxels;
  const fill = (sel, items) => $(sel).replaceChildren(...items.map((x) => new Option(x, x)));
  fill("#vox-mode", v.modes);
  fill("#vox-init", v.inits);
  fill("#vol-shading", v.shadings);
  fill("#vol-light-space", v.light_spaces);
  fill("#asm-composite", options.solids.composites);
  fill("#onoff-composite", options.solids.composites);
  $("#vox-mesh-select").replaceChildren(new Option("(path below)", ""),
    ...options.patterns.map((p) => new Option(p.name, p.path)));

  bindPanel($("#vox-panel"), "vox", () => voxels, syncVoxelControls);
  bindPanel($("#asm-panel"), "asm", () => assembly);
  bindPanel($("#onoff-panel"), "onoff", () => onOff, syncOnOffSelect);
  bindPanel($("#ext-panel"), "ext", () => extrusions);
  bindPanel($("#vol-render-panel"), "vr", volumeConfig, syncVolumeRender);
  $("#field-type").addEventListener("change", writeVolumeRender);

  $("#vox-mesh-select").addEventListener("change", (e) => { if (e.target.value) setVoxelMesh(e.target.value); });
  $("#vox-mesh-upload").onclick = () => $("#vox-mesh-file").click();
  $("#vox-mesh-file").onchange = async (e) => {
    if (!e.target.files[0]) return;
    try {
      const path = await uploadInto(e.target.files[0], [options.patterns, options.volumes]);
      for (const select of [$("#vox-mesh-select"), $("#mesh-select"), $("#pattern-select")]) {
        if (![...select.options].some((o) => o.value === path)) select.appendChild(new Option(fileName(path), path));
      }
      setVoxelMesh(path);
    } catch (err) { setStatus(err.message, true); }
    e.target.value = "";
  };

  $("#asm-group-add").onclick = () => {
    const groups = assembly.groups;
    groups.push(structuredClone(groups[groups.length - 1] ?? options.solids.group_default));
    renderAssemblyGroups();
    saveLocal();
    schedulePreview();
  };
  $("#asm-upload").onclick = () => $("#asm-file").click();
  $("#asm-file").onchange = async (e) => {
    if (!e.target.files[0]) return;
    try {
      const path = await uploadInto(e.target.files[0], [options.volumes]);
      const groups = assembly.groups;
      const last = groups[groups.length - 1];
      if (last && !last.path) last.path = path;  // fill an empty group, else add one
      else groups.push({ ...structuredClone(last ?? options.solids.group_default), path });
      renderAssemblyGroups();
      saveLocal();
      schedulePreview();
    } catch (err) { setStatus(err.message, true); }
    e.target.value = "";
  };

  $("#onoff-select").addEventListener("change", (e) => {
    if (!e.target.value) return;
    onOff.path = $("#onoff-path").value = e.target.value;
    saveLocal();
    schedulePreview();
  });
  $("#onoff-upload").onclick = () => $("#onoff-file").click();
  $("#onoff-file").onchange = async (e) => {
    if (!e.target.files[0]) return;
    try {
      onOff.path = $("#onoff-path").value = await uploadInto(e.target.files[0], [options.patterns, options.volumes]);
      syncOnOffSelect();
      saveLocal();
      schedulePreview();
    } catch (err) { setStatus(err.message, true); }
    e.target.value = "";
  };

  $("#ext-upload").onclick = () => $("#ext-file").click();
  $("#ext-file").onchange = async (e) => {
    if (!e.target.files[0]) return;
    try {
      extrusions.patterns.xy = await uploadInto(e.target.files[0], [options.patterns, options.volumes]);
      renderExtrusionPlanes();
      saveLocal();
      schedulePreview();
    } catch (err) { setStatus(err.message, true); }
    e.target.value = "";
  };
}

// ── Match-target thumbnails ────────────────────────────────────────────────────
// Each match term compares against the start image unless it has an override: click its
// thumbnail to upload one, ↺ reverts to the start image.

function setupMatchTargets() {
  for (const label of document.querySelectorAll("label[data-match]")) {
    const term = label.dataset.match;
    const revert = document.createElement("button");
    revert.type = "button";
    revert.className = "revert";
    revert.textContent = "↺";
    revert.title = "revert to the start image";
    revert.onclick = () => { delete matchTargets[term]; refreshMatchTargets(); saveLocal(); };
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "match-pick";
    pick.onclick = () => {
      const file = document.createElement("input");
      file.type = "file";
      file.accept = "image/*";
      file.onchange = async () => {
        if (!file.files[0]) return;
        try {
          matchTargets[term] = await uploadFile(file.files[0]);
          refreshMatchTargets();
          saveLocal();
        } catch (err) { setStatus(err.message, true); }
      };
      file.click();
    };
    const box = document.createElement("span");
    box.className = "match-target";
    box.append(revert, pick);
    const input = label.querySelector("input");
    label.insertBefore(box, input);
    // the term's name in its own element, so it can fade with the rest when the term is off
    const textNode = [...label.childNodes].find((n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim());
    const name = document.createElement("span");
    name.className = "match-name";
    name.textContent = textNode.textContent.trim();
    textNode.replaceWith(name);
    // ∅: switch the term off (weight 0) / on (weight 1)
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "match-toggle";
    toggle.textContent = "∅";
    toggle.onclick = (e) => {
      e.preventDefault();  // a button in a label: don't also focus the input
      input.value = Number(input.value) ? 0 : 1;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    };
    label.append(toggle);
    input.addEventListener("input", syncMatchOff);
    input.addEventListener("change", syncMatchOff);
  }
}

// A match term at weight 0 fades (name, thumbnail, weight); its ∅ button stays solid.
function syncMatchOff() {
  for (const label of document.querySelectorAll("label[data-match]")) {
    const off = !Number(label.querySelector("input").value);
    label.classList.toggle("off", off);
    const toggle = label.querySelector(".match-toggle");
    if (toggle) toggle.title = off ? "off -- click to turn on (weight 1)" : "click to turn off (weight 0)";
  }
}

function refreshMatchTargets() {
  syncMatchOff();
  const start = $("#start-image").value.trim();
  for (const label of document.querySelectorAll("label[data-match]")) {
    const override = matchTargets[label.dataset.match];
    const path = override || start;
    const pick = label.querySelector(".match-pick");
    if (!pick) continue;
    label.querySelector(".revert").hidden = !override;
    pick.classList.toggle("override", !!override);
    pick.title = override ? `override: ${override}\nclick to replace`
      : start ? "start image -- click to upload an override" : "no start image -- click to upload a target";
    if (path) {
      const img = document.createElement("img");
      img.src = thumbUrl(path, 64);
      img.alt = "";
      pick.replaceChildren(img);
    } else {
      pick.textContent = "+";
    }
  }
}

function updateStartThumb() {
  refreshMatchTargets();
  const path = $("#start-image").value.trim();
  const img = $("#start-image-thumb");
  img.hidden = !path;
  if (!path) return;
  const cfg = readForm();
  const q = new URLSearchParams({ path, size: 320, fit: cfg.start_image_fit || "stretch", w: cfg.canvas[0], h: cfg.canvas[1] });
  img.src = `/api/preview/image?${q}`;
}

function updatePatternThumb() {
  const img = $("#pattern-thumb");
  const path = $("#shape-src").value.trim();
  if (!path || !["SHAPES", "RELIEF"].includes($("#field-type").value)) { img.hidden = true; return; }
  const cfg = readForm();
  img.hidden = false;
  img.src = `/api/preview/pattern?path=${encodeURIComponent(path)}&w=${cfg.canvas[0]}&h=${cfg.canvas[1]}&border=${cfg.shape_border || 0}`;
}

function syncPatternSelect() {
  const select = $("#pattern-select");
  const path = $("#shape-src").value.trim();
  const match = [...select.options].find((o) => o.value === path);
  select.value = match ? path : "";
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data.path;
}

// Overlay a server preset on the current form; prompts, images, model stay as they are.
function applyPreset(name) {
  const cfg = { ...readForm(), ...options.presets[name] };
  if (name === "generation" && ["DISTORTION", "REARRANGE"].includes(cfg.field_type)) cfg.field_type = "PIXELS_COLOR";
  writeForm(cfg);
  saveLocal();
  $("#preset-note").textContent = `${name} defaults applied -- prompts, images and model unchanged`;
}

// ── Run control ────────────────────────────────────────────────────────────────

// One line in the status bar (the full text in its tooltip)
function setStatus(text, isError = false) {
  const el = $("#status");
  el.textContent = text.replace(/\n+/g, "  ·  ");
  el.title = text;
  el.classList.toggle("error", isError);
}

// Run control icons (24 x 24, currentColor)
const ICONS = {
  play: '<path d="M7 4.5v15l12.5-7.5z"/>',
  pause: '<path d="M6.5 5h4v14h-4zM13.5 5h4v14h-4z"/>',
  stop: '<path d="M6 6h12v12H6z"/>',
  restart: '<path d="M12 5a7 7 0 1 1-6.6 4.7l1.9.7A5 5 0 1 0 12 7v3L7.5 6 12 2z"/>',
};
function setIcon(button, name, title) {
  if (button.dataset.icon !== name) {
    button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;
    button.dataset.icon = name;
  }
  button.title = title;
  button.setAttribute("aria-label", title);
}

// The run buttons act on press (pointer down), not click: pressed, the button slides into its
// depth and its clip-path shrinks the hit area, so a mouse-up near its edge would miss it and
// the click never fire. Keyboard activation (Enter / Space: a click with detail 0) still works.
function onPress(button, action) {
  button.addEventListener("pointerdown", (e) => { if (e.button === 0) action(); });
  button.addEventListener("click", (e) => { if (e.detail === 0) action(); });
}

// Run controls by run state: idle / finished -> play; running -> restart, pause, stop;
// paused -> restart, resume (play), stop.
function syncRunControls(status, paused) {
  const active = ["loading", "running", "paused"].includes(status);
  setIcon($("#start"), active ? "restart" : "play",
          active ? "Restart: stop this run and start over with the current settings" : "Start a run with these settings");
  $("#pause").hidden = $("#stop").hidden = !active;
  setIcon($("#pause"), paused ? "play" : "pause", paused ? "Resume" : "Pause at the next step");
  $("#pause").classList.toggle("on", !!paused);
  setIcon($("#stop"), "stop", "Stop the run (what it has so far is saved)");
}

// Results of the save / load buttons (the status box belongs to the run)
function setNote(text, isError = false) {
  const el = $("#note");
  el.hidden = !text;
  el.textContent = text.replace(/\n+/g, "  ·  ");
  el.title = text;
  el.classList.toggle("error", isError);
}

async function postJson(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

const embedFiles = () => $("#embed-files").checked;

async function togglePause() {
  try {
    const data = await postJson("/api/pause", { paused: !lastState?.paused });
    syncRunControls(lastState?.status, data.paused);
  } catch (err) { setNote(err.message, true); }
}

async function saveSnapshot() {
  setNote("saving ...");
  try {
    const data = await postJson("/api/snapshot", { embed: embedFiles() });
    setNote(`saved ${data.files.length} files to ${data.folder}`);
  } catch (err) { setNote(err.message, true); }
}

// The settings as a config file, downloaded (images embedded unless "embed images" is off)
async function saveConfig() {
  try {
    const doc = await postJson("/api/config/pack", { config: readForm(), embed: embedFiles() });
    const blob = new Blob([JSON.stringify(doc, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `optimization_config_${doc.saved.replace(/[-: ]/g, "").replace(/(\d{8})(\d{6})/, "$1_$2")}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    const n = Object.keys(doc.files).length;
    setNote(`config saved${embedFiles() ? `, ${n} file${n === 1 ? "" : "s"} embedded` : " (paths only)"}`);
  } catch (err) { setNote(err.message, true); }
}

async function loadConfig(file) {
  try {
    const data = await postJson("/api/config/unpack", JSON.parse(await file.text()));
    writeForm(data.config);
    saveLocal();
    setNote([`loaded ${file.name}`, ...data.notes].join("\n"), data.notes.some((n) => n.startsWith("missing")));
  } catch (err) { setNote(`${file.name}: ${err.message}`, true); }
}

async function startRun() {
  saveLocal();
  const res = await fetch("/api/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(readForm()),
  });
  const data = await res.json();
  if (!res.ok) { setStatus(data.error || res.statusText, true); return; }
  resetRunView(data.run_id);
  setStatus("starting ...");
}

function resetRunView(id) {
  runId = id;
  traces = { plus: [], minus: [] };
  termTraces = { plus: [], minus: [] };
  frameVersion = -1;
  frameNames = new Set();
  for (const id of ["#img-canvas", "#img-gradient"]) $(id).removeAttribute("src");
  for (const img of document.querySelectorAll("#inspector figure img")) img.removeAttribute("src");
  syncFrames();
}

// ── Frames: the current image (or the composite, once done) in the middle, the rest in the
// inspector, each shown when its toggle is on and the run has that frame ──────────────────

const SHOW_KEY = "clip-optimization-show";
let frameNames = new Set();

const showing = (what) => !!document.querySelector(`#inspector input[data-show="${what}"]`)?.checked;

function syncFrames() {
  for (const fig of document.querySelectorAll("#inspector figure[data-frame]")) {
    fig.hidden = !showing(fig.dataset.show) || !frameNames.has(fig.dataset.frame);
  }
  // The gradient is the canvas view's: not over the sheet of all views.
  $("#img-gradient").hidden = !showing("gradient") || !frameNames.has("gradient") || showingAllViews();
  syncAllViewsButton();
}

// ── All views: the 3D fields' toggle over the image (top right). On, the main image is the
// "all_views" frame, every camera tiled, which the server renders only while it is asked for. ──

const ALL_VIEWS_KEY = "clip-optimization-all-views";
const ALL_VIEWS_FIELDS = ["REVOLUTION", "MESH_TEXTURE", "GAUSSIANS_3D", "VOXELS", "ASSEMBLY", "SOLIDS_ON_OFF",
                          "EXTRUSIONS"];
let allViews = false;
try { allViews = localStorage.getItem(ALL_VIEWS_KEY) === "true"; } catch { /* storage unavailable */ }

// The run's field type once there is a run, else the form's.
const viewFieldType = () => (lastState?.run_id ? lastState.field_type : $("#field-type")?.value);
const allViewsAvailable = () => ALL_VIEWS_FIELDS.includes(viewFieldType()) && frameNames.has("canvas");
const showingAllViews = () => allViews && allViewsAvailable() && frameNames.has("all_views");

function syncAllViewsButton() {
  const button = $("#all-views");
  button.hidden = !allViewsAvailable();
  button.classList.toggle("on", allViews);
  button.setAttribute("aria-pressed", String(allViews));
}

function setupAllViews() {
  $("#all-views").onclick = () => {
    allViews = !allViews;
    try { localStorage.setItem(ALL_VIEWS_KEY, String(allViews)); } catch { /* storage unavailable */ }
    sendDisplay();
    frameVersion = -1;  // the other image at the next poll
    syncFrames();
  };
}

function setupInspector() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(SHOW_KEY) || "{}"); } catch { /* storage unavailable */ }
  for (const box of document.querySelectorAll("#inspector input[data-show]")) {
    if (box.dataset.show in saved) box.checked = !!saved[box.dataset.show];
    box.addEventListener("change", () => {
      const state = Object.fromEntries([...document.querySelectorAll("#inspector input[data-show]")]
        .map((b) => [b.dataset.show, b.checked]));
      try { localStorage.setItem(SHOW_KEY, JSON.stringify(state)); } catch { /* storage unavailable */ }
      if (box.dataset.show === "gradient") sendDisplay();
      frameVersion = -1;  // fetch what just came on show at the next poll
      syncFrames();
    });
  }
  sendDisplay();
}

// The server keeps / sends the gradient and renders all views only while they are shown (they cost
// the run a little).
function sendDisplay() {
  postJson("/api/display", { gradient: showing("gradient"), all_views: allViews })
    .catch((err) => setNote(err.message, true));
}

async function poll() {
  try {
    const q = new URLSearchParams({ run: runId, plus: traces.plus.length, minus: traces.minus.length,
                                    tplus: termTraces.plus.length, tminus: termTraces.minus.length,
                                    log: statusLogVersion });
    const res = await fetch(`/api/state?${q}`);
    const state = await res.json();
    showStatusLog(state.status_log);
    if (state.run_id !== runId) {
      resetRunView(state.run_id);
    } else {
      for (const pole of ["plus", "minus"]) {
        // Only append when the server's offset matches what we hold (no gaps / duplicates).
        if (state.trace_offsets[pole] === traces[pole].length) traces[pole].push(...state.traces[pole]);
        if (state.term_offsets[pole] === termTraces[pole].length) termTraces[pole].push(...state.term_traces[pole]);
      }
    }
    lastState = state;
    showState(state);
  } catch (err) {
    setStatus(`server unreachable: ${err.message}`, true);
  }
  setTimeout(poll, POLL_MS);
}

function formatDuration(s) {
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h${String(Math.floor(s / 60) % 60).padStart(2, "0")}m`;
}

// "  ·  12.3 fps  ·  eta 1m20s": the recent steps / second and the time left at that pace
// (both poles of a bidirectional run; an upper bound when a similarity budget can stop it early).
function speedText(state) {
  if (!state.fps) return "";
  let left = state.epochs - state.step;
  if (state.bidirectional && state.pole === "plus") left += state.epochs;
  const fps = state.fps >= 10 ? state.fps.toFixed(0) : state.fps.toFixed(1);
  return `  ·  ${fps} fps  ·  eta ${formatDuration(left / state.fps)}`;
}

function showState(state) {
  const pole = state.pole === "minus" ? "−" : "+";
  let text = state.status;
  if (state.status === "running") text = `running  [${pole}]  step ${state.step}/${state.epochs}${speedText(state)}`;
  if (state.status === "paused") text = `paused  [${pole}]  step ${state.step}/${state.epochs}`;
  if (["loading", "running", "paused"].includes(state.status)) text += `\non ${state.device}`;
  const active = state.axes?.[state.active_target];
  if (state.status === "running" && active) {
    text += `
target ${state.active_target + 1}: ${active.kind === "direction" ? `${active.source} → ${active.label}` : active.label}`;
  }
  if (state.message) text += `\n${state.message}`;
  if (state.status === "done" && state.result_folder) text += `\nsaved: ${state.result_folder}`;
  setStatus(text, state.status === "error");
  syncRunControls(state.status, state.paused);

  const sim = state.stats?.["sim->start"];
  const stop = state.stop_at_similarity;
  $("#budget").hidden = stop === null || sim === undefined;
  if (stop !== null && sim !== undefined) {
    $("#budget-bar").style.width = `${Math.min(100, Math.max(0, (1 - sim) / (1 - stop) * 100))}%`;
  }

  if (state.frame_version !== frameVersion) {
    frameVersion = state.frame_version;
    const v = `?v=${state.run_id}-${frameVersion}`;
    frameNames = new Set(state.frames);
    const set = (img, name) => { if (frameNames.has(name)) img.src = `/api/frame/${name}.jpg${v}`; };
    set($("#img-canvas"), showingAllViews() ? "all_views" : "canvas");
    if (showing("gradient")) set($("#img-gradient"), "gradient");
    // only the frames on show are fetched
    for (const fig of document.querySelectorAll("#inspector figure[data-frame]")) {
      if (showing(fig.dataset.show)) set(fig.querySelector("img"), fig.dataset.frame);
    }
    syncFrames();
  }
  if (JSON.stringify(state.field_values) !== JSON.stringify(fieldValues)) showFieldValues(state.field_values);
  drawPlot();
  drawValues();
}

// ── Target map ─────────────────────────────────────────────────────────────────
// Each axis is a full line through the origin at angle k*pi/K. A point with axis
// values v projects to sum_k (v_k / s_k) * u_k; s_k is the largest |v_k| seen on the
// traces (normalize) or the largest over all axes (shared scale).

// [+ end, - end] image paths (null = text only). A target image shows at its + end only.
function axisEndImages(axis) {
  if (axis.kind === "direction") return [axis.image || null, axis.source_image || null];
  return [axis.image || null, null];
}

// Plot thumbnails, loaded once; the plot redraws when one arrives.
const plotThumbs = new Map();
function plotThumb(path) {
  if (!plotThumbs.has(path)) {
    const img = new Image();
    img.onload = drawPlot;
    img.src = thumbUrl(path, 96);
    plotThumbs.set(path, img);
  }
  const img = plotThumbs.get(path);
  return img.complete && img.naturalWidth ? img : null;
}

// Several models: axes are named by their model too
const severalModels = (axes) => new Set(axes.map((a) => a.model)).size > 1;
const axisName = (axis, axes) => (severalModels(axes) ? `${axis.model}: ` : "") + axis.label;

// Map labels are cut to this many characters (long prompts, file names); the table has them whole.
const LABEL_MAX = 28;
const shorten = (text) => (text.length > LABEL_MAX ? `${text.slice(0, LABEL_MAX - 1)}…` : text);

function axisEnds(axis, axes) {
  const w = axis.weight >= 0 ? `+${axis.weight}` : `${axis.weight}`;
  const label = shorten(axisName(axis, axes));
  if (axis.kind === "direction") return [`${label}  (w ${w})`, shorten(axis.source)];
  return [`${label}  (w ${w})`, `not: ${shorten(axis.label)}`];
}

// Trailing moving average over `n` points: each step's values are a mean over random
// augmentations, so the raw trace jitters. Display only; the table shows raw values.
function smoothed(points, n) {
  if (n <= 1 || points.length < 2) return points;
  const k = points[0].length;
  const sum = new Array(k).fill(0);
  return points.map((p, i) => {
    for (let j = 0; j < k; j++) sum[j] += p[j];
    if (i >= n) for (let j = 0; j < k; j++) sum[j] -= points[i - n][j];
    const count = Math.min(i + 1, n);
    return sum.map((s) => s / count);
  });
}

function displayTraces() {
  const n = Math.max(1, Math.round(Number($("#smooth").value) || 1));
  return { plus: smoothed(traces.plus, n), minus: smoothed(traces.minus, n) };
}

function axisScales(axes, shown) {
  const k = axes.length;
  const maxAbs = new Array(k).fill(0);
  for (const pole of ["plus", "minus"]) {
    for (const p of shown[pole]) for (let i = 0; i < k; i++) maxAbs[i] = Math.max(maxAbs[i], Math.abs(p[i] ?? 0));
  }
  if ($("#normalize").checked) return maxAbs.map((m) => Math.max(m, 1e-6));
  const shared = Math.max(...maxAbs, 1e-6);
  return maxAbs.map(() => shared);
}

function drawPlot() {
  const canvas = $("#plot");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.font = '12px "Courier New", monospace';

  const axes = lastState?.axes || [];
  const K = axes.length;
  if (!K) {
    ctx.fillStyle = "#888";
    ctx.fillText(lastState?.match_axes?.length ? "no map targets -- match terms only (see the table)"
                                                : "no targets yet -- start a run", 12, 20);
    return;
  }
  const cx = W / 2, cy = H / 2;
  const dirs = axes.map((_, k) => [Math.cos(Math.PI * k / K), -Math.sin(Math.PI * k / K)]);
  const shown = displayTraces();
  const scales = axisScales(axes, shown);
  const project = (v) => {
    let x = 0, y = 0;
    for (let k = 0; k < K; k++) {
      const t = (v[k] ?? 0) / scales[k];
      x += t * dirs[k][0];
      y += t * dirs[k][1];
    }
    return [x, y];
  };

  // Fit the traces (not the noisier view cloud) inside the plot.
  let rFit = 1;
  for (const pole of ["plus", "minus"]) {
    for (const p of shown[pole]) { const [x, y] = project(p); rFit = Math.max(rFit, Math.hypot(x, y)); }
  }
  const R = 0.40 * Math.min(W, H);
  const px = R / rFit;
  const toScreen = ([x, y]) => [cx + x * px, cy + y * px];

  // Axes, unit ticks (= the axis' scale value), end labels, push arrows.
  const reach = Math.hypot(W, H);
  const placed = [];  // label boxes {x, y (top), w, h} drawn so far, to step colliding labels towards the centre
  const THUMB = 44;
  const active = lastState?.status === "running" ? lastState.active_target : null;
  axes.forEach((axis, k) => {
    const [dx, dy] = dirs[k];
    ctx.strokeStyle = k === active ? "#000" : COLORS.axis;
    ctx.lineWidth = k === active ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(cx - dx * reach, cy - dy * reach);
    ctx.lineTo(cx + dx * reach, cy + dy * reach);
    ctx.stroke();

    for (const s of [-1, 1]) {
      const tx = cx + s * dx * px, ty = cy + s * dy * px;
      ctx.beginPath();
      ctx.moveTo(tx - dy * 4, ty + dx * 4);
      ctx.lineTo(tx + dy * 4, ty - dx * 4);
      ctx.stroke();
    }
    ctx.lineWidth = 1;
    ctx.fillStyle = "#888";
    ctx.fillText(scales[k].toFixed(3), cx + dx * px + 4, cy + dy * px - 4);

    // Labels where the axis leaves a box slightly inside the canvas.
    const [posLabel, negLabel] = axisEnds(axis, axes);
    const [posImage, negImage] = axisEndImages(axis);
    const pushSign = axis.weight >= 0 ? 1 : -1;  // the + pole is pushed towards this end
    for (const [s, label, image] of [[1, posLabel, posImage], [-1, negLabel, negImage]]) {
      const margin = 16;
      const t = Math.min(
        Math.abs(dx) > 1e-6 ? (W / 2 - margin) / Math.abs(dx) : Infinity,
        Math.abs(dy) > 1e-6 ? (H / 2 - margin) / Math.abs(dy) : Infinity);
      const ex = cx + s * dx * t, ey = cy + s * dy * t;
      ctx.fillStyle = COLORS.label;
      ctx.font = s === pushSign ? 'bold 12px "Courier New", monospace' : '12px "Courier New", monospace';
      const width = Math.max(ctx.measureText(label).width, image ? THUMB : 0);
      const lx = Math.min(Math.max(ex - width / 2, 4), W - width - 4);
      let ly = Math.min(Math.max(ey + (s * dy > 0 ? -6 : 14), 12), H - 4);
      const inward = ly < cy ? 1 : -1;
      // An image sits on the inward side of its text: block = [text, thumb] or [thumb, text].
      const thumbH = image ? THUMB + 4 : 0;
      const top = (y) => (inward > 0 ? y - 11 : y - 11 - thumbH);
      const h = 14 + thumbH;
      const hits = (y) => placed.some((b) => lx < b.x + b.w + 6 && b.x < lx + width + 6
                                             && top(y) < b.y + b.h && b.y < top(y) + h);
      for (let i = 0; i < 12 && hits(ly); i++) ly += 14 * inward;
      placed.push({ x: lx, y: top(ly), w: width, h });
      ctx.fillText(label, lx, ly);
      const thumb = image && plotThumb(image);
      if (thumb) {
        const ty = inward > 0 ? ly + 6 : ly - 11 - THUMB - 2;
        ctx.drawImage(thumb, lx, ty, THUMB, THUMB);
        ctx.strokeStyle = s === pushSign ? "#000" : COLORS.axis;
        ctx.strokeRect(lx + 0.5, ty + 0.5, THUMB - 1, THUMB - 1);
      }
      if (s === pushSign) {
        const ax = cx + s * dx * (t - 24), ay = cy + s * dy * (t - 24);
        arrowHead(ctx, ax, ay, s * dx, s * dy);
      }
    }
  });
  ctx.font = '12px "Courier New", monospace';

  // Origin = the start.
  ctx.strokeStyle = "#000";
  ctx.beginPath();
  ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
  ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
  ctx.stroke();

  // Current augmented views.
  if ($("#show-views").checked && lastState?.status === "running") {
    ctx.strokeStyle = COLORS.views;
    for (const v of lastState.views || []) {
      const [x, y] = toScreen(project(v));
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }

  // Traces.
  for (const pole of ["plus", "minus"]) {
    const pts = shown[pole];
    if (!pts.length) continue;
    ctx.strokeStyle = COLORS[pole];
    ctx.fillStyle = COLORS[pole];
    ctx.lineWidth = 1.5;
    ctx.setLineDash(pole === "minus" ? [5, 3] : []);
    ctx.beginPath();
    pts.forEach((p, i) => {
      const [x, y] = toScreen(project(p));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    const [x, y] = toScreen(project(pts[pts.length - 1]));
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText(pole === "plus" ? "+" : "−", x + 7, y - 7);
  }
}

function arrowHead(ctx, x, y, dx, dy) {
  const size = 8;
  ctx.beginPath();
  ctx.moveTo(x + dx * size, y + dy * size);
  ctx.lineTo(x - dy * size * 0.6, y + dx * size * 0.6);
  ctx.lineTo(x + dy * size * 0.6, y - dx * size * 0.6);
  ctx.closePath();
  ctx.fill();
}

// A row's history as a solid black area on white, downsampled to the graph's width.
const GRAPH_W = 120, GRAPH_H = 20;
function lossGraph(values) {
  const canvas = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  canvas.className = "loss-graph";
  canvas.width = GRAPH_W * dpr;
  canvas.height = GRAPH_H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const vals = values.filter((v) => v !== null && v !== undefined && Number.isFinite(v));
  if (vals.length < 2) return canvas;
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const n = vals.length, cols = Math.min(GRAPH_W, n);
  ctx.fillStyle = "#000";
  ctx.beginPath();
  ctx.moveTo(0, GRAPH_H);
  for (let c = 0; c < cols; c++) {
    const v = vals[Math.round(c * (n - 1) / Math.max(1, cols - 1))];
    ctx.lineTo(c * (GRAPH_W - 1) / Math.max(1, cols - 1), GRAPH_H - 1 - (v - lo) / span * (GRAPH_H - 2));
  }
  ctx.lineTo(GRAPH_W - 1, GRAPH_H);
  ctx.closePath();
  ctx.fill();
  return canvas;
}

function drawValues() {
  const axes = lastState?.axes || [];
  const terms = lastState?.terms || [];
  const table = $("#values");
  if (!axes.length && !terms.length) { table.replaceChildren(); return; }
  // the values of the pole running (or the last one run)
  const pole = lastState.pole === "minus" ? "minus" : "plus";
  const trace = traces[pole], termTrace = termTraces[pole];
  const last = trace[trace.length - 1], lastTerms = termTrace[termTrace.length - 1];
  const fmt = (v) => (v === undefined || v === null ? "" : (v >= 0 ? "+" : "") + v.toFixed(4));
  const graphs = $("#show-graphs").checked;
  const active = lastState.status === "running" ? lastState.active_target : null;
  const rows = [["loss", "kind", "weight", "value", ...(graphs ? ["history"] : [])]];
  const images = [null];  // per row: image paths shown before the name
  axes.forEach((axis, k) => {
    const name = axis.kind === "direction" ? `${axis.source} → ${axisName(axis, axes)}` : axisName(axis, axes);
    const kind = axis.kind === "direction" ? `${axis.target} direction` : axis.target;
    rows.push([name, kind, String(axis.weight), fmt(last?.[k]), ...(graphs ? [lossGraph(trace.map((p) => p[k]))] : [])]);
    images.push([axis.kind === "direction" ? axis.source_image : null, axis.image].filter(Boolean));
  });
  // match / regularization / field terms, the total last
  terms.forEach((term, j) => {
    rows.push([term.key, term.kind, term.weight === null ? "" : String(term.weight), fmt(lastTerms?.[j]),
               ...(graphs ? [lossGraph(termTrace.map((p) => p[j]))] : [])]);
    images.push(null);
  });
  table.replaceChildren(...rows.map((cells, i) => {
    const tr = document.createElement("tr");
    if (i > axes.length && terms[i - axes.length - 1]?.kind === "total") tr.className = "total";
    else if (i >= 1 && i - 1 === active) tr.className = "active";  // sequential mode: the target scored now
    cells.forEach((c, j) => {
      const td = document.createElement(i === 0 ? "th" : "td");
      if (j === 0 && images[i]?.length) {
        for (const path of images[i]) {
          const img = document.createElement("img");
          img.className = "axis-thumb";
          img.src = thumbUrl(path, 64);
          td.appendChild(img);
        }
      }
      td.append(c);
      tr.appendChild(td);
    });
    return tr;
  }));
}

// ── Init ───────────────────────────────────────────────────────────────────────

async function init() {
  options = await (await fetch("/api/options")).json();
  const fieldLabels = { GAUSSIANS: "GAUSSIAN SPLATS", REVOLUTION: "SURFACE OF REVOLUTION", MESH_TEXTURE: "MESH TEXTURE",
                        GAUSSIANS_3D: "3D GAUSSIAN SPLATS", SOLIDS_ON_OFF: "SOLIDS ON/OFF" };
  const fieldOption = (f) => new Option(fieldLabels[f] ?? f, f);
  // The 3D fields in a group of their own, after a separator (an <hr> in a <select>: where the
  // browser doesn't draw it, the group label still sets them apart).
  const threeD = Object.assign(document.createElement("optgroup"), { label: "3D" });
  threeD.append(...options.field_types.filter((f) => FIELDS_3D.includes(f)).map(fieldOption));
  $("#field-type").replaceChildren(...options.field_types.filter((f) => !FIELDS_3D.includes(f)).map(fieldOption),
    document.createElement("hr"), threeD);
  $("#model").replaceChildren(...options.models.map((m) => new Option(m, m)));
  const familyNames = { clip: "CLIP", imagenet: "ImageNet", dino: "DINOv2", vlm: "VLM (vision-language)",
                        aesthetic: "Aesthetic Scoring", custom: "Custom" };
  const kindNames = { classifier: "classifiers", autoencoder: "autoencoders", ranknet: "RankNets" };
  $("#aesthetic-model").replaceChildren(...options.custom_models.filter((m) => m.kind === "aesthetic")
    .map((m) => new Option(m.label, m.id)));
  $("#custom-model").replaceChildren(...Object.entries(kindNames).map(([kind, title]) => {
    const group = document.createElement("optgroup");
    group.label = title;
    group.append(...options.custom_models.filter((m) => m.kind === kind).map((m) => new Option(m.label, m.id)));
    return group;
  }).filter((g) => g.children.length));
  $("#sheet-score").replaceChildren(...options.imagenet_scores.map((s) => new Option(s, s)));
  $("#sheet-precision").replaceChildren(...options.custom_precisions.map((p) => new Option(p, p)));
  $("#sheet-goal").replaceChildren(...options.score_goals.map((g) => new Option(g === "value" ? "to a value" : g, g)));
  $("#sheet-tokens").replaceChildren(...options.dino_tokens.map((t) => new Option(t === "cls" ? "CLS token" : "patch tokens", t)));
  $("#model-family").replaceChildren(...options.model_families.map((f) => new Option(familyNames[f] || f, f)));
  $("#dino-model").replaceChildren(...options.dino_models.map((m) => new Option(m.split("~")[1], m)));
  $("#imagenet-model").replaceChildren(...options.imagenet_models.map((m) => new Option(m.split("~").slice(1).join(" "), m)));
  $("#vlm-model").replaceChildren(...options.vlm_models.map((m) => new Option(m.split("~")[1], m)));
  $("#start-image-fit").replaceChildren(...options.image_fits.map((f) => new Option(f, f)));
  $("#canvas-init").replaceChildren(...options.canvas_inits.map((i) => new Option(i === "start" ? "start image" : i, i)));
  $("#symmetry-axis").replaceChildren(...options.symmetry_axes.map((a) => new Option(a, a)));
  $("#edge-operator").replaceChildren(...options.edge_operators.map((e) => new Option(e, e)));
  $("#distortion-offset-mode").replaceChildren(...options.distortion_offset_modes.map((m) => new Option(m === "resolution" ? "resolution dependent" : m, m)));
  $("#distortion-edge-mode").replaceChildren(...options.distortion_edge_modes.map((m) => new Option(m, m)));
  $("#distortion-offset-mode").addEventListener("change", syncDistortionControls);
  $("#prevent-folds").addEventListener("change", syncDistortionControls);
  $("#patch-randomize").addEventListener("change", syncPatchControls);
  $("#patch-source").addEventListener("change", syncPatchControls);
  $("#shape-mode").addEventListener("change", syncShapeControls);
  $("#patch-image-add").onclick = () => pickImageFile((path) => {
    patchImages.push({ path, count: 4 });
    renderPatchImages();
    saveLocal();
  });
  for (const sel of ["#conv-inplace", '[data-key="conv_layers"]', '[data-key="conv_input_res"]']) {
    for (const el of document.querySelectorAll(sel)) el.addEventListener("input", syncConvControls);
  }
  $("#sequential-targets").addEventListener("change", syncSequentialControls);
  $("#depth-normalization").replaceChildren(...options.depth_normalizations.map((n) => new Option(n, n)));
  $("#vae-model").replaceChildren(...options.vae_models.map((m) => Object.assign(new Option(m.id, m.id), { title: m.repo })));
  $("#conv-norm").replaceChildren(...options.conv_norms.map((n) => new Option(n, n)));
  $("#conv-activation").replaceChildren(...options.conv_activations.map((a) => new Option(a, a)));
  $("#vae-precision").replaceChildren(...options.vae_precisions.map((p) => new Option(p, p)));
  $("#pattern-select").replaceChildren(new Option("(path below)", ""),
    ...options.patterns.map((p) => new Option(p.name, p.path)));
  $("#device").replaceChildren(...options.devices.map((d) => new Option(d.label, d.id)));
  setupSurface();
  setupBackgroundImage();
  setupLevelset();
  setupFunctionStack();
  setupRevolution();
  setupMeshTexture();
  setupGaussians3D();
  setupVolumes();
  setupMatchTargets();
  setupInspector();
  setupAllViews();
  syncRunControls("idle", false);
  for (const b of document.querySelectorAll(".tabs [data-tab]")) b.onclick = () => showTab(b.dataset.tab);
  let tab = null;
  try { tab = localStorage.getItem(TAB_KEY); } catch { /* storage unavailable */ }
  showTab(tab);

  // A running/finished server config wins, then this browser's last config, then the script defaults.
  writeForm(options.run_id ? options.config : (loadLocal() || options.defaults));
  runId = options.run_id || "";

  onPress($("#start"), startRun);
  onPress($("#stop"), () => fetch("/api/stop", { method: "POST" }));
  onPress($("#pause"), togglePause);
  $("#snapshot").onclick = saveSnapshot;
  $("#config-save").onclick = saveConfig;
  $("#config-load").onclick = () => $("#config-file").click();
  $("#config-file").onchange = (e) => { if (e.target.files[0]) loadConfig(e.target.files[0]); e.target.value = ""; };
  try { $("#embed-files").checked = localStorage.getItem(EMBED_KEY) !== "false"; } catch { /* storage unavailable */ }
  $("#embed-files").addEventListener("change", (e) => {
    e.stopPropagation();  // a preference of this browser, not part of the config
    try { localStorage.setItem(EMBED_KEY, String(e.target.checked)); } catch { /* storage unavailable */ }
  });
  $("#reset-defaults").onclick = () => { writeForm(options.defaults); saveLocal(); $("#preset-note").textContent = ""; };
  for (const name of ["editing", "generation"]) {
    $(`#preset-${name}`).onclick = () => applyPreset(name);
  }
  $("#add-model").onclick = addModel;
  // +img / +text (class, question) / Δimg / Δtext / Δ yes/no (a VLM's No -> Yes): a row of the selected model
  for (const button of document.querySelectorAll("#model-sheet button[data-add]")) {
    const kind = button.dataset.add;
    if ("yesNo" in button.dataset) {
      button.onclick = () => $("#sheet-targets").appendChild(targetRow("direction",
        [{ question: "", answer: "No" }, { question: "", answer: "Yes" }, 1], modelInfo(models[selected])));
      continue;
    }
    const end = "image" in button.dataset ? { image: "" } : "";
    button.onclick = () => {
      $("#sheet-targets").appendChild(targetRow(kind, kind === "direction" ? [end, end, 1] : [end, 1], modelInfo(models[selected])));
    };
  }
  // edits in the sheet: into the entry, its summary in the list
  $("#model-sheet").addEventListener("input", () => { commitSheet(); renderModelList(); });
  $("#model-sheet").addEventListener("change", () => {
    $("#sheet-value-label").hidden = $("#sheet-goal").value !== "value";
    syncTemplatesEnabled();
    commitSheet();
    renderModelList();
    if (models[selected]) $("#sheet-title").textContent = modelInfo(models[selected]).label;
  });
  $("#model-family").addEventListener("change", updateFamilyVisibility);
  $("#field-type").addEventListener("change", updateFieldVisibility);
  $("#start-image").addEventListener("change", updateStartThumb);
  $("#start-image-fit").addEventListener("change", updateStartThumb);
  $("#shape-src").addEventListener("change", () => { syncPatternSelect(); updatePatternThumb(); });
  for (const sel of ['[data-key="canvas"]', '[data-key="shape_border"]']) {
    for (const el of document.querySelectorAll(sel)) el.addEventListener("change", updatePatternThumb);
  }
  for (const el of document.querySelectorAll('[data-key="canvas"]')) el.addEventListener("change", updateStartThumb);
  $("#pattern-select").addEventListener("change", (e) => {
    if (e.target.value) $("#shape-src").value = e.target.value;
    updatePatternThumb();
    saveLocal();
  });
  $("#start-image-upload").onclick = () => $("#start-image-file").click();
  $("#start-image-file").onchange = async (e) => {
    if (!e.target.files[0]) return;
    try {
      $("#start-image").value = await uploadFile(e.target.files[0]);
      updateStartThumb();
      saveLocal();
    } catch (err) { setStatus(err.message, true); }
    e.target.value = "";
  };
  $("#start-image-random").onclick = async () => {
    const path = $("#start-image").value.trim();
    const res = await fetch(`/api/random_image?path=${encodeURIComponent(path)}`);
    const data = await res.json();
    if (!res.ok) { setStatus(data.error, true); return; }
    $("#start-image").value = data.path;
    updateStartThumb();
    saveLocal();
  };
  $("#pattern-upload").onclick = () => $("#pattern-file").click();
  $("#pattern-file").onchange = async (e) => {
    if (!e.target.files[0]) return;
    try {
      const path = await uploadFile(e.target.files[0]);
      const select = $("#pattern-select");
      if (![...select.options].some((o) => o.value === path)) select.appendChild(new Option(path.split(/[\\/]/).pop(), path));
      $("#shape-src").value = path;
      syncPatternSelect();
      updatePatternThumb();
      saveLocal();
    } catch (err) { setStatus(err.message, true); }
    e.target.value = "";
  };
  $("#sidebar").addEventListener("change", saveLocal);
  $("#inspector").addEventListener("change", saveLocal);
  $("#normalize").addEventListener("change", drawPlot);
  $("#show-views").addEventListener("change", drawPlot);
  $("#smooth").addEventListener("input", drawPlot);
  $("#show-graphs").addEventListener("change", drawValues);
  // The plot's height comes from the layout (it takes what the images leave), so redraw on any size change.
  new ResizeObserver(drawPlot).observe($("#plot"));
  poll();
}

init();
