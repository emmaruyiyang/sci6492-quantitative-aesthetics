// Dataset tab: make a new dataset from one, every image converted (grayscale,
// Canny edges, Depth Anything depth, depth edges). A few images are converted
// as a preview whenever a setting changes; "Convert" converts all of them in
// the background (GET /api/convert/status about every second).
//
// Uses the helpers of embedding_map.js and training.js, loaded before this file.

const data = {
  options: null,            // /api/convert/options
  nameEdited: false,        // the name was typed: stop suggesting one
  previewOffset: 0,         // "Other images" shows the next few
  previewRun: 0,
  timer: null,
  lastState: null,
};

const PARAM_LABELS = {
  blur: "blur radius (px)", low: "low threshold", high: "high threshold",
  invert: "invert (black on white)", depth_model: "depth model",
};
const PARAM_STEPS = { blur: 1, low: 5, high: 5 };

const CONVERSION_HINTS = {
  grayscale: "Color removed, brightness kept.",
  canny: "White lines where the brightness changes sharply. The image is blurred first so texture and noise "
    + "don't become edges. A change above 'high' starts an edge, one above 'low' continues it: lower = more edges.",
  depth: "Depth Anything V2 guesses how far away every pixel is from the image alone: near = bright, far = dark "
    + "(relative, scaled per image). Small is fast, large is more detailed. The first use downloads / loads the model.",
  depth_edges: "Canny edges of the depth image: the outlines of shapes in front of others, without the edges of "
    + "their texture or color. Depth images are smooth, so the thresholds are lower than for Canny on the image.",
};

async function dataViewShown() {
  if (!data.options) await loadDataOptions();
  pollDataSoon(0);
}

async function loadDataOptions() {
  try {
    data.options = await api("/api/convert/options");
  } catch (e) {
    $("d-form-status").textContent = e.message;
    return;
  }
  const o = data.options;
  fillDatasetSelect($("d-source"), o.datasets, $("d-source").value || undefined);
  if (!$("d-conversion").options.length) {
    fillSelect($("d-conversion"), Object.entries(o.conversions).map(([k, c]) => [k, c.label]), "depth");
    conversionChanged();
  }
  sourceChanged();
}

function sourceChanged() {
  const d = data.options.datasets.find((x) => x.id === $("d-source").value);
  $("d-source-info").textContent = d
    ? `${d.count} images${d.recursive ? ", subfolders kept as subfolders" : ""}` : "no image folders found";
  data.previewOffset = 0;
  suggestName();
  schedulePreview();
}

// One input per parameter of the conversion, with its default value
function conversionChanged() {
  const c = data.options.conversions[$("d-conversion").value];
  const box = $("d-params");
  box.innerHTML = "";
  for (const [key, value] of Object.entries(c.params)) {
    const label = document.createElement("label");
    let input;
    if (key === "depth_model") {
      label.className = "pick";
      input = document.createElement("select");
      data.options.depth_models.forEach((m) => input.add(new Option(m, m)));
      input.value = value;
      label.append(PARAM_LABELS[key], " ", input);
    } else if (typeof value === "boolean") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = value;
      label.append(input, ` ${PARAM_LABELS[key] || key}`);
    } else {
      label.className = "pick";
      input = document.createElement("input");
      input.type = "number";
      input.min = 0;
      input.step = PARAM_STEPS[key] || "any";
      input.value = value;
      label.append(PARAM_LABELS[key] || key, " ", input);
    }
    input.dataset.param = key;
    input.addEventListener("change", schedulePreview);
    box.appendChild(label);
  }
  $("d-conversion-hint").textContent = CONVERSION_HINTS[$("d-conversion").value] || "";
  suggestName();
  schedulePreview();
}

function conversionParams() {
  const params = {};
  for (const input of $("d-params").querySelectorAll("[data-param]")) {
    params[input.dataset.param] = input.type === "checkbox" ? input.checked
      : input.tagName === "SELECT" ? input.value : Number(input.value);
  }
  return params;
}

function dataSettings() {
  return {
    dataset: $("d-source").value,
    conversion: $("d-conversion").value,
    params: conversionParams(),
    name: $("d-name").value.trim(),
    format: $("d-format").value,
    max_side: Number($("d-max-side").value),
  };
}

// source folder + conversion, e.g. "Chairs_depth", until a name is typed
function suggestName() {
  if (data.nameEdited || !data.options) return;
  const d = data.options.datasets.find((x) => x.id === $("d-source").value);
  if (!d) return;
  const base = d.folder === "." ? "datasets" : d.folder.split("/").pop();
  $("d-name").value = `${base}_${$("d-conversion").value}`;
  checkDataName();
}

let dataNameTimer = null;
function checkDataName() {
  clearTimeout(dataNameTimer);
  dataNameTimer = setTimeout(async () => {
    const name = $("d-name").value.trim();
    const info = $("d-name-info");
    info.classList.remove("warn");
    if (!name) {
      info.textContent = "the name of the new folder";
      return;
    }
    try {
      const r = await api(`/api/convert/check?${query({ name })}`);
      if (r.exists && r.files) {
        info.textContent = `${name} exists (${r.files} files): converting writes into it, replacing files with the same names.`;
        info.classList.add("warn");
      } else {
        info.textContent = `new folder: ${r.folder}`;
      }
    } catch (e) {
      info.textContent = e.message.replace(/^.*?: \d+ /, "");
      info.classList.add("warn");
    }
  }, 250);
}

// ---------------------------------------------------------------------------
// Preview
// ---------------------------------------------------------------------------

let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(preview, 400);
}

async function preview() {
  const s = dataSettings();
  if (!s.dataset || !s.conversion || $("d-converter").hidden) return;
  const run = ++data.previewRun;
  $("d-preview-info").textContent = s.conversion.startsWith("depth")
    ? "converting… (the depth model may take a few seconds to load the first time)" : "converting…";
  try {
    const r = await postJSON("/api/convert/preview", { ...s, offset: data.previewOffset });
    if (run !== data.previewRun) return;
    $("d-preview-info").textContent = `before | after, ${r.images.length} images in ${r.seconds}s (not saved)`;
    $("d-preview").innerHTML = r.images.map((im) => `
      <div class="convert-pair">
        <div class="images"><img src="${im.before}" alt="before"><img src="${im.after}" alt="after"></div>
        <div class="name">${escapeHTML(im.name)}</div>
      </div>`).join("");
  } catch (e) {
    if (run === data.previewRun) $("d-preview-info").textContent = e.message;
  }
}

// ---------------------------------------------------------------------------
// Converting
// ---------------------------------------------------------------------------

async function startConversion(overwrite = false) {
  const s = { ...dataSettings(), overwrite };
  if (!s.name) {
    $("d-form-status").textContent = "give the new dataset a name";
    $("d-name").focus();
    return;
  }
  $("d-form-status").textContent = "starting…";
  const response = await fetch("/api/convert/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(s),
  });
  const body = await response.text();
  if (response.status === 409) {
    let r = {};
    try { r = JSON.parse(body); } catch { /* plain text */ }
    if (r.exists) {
      if (confirm(`${r.error}.\n\nWrite into it anyway (files with the same names are replaced)?`)) return startConversion(true);
      $("d-form-status").textContent = "not started";
      return;
    }
  }
  if (!response.ok) {
    $("d-form-status").textContent = body;
    return;
  }
  $("d-form-status").textContent = "";
  pollDataSoon(0);
}

function pollDataSoon(delay) {
  clearTimeout(data.timer);
  data.timer = setTimeout(pollData, delay);
}

async function pollData() {
  const visible = !$("view-data").hidden;
  let st = null;
  try {
    st = await api("/api/convert/status");
    if (visible) renderConversion(st);
    await noticeConverted(st);
  } catch (e) {
    if (visible) $("d-form-status").textContent = e.message;
  }
  const active = st && ["running", "stopping"].includes(st.state);
  if (visible) pollDataSoon(1000);
  else if (active) pollDataSoon(4000);
}

// A conversion ended: the new dataset is in every dataset list now
async function noticeConverted(st) {
  const ended = !["running", "stopping", "idle"].includes(st.state);
  if (ended && data.lastState && ["running", "stopping"].includes(data.lastState)) {
    try {
      applyOptions(await api("/api/options"), settings());
      if (train.options) await loadTrainOptions();
      await loadDataOptions();
    } catch { /* "Rescan folders" catches up */ }
  }
  data.lastState = st.state;
}

function renderConversion(st) {
  $("d-job").hidden = st.state === "idle";
  const active = ["running", "stopping"].includes(st.state);
  $("d-start").disabled = active;
  $("a-start").disabled = active;
  if (st.state === "idle") return;
  $("d-job-title").textContent = st.conversion;
  $("d-job-target").textContent = `→ ${st.target}`;
  $("d-job-state").textContent = st.state;
  $("d-job-state").className = `badge ${st.state}`;
  $("d-stop").disabled = !active || st.state === "stopping";
  $("d-open").hidden = active || !st.done;
  $("d-job-count").textContent = `${st.done} / ${st.total} images` + (st.failed ? `, ${st.failed} unreadable` : "");
  $("d-job-time").textContent = `${formatSeconds(st.elapsed)}${st.eta ? `, about ${formatSeconds(st.eta)} left` : ""}`;
  $("d-job-fill").style.width = `${(100 * (st.done + st.failed) / Math.max(1, st.total)).toFixed(1)}%`;
  $("d-job-message").hidden = !st.message;
  $("d-job-message").textContent = st.message;
  const log = $("d-log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = st.log.join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
  data.target = st.target;
}

// Show the new dataset on the map: "all of name/" if it has subfolders, else the folder itself
function openConverted() {
  const root = data.options.datasets_folder.replace(/[\\/]+$/, "");
  const folder = data.target.slice(root.length + 1).replaceAll("\\", "/");
  const list = state.options ? state.options.datasets : [];
  const pick = list.find((d) => d.folder === folder && d.recursive) || list.find((d) => d.folder === folder);
  if (!pick) {
    $("d-form-status").textContent = `${folder} is not in the dataset list yet, try "Rescan folders"`;
    return;
  }
  $("dataset").value = pick.id;
  showView("map");
  updateControls();
  scheduleRun();
}

// ---------------------------------------------------------------------------
// Audio: a sound file -> spectrogram images + their sound fragments
// ---------------------------------------------------------------------------

const audio = { token: null, name: "", duration: 0, nameEdited: false, offset: 0, previewRun: 0 };

const NORMALIZATION_HINTS = {
  fragment: "Each spectrogram uses its own loudest point as white: quiet fragments look as bright as loud ones "
    + "(compares the shape of the sounds).",
  clip: "The loudest point of the whole sound is white in every spectrogram: quiet fragments are darker "
    + "(keeps the loudness differences).",
};

function setDataMode(mode) {
  document.querySelectorAll("#d-mode button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  const isAudio = mode === "audio";
  $("d-converter").hidden = isAudio;
  $("d-audio").hidden = !isAudio;
  $("d-preview").hidden = isAudio;
  $("a-preview").hidden = !isAudio;
  $("d-other").hidden = isAudio;
  $("a-other").hidden = !isAudio;
  if (isAudio) {
    $("d-preview-info").textContent = audio.token ? "" : "choose a sound file on the left";
    audioSettingsChanged();
  } else {
    preview();
  }
}

document.querySelectorAll("#d-mode button").forEach((b) => b.addEventListener("click", () => setDataMode(b.dataset.mode)));

function audioSettings() {
  return {
    duration: Number($("a-duration").value),
    overlap: Number($("a-overlap").value),
    sample_rate: Number($("a-sample-rate").value),
    resolution: Number($("a-resolution").value),
    scale: $("a-scale").value,
    normalization: $("a-normalization").value,
    dynamic_range: Number($("a-dynamic").value),
    audio_format: $("a-format").value,
  };
}

async function uploadAudio() {
  const file = $("a-file").files[0];
  if (!file) return;
  audio.token = null;
  $("a-file-info").textContent = `uploading ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    const r = await api("/api/audio/upload", { method: "POST", body: form });
    Object.assign(audio, { token: r.token, name: r.name, duration: r.duration, offset: 0 });
    $("a-file-info").textContent = `${r.name}: ${r.duration.toFixed(1)} s`;
    if (!audio.nameEdited) {
      $("a-name").value = `${r.name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_.-]+/g, "_")}_spectrograms`;
      checkAudioName();
    }
    audioSettingsChanged();
  } catch (e) {
    $("a-file-info").textContent = e.message.replace(/^.*?: \d+ /, "");
  }
}

function audioSettingsChanged() {
  const s = audioSettings();
  $("a-normalization-hint").textContent = NORMALIZATION_HINTS[s.normalization];
  if (audio.duration) {
    // the same count as the server: whole fragments only
    const step = s.duration * (1 - s.overlap / 100);
    const count = audio.duration >= s.duration ? Math.floor((audio.duration - s.duration) / step + 1e-9) + 1 : 0;
    $("a-fragments-info").textContent = `${count} fragments of ${s.duration} s, a new one every ${step.toFixed(2)} s`;
  }
  scheduleAudioPreview();
}

let audioPreviewTimer = null;
function scheduleAudioPreview() {
  clearTimeout(audioPreviewTimer);
  audioPreviewTimer = setTimeout(previewAudio, 350);
}

async function previewAudio() {
  if (!audio.token || $("d-audio").hidden) return;
  const run = ++audio.previewRun;
  $("d-preview-info").textContent = "making spectrograms…";
  try {
    const r = await postJSON("/api/audio/preview", { token: audio.token, settings: audioSettings(), offset: audio.offset });
    if (run !== audio.previewRun) return;
    $("d-preview-info").textContent = `${r.previews.length} of ${r.fragments} fragments (not saved). `
      + "Click a spectrogram to hear its fragment.";
    $("a-preview").innerHTML = r.previews.map((p) => `
      <div class="audio-card">
        <img src="${p.image}" alt="spectrogram of fragment ${p.index}">
        <audio controls preload="auto" src="${p.sound}"></audio>
        <div class="name">fragment ${p.index}: ${p.start.toFixed(2)} – ${p.end.toFixed(2)} s</div>
      </div>`).join("");
    $("a-preview").querySelectorAll(".audio-card").forEach((card) => {
      card.querySelector("img").addEventListener("click", () => {
        const sound = card.querySelector("audio");
        sound.currentTime = 0;
        sound.play();
      });
    });
  } catch (e) {
    if (run === audio.previewRun) $("d-preview-info").textContent = e.message.replace(/^.*?: \d+ /, "");
  }
}

let audioNameTimer = null;
function checkAudioName() {
  clearTimeout(audioNameTimer);
  audioNameTimer = setTimeout(async () => {
    const name = $("a-name").value.trim();
    const info = $("a-name-info");
    info.classList.remove("warn");
    if (!name) {
      info.textContent = "the name of the new folder";
      return;
    }
    try {
      const r = await api(`/api/convert/check?${query({ name })}`);
      info.textContent = r.exists && r.files
        ? `${name} exists (${r.files} files): fragments with the same names are replaced.`
        : `new folder: ${r.folder}`;
      info.classList.toggle("warn", Boolean(r.exists && r.files));
    } catch (e) {
      info.textContent = e.message.replace(/^.*?: \d+ /, "");
      info.classList.add("warn");
    }
  }, 250);
}

async function startAudio(overwrite = false) {
  if (!audio.token) {
    $("a-form-status").textContent = "choose a sound file first";
    return;
  }
  const name = $("a-name").value.trim();
  if (!name) {
    $("a-form-status").textContent = "give the new dataset a name";
    $("a-name").focus();
    return;
  }
  $("a-form-status").textContent = "starting…";
  const response = await fetch("/api/audio/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: audio.token, name, settings: audioSettings(), overwrite }),
  });
  const body = await response.text();
  if (response.status === 409) {
    let r = {};
    try { r = JSON.parse(body); } catch { /* plain text */ }
    if (r.exists) {
      if (confirm(`${r.error}.\n\nWrite into it anyway (files with the same names are replaced)?`)) return startAudio(true);
      $("a-form-status").textContent = "not started";
      return;
    }
  }
  $("a-form-status").textContent = response.ok ? "" : body;
  if (response.ok) pollDataSoon(0);
}

$("a-file").addEventListener("change", uploadAudio);
for (const id of ["a-duration", "a-overlap", "a-sample-rate", "a-resolution", "a-scale", "a-normalization", "a-dynamic"]) {
  $(id).addEventListener("change", audioSettingsChanged);
}
$("a-name").addEventListener("input", () => { audio.nameEdited = true; checkAudioName(); });
$("a-other").addEventListener("click", () => { audio.offset++; previewAudio(); });
$("a-start").addEventListener("click", () => startAudio());

$("d-source").addEventListener("change", sourceChanged);
$("d-conversion").addEventListener("change", conversionChanged);
$("d-max-side").addEventListener("change", schedulePreview);
$("d-name").addEventListener("input", () => { data.nameEdited = true; checkDataName(); });
$("d-other").addEventListener("click", () => { data.previewOffset++; preview(); });
$("d-start").addEventListener("click", () => startConversion());
$("d-stop").addEventListener("click", () => postJSON("/api/convert/stop", {}).then(() => pollDataSoon(0)));
$("d-open").addEventListener("click", openConverted);
