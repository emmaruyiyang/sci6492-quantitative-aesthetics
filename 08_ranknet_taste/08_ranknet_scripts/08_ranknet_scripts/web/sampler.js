// Dataset tab, Sampler: a small, well spread dataset out of a large one
// (dataset_sampler.py): the images clustered by a model's vectors, a few
// taken from every cluster. Runs in the background like a conversion, in the
// same monitor (GET /api/convert/status, dataset.js).
//
// Uses the helpers of embedding_map.js, training.js and dataset.js, loaded before this file.

const sampler = { options: null, nameEdited: false };

async function samplerShown() {
  try {
    sampler.options = await api("/api/sample/options");
  } catch (e) {
    $("s-form-status").textContent = e.message;
    return;
  }
  const o = sampler.options;
  fillDatasetSelect($("s-source"), o.datasets, $("s-source").value || undefined);
  const models = o.models.map((m) => [m.id, `${m.label}  (${m.kind})`]);
  fillSelect($("s-model"), models, $("s-model").value || o.default_model);
  if (!$("s-group-by").options.length) fillSelect($("s-group-by"), Object.entries(o.groupings), o.defaults.group_by);
  samplerSourceChanged();
  pollDataSoon(0);
}

function samplerSourceChanged() {
  const d = sampler.options.datasets.find((x) => x.id === $("s-source").value);
  $("s-source-info").textContent = d ? `${d.count} images${d.recursive ? " (subfolders included)" : ""}` : "no image folders found";
  suggestSamplerName();
  samplerSettingsChanged();
}

function samplerSettingsChanged() {
  const d = sampler.options && sampler.options.datasets.find((x) => x.id === $("s-source").value);
  const count = Number($("s-count").value), clusters = Number($("s-clusters").value);
  const per = count / Math.max(1, clusters);
  $("s-ratio").textContent = (d && count > d.count ? `the dataset has only ${d.count} images. ` : "")
    + `About ${per.toFixed(1)} images per cluster: `
    + (per < 2 ? "almost every pair is between different regions (only the coarse differences)."
      : per <= 6 ? "balanced: coarse differences and pairs of alike images."
        : "many alike images: fine distinctions within a narrow domain.");
  $("s-group-cap-row").hidden = $("s-group-by").value === "none";
}

function suggestSamplerName() {
  if (sampler.nameEdited || !sampler.options) return;
  const d = sampler.options.datasets.find((x) => x.id === $("s-source").value);
  if (!d) return;
  const base = d.folder === "." ? "datasets" : d.folder.split("/").pop();
  $("s-name").value = `${base}_sample_${$("s-count").value}`;
  checkSamplerName();
}

let samplerNameTimer = null;
function checkSamplerName() {
  clearTimeout(samplerNameTimer);
  samplerNameTimer = setTimeout(async () => {
    const name = $("s-name").value.trim();
    const info = $("s-name-info");
    info.classList.remove("warn");
    if (!name) {
      info.textContent = "the name of the new folder";
      return;
    }
    try {
      const r = await api(`/api/convert/check?${query({ name })}`);
      info.textContent = r.exists && r.files ? `${name} exists (${r.files} files): sampling writes into it.` : `new folder: ${r.folder}`;
      info.classList.toggle("warn", Boolean(r.exists && r.files));
    } catch (e) {
      info.textContent = e.message.replace(/^.*?: \d+ /, "");
      info.classList.add("warn");
    }
  }, 250);
}

function samplerSettings() {
  return {
    dataset: $("s-source").value,
    model: $("s-model").value,
    name: $("s-name").value.trim(),
    params: {
      count: Number($("s-count").value),
      clusters: Number($("s-clusters").value),
      dedup: Number($("s-dedup").value),
      group_by: $("s-group-by").value,
      group_cap: $("s-group-by").value === "none" ? 0 : Number($("s-group-cap").value),
      max_side: Number($("s-max-side").value),
      max_short_side: Number($("s-max-short").value),
      seed: Number($("s-seed").value),
    },
  };
}

async function startSampler(overwrite = false) {
  const s = { ...samplerSettings(), overwrite };
  if (!s.name) {
    $("s-form-status").textContent = "give the new dataset a name";
    $("s-name").focus();
    return;
  }
  $("s-form-status").textContent = "starting…";
  const response = await fetch("/api/sample/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(s),
  });
  const body = await response.text();
  if (response.status === 409) {
    let data = {};
    try { data = JSON.parse(body); } catch { /* plain text error */ }
    if (data.exists && confirm(`${data.error}.\n\nWrite into it anyway?`)) return startSampler(true);
    $("s-form-status").textContent = data.exists ? "not started" : body;
    return;
  }
  if (!response.ok) {
    $("s-form-status").textContent = body;
    return;
  }
  $("s-form-status").textContent = "";
  pollDataSoon(0);
}

$("s-source").addEventListener("change", samplerSourceChanged);
for (const id of ["s-count", "s-clusters", "s-group-by"]) {
  $(id).addEventListener("input", () => { samplerSettingsChanged(); if (id === "s-count") suggestSamplerName(); });
}
$("s-name").addEventListener("input", () => { sampler.nameEdited = true; checkSamplerName(); });
$("s-start").addEventListener("click", () => startSampler());
