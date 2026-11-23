// Dataset tab, Ranking: two images of a dataset at a time, which one is
// better. The answers are kept here and saved (POST /api/ranking/save) every
// few answers, when closing and when leaving the page, into a ranking file of
// the dataset's folder (ranking_<name>.json, see ranknet/rankings.py). A
// RankNet trains on such a file (Training tab).
//
// Pairs are stored as [a, b, pref] with a < b (indices of the dataset's
// images), pref about a: 1 = a better, 0 = about the same, -1 = b better.
// Which pair comes next ("choose"), always one not answered yet:
//
//   random     any pair. Keeps the scale of the whole dataset right.
//   similar    two images of the same cluster ("alike by": the Sampler's
//              clusters of the dataset, or k-means on a model's vectors):
//              "these are alike, but I prefer this one", where the fine
//              distinctions are.
//   ranking    two images close in a ranking ("ranking by": a RankNet or an
//              aesthetic score): the pairs it is least sure about, where an
//              answer teaches it most.
//   mixed      half random, half similar / ranking (whichever is set). Only
//              close-in-ranking pairs would end up on images that really are
//              equal: coin flips.
//
// And now and then (a few % of the time) a pair answered a while ago, on
// random sides: the two answers measure how consistent the ranking is (the
// "repeats" of the file). From DiffDrawing's taste page (frontend/src/taste.js).
//
// Uses the helpers of embedding_map.js, training.js and dataset.js, loaded before this file.

const NEW_RANKING = "__new__";
const RANK_IMAGE_SIZE = 768;
const AUTOSAVE_EVERY = 10;      // answers

const rank = {
  open: false,
  dataset: null, file: null, label: "", question: "", created: null, count: 0,
  pairs: [],                // [a, b, pref], a < b
  byKey: new Map(),         // "a:b" -> index into pairs
  repeats: [],              // [a, b, first answer, second answer]
  history: [],              // "pair" | "repeat": what undo takes back
  skipped: new Set(),       // "a:b" waved away this session (not saved)
  sinceRepeat: 0,
  current: null,            // {pair: [left, right], repeat: null | {a, b, first}} on screen
  unsaved: 0,
  saving: false,
  manifestClusters: null,   // the Sampler's cluster of every image, if the dataset was made by it
  byCluster: null,          // Map cluster -> image indices (2 or more): "similar" pairs
  rankOrder: null,          // image indices, best first: "ranking" pairs
  helperRun: 0,             // number of the latest helper request (older answers are ignored)
};
const helpers = new Map();  // "dataset|kind|model|prompt" -> clusters or scores, computed once

const pairKey = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);

// ---------------------------------------------------------------------------
// Form
// ---------------------------------------------------------------------------

async function rankingShown() {
  if (!data.options) await loadDataOptions();
  if (!data.options) return;
  fillStrategyModels();
  fillDatasetSelect($("r-dataset"), data.options.datasets, $("r-dataset").value || undefined);
  await loadRankingFiles();
}

// leaving the Ranking mode: keep what was answered
function rankingHidden() {
  if (rank.open && rank.unsaved) saveRanking();
}

async function loadRankingFiles(select) {
  const dataset = $("r-dataset").value;
  if (!dataset) {
    $("r-dataset-info").textContent = "no image folders found";
    return;
  }
  let list;
  try {
    list = await api(`/api/ranking/list?${query({ dataset })}`);
  } catch (e) {
    $("r-form-status").textContent = e.message;
    return;
  }
  rank.files = list.rankings;
  const d = data.options.datasets.find((x) => x.id === dataset);
  $("r-dataset-info").textContent = `${list.count} images${d && d.recursive ? " (subfolders included)" : ""}, `
    + `${list.rankings.length || "no"} ranking file${list.rankings.length === 1 ? "" : "s"} in ${list.folder}`;
  fillSelect($("r-file"), [...list.rankings.map((r) => [r.file, `${r.label} (${r.error ? "unreadable" : r.pairs + " pairs"})`]),
    [NEW_RANKING, "new ranking…"]], select ?? $("r-file").value);
  rankingFileChanged();
}

function rankingFileChanged() {
  const isNew = $("r-file").value === NEW_RANKING;
  $("r-new").hidden = !isNew;
  const r = (rank.files || []).find((x) => x.file === $("r-file").value);
  $("r-file-info").textContent = isNew ? "a new file, for a person or a question"
    : r ? rankingSummary(r) + (r.legacy ? " Old DiffDrawing format: the answers are saved to a new ranking_ file." : "") : "";
  $("r-start").textContent = isNew ? "Create & rank" : "Rank";
}

$("r-dataset").addEventListener("change", () => loadRankingFiles());
$("r-file").addEventListener("change", rankingFileChanged);

async function startRanking() {
  $("r-form-status").textContent = "";
  if (rank.open && rank.unsaved) await saveRanking();
  const dataset = $("r-dataset").value;
  let file = $("r-file").value;
  try {
    if (file === NEW_RANKING) {
      const name = $("r-name").value.trim();
      if (!name) {
        $("r-form-status").textContent = "give the ranking a name (yours, or the question's)";
        $("r-name").focus();
        return;
      }
      file = (await postJSON("/api/ranking/new", { dataset, name, question: $("r-question").value.trim() })).file;
      await loadRankingFiles(file);
    }
    openRanking(await api(`/api/ranking/open?${query({ dataset, file })}`), dataset);
  } catch (e) {
    $("r-form-status").textContent = e.message.replace(/^.*?: \d+ /, "");
  }
}

$("r-start").addEventListener("click", startRanking);

// ---------------------------------------------------------------------------
// Answers
// ---------------------------------------------------------------------------

function openRanking(r, dataset) {
  if (r.count < 2) {
    $("r-form-status").textContent = "a ranking needs at least 2 images";
    return;
  }
  Object.assign(rank, {
    open: true, dataset, file: r.file, label: r.label, question: r.question, created: r.created, count: r.count,
    pairs: [], byKey: new Map(), repeats: r.repeats, history: [], skipped: new Set(), sinceRepeat: 0,
    unsaved: 0, current: null,
  });
  for (const [a, b, pref] of r.pairs) addAnswer(a, b, pref);
  rank.sinceRepeat = 0;     // a reopened ranking doesn't start with a repeat
  rank.manifestClusters = r.clusters;
  rank.byCluster = rank.rankOrder = null;
  fillStrategyModels();
  if (r.dropped) $("r-form-status").textContent = `${r.dropped} pairs of the file are of images not in this dataset (kept in the file)`;
  $("r-empty").hidden = true;
  $("r-session").hidden = false;
  $("r-title").textContent = `${r.label}  ·  ${dataset}`;
  $("r-ask").textContent = r.question || "Which one is better?";
  showNextPair();
}

// Record an answer about a pair as shown (a on the left). A repeat is scored
// against the earlier answer instead of being stored twice.
function addAnswer(a, b, pref, repeat = null) {
  const key = pairKey(a, b);
  const [lo, hi, value] = a < b ? [a, b, pref] : [b, a, -pref];
  if (repeat && pairKey(repeat.a, repeat.b) === key) {
    rank.repeats.push([lo, hi, repeat.first, value]);
    rank.sinceRepeat = 0;
    rank.history.push("repeat");
    return true;
  }
  if (rank.byKey.has(key)) return false;
  rank.byKey.set(key, rank.pairs.length);
  rank.pairs.push([lo, hi, value]);
  rank.skipped.delete(key);
  rank.sinceRepeat++;
  rank.history.push("pair");
  return true;
}

function undoAnswer() {
  const kind = rank.history.pop();
  if (kind === "repeat") {
    const last = rank.repeats.pop();
    return last ? [last[0], last[1]] : null;
  }
  const last = rank.pairs.pop();
  if (!last) return null;
  rank.byKey.delete(pairKey(last[0], last[1]));
  return last;
}

const fresh = (a, b) => a !== b && !rank.byKey.has(pairKey(a, b)) && !rank.skipped.has(pairKey(a, b));

// Any pair not answered yet: random draws while most are free (always, in
// practice: 200 images have 19900 pairs), the remaining ones listed otherwise
function randomPair() {
  const n = rank.count;
  for (let attempt = 0; attempt < 200; attempt++) {
    const a = Math.floor(Math.random() * n);
    let b = Math.floor(Math.random() * (n - 1));
    if (b >= a) b += 1;
    if (fresh(a, b)) return [a, b];
  }
  const remaining = [];
  for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) if (fresh(a, b)) remaining.push([a, b]);
  return remaining.length ? remaining[Math.floor(Math.random() * remaining.length)] : null;
}

// Now and then a pair answered a while ago (not one of the last few), on
// random sides so that the layout doesn't remind of the first answer. At most
// about 12% of the pairs, not before 15 answers, 12 answers apart.
function repeatPair() {
  if (rank.pairs.length < 15 || rank.sinceRepeat < 12) return null;
  if (rank.repeats.length >= Math.max(1, rank.pairs.length * 0.12)) return null;
  const eligible = rank.pairs.slice(0, rank.pairs.length - 8);
  if (!eligible.length) return null;
  const [a, b, first] = eligible[Math.floor(Math.random() * eligible.length)];
  return { pair: Math.random() < 0.5 ? [a, b] : [b, a], repeat: { a, b, first } };
}

// Two images of the same cluster, clusters drawn by how many pairs they have
function similarPair() {
  if (!rank.byCluster) return null;
  const clusters = [...rank.byCluster.values()];
  const weights = clusters.map((m) => (m.length * (m.length - 1)) / 2);
  const total = weights.reduce((sum, w) => sum + w, 0);
  for (let attempt = 0; attempt < 80; attempt++) {
    let roll = Math.random() * total;
    let members = clusters[clusters.length - 1];
    for (let c = 0; c < clusters.length; c++) {
      roll -= weights[c];
      if (roll <= 0) { members = clusters[c]; break; }
    }
    const a = members[Math.floor(Math.random() * members.length)];
    const b = members[Math.floor(Math.random() * members.length)];
    if (fresh(a, b)) return [a, b];
  }
  return null;
}

// Two images at most 5 places apart in the ranking
function rankingPair() {
  const order = rank.rankOrder;
  if (!order || order.length < 2) return null;
  for (let attempt = 0; attempt < 80; attempt++) {
    const i = Math.floor(Math.random() * order.length);
    const span = 1 + Math.floor(Math.random() * 5);
    const j = Math.random() < 0.5 ? i - span : i + span;
    if (j >= 0 && j < order.length && fresh(order[i], order[j])) return [order[i], order[j]];
  }
  return null;
}

function drawPair() {
  const strategy = $("r-strategy").value;
  if (strategy === "random") return randomPair();
  if (strategy === "similar") return similarPair() ?? randomPair();
  if (strategy === "ranking") return rankingPair() ?? randomPair();
  if (Math.random() < 0.5) return randomPair();
  const sources = [];
  if (rank.byCluster) sources.push(similarPair);
  if (rank.rankOrder) sources.push(rankingPair);
  if (Math.random() < 0.5) sources.reverse();
  for (const draw of sources) {
    const pair = draw();
    if (pair) return pair;
  }
  return randomPair();
}

// No side effects: also called to preload the images of the pair after this one
function nextPair() {
  const probe = repeatPair();
  if (probe) return probe;
  const pair = drawPair();
  return pair && { pair: Math.random() < 0.5 ? pair : [pair[1], pair[0]], repeat: null };
}

const rankImageURL = (i) => `/api/preview?${query({ dataset: rank.dataset, i, size: RANK_IMAGE_SIZE })}`;

function showPair(current) {
  rank.current = current;
  const done = !current;
  $("r-left").querySelector("img").src = done ? "" : rankImageURL(current.pair[0]);
  $("r-right").querySelector("img").src = done ? "" : rankImageURL(current.pair[1]);
  for (const id of ["r-left", "r-right", "r-pick-left", "r-pick-equal", "r-pick-right", "r-skip"]) $(id).disabled = done;
  if (done) $("r-ask").textContent = "Every pair has an answer.";
  // the browser loads the following pair's images while this one is looked at
  const following = nextPair();
  if (following) following.pair.forEach((i) => { new Image().src = rankImageURL(i); });
  updateRankingCounts();
}

function showNextPair() {
  showPair(nextPair());
}

function updateRankingCounts() {
  const n = rank.count;
  const strict = rank.pairs.filter((p) => p[2] !== 0).length;
  const agree = rank.repeats.filter(([, , first, second]) => Math.sign(first) === Math.sign(second)).length;
  $("r-counts").textContent = `${rank.pairs.length} pairs answered (${strict} with a preference) of ${n * (n - 1) / 2} possible`
    + (rank.repeats.length ? `  ·  self-consistency ${(100 * agree / rank.repeats.length).toFixed(0)}% over ${rank.repeats.length} repeats` : "");
  $("r-saved").textContent = rank.saving ? "saving" : rank.unsaved ? `${rank.unsaved} unsaved` : "saved";
  $("r-saved").className = `badge${rank.unsaved ? " running" : ""}`;
  $("r-undo").disabled = !rank.history.length;
}

function answer(pref) {
  if (!rank.current) return;
  const [a, b] = rank.current.pair;
  addAnswer(a, b, pref, rank.current.repeat);
  rank.unsaved++;
  if (rank.unsaved >= AUTOSAVE_EVERY) saveRanking();
  showNextPair();
}

function skip() {
  if (!rank.current) return;
  // a skipped repeat is simply not asked again now; a skipped pair may come back another session
  if (!rank.current.repeat) rank.skipped.add(pairKey(...rank.current.pair));
  else rank.sinceRepeat = 0;
  showNextPair();
}

function undo() {
  const last = undoAnswer();
  if (!last) return;
  rank.unsaved++;
  showPair({ pair: [last[0], last[1]], repeat: null });
}

$("r-left").addEventListener("click", () => answer(1));
$("r-right").addEventListener("click", () => answer(-1));
$("r-pick-left").addEventListener("click", () => answer(1));
$("r-pick-right").addEventListener("click", () => answer(-1));
$("r-pick-equal").addEventListener("click", () => answer(0));
$("r-skip").addEventListener("click", skip);
$("r-undo").addEventListener("click", undo);

window.addEventListener("keydown", (event) => {
  if (!rank.open || $("view-data").hidden || $("r-board").hidden || $("r-session").hidden) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
  const actions = { ArrowLeft: () => answer(1), ArrowRight: () => answer(-1), ArrowDown: () => answer(0),
                    s: skip, S: skip, Backspace: undo };
  if (actions[event.key]) {
    event.preventDefault();
    actions[event.key]();
  }
});

// ---------------------------------------------------------------------------
// How pairs are chosen: the clusters / ranking they need
// ---------------------------------------------------------------------------

const SCORE_KINDS = ["ranknet", "aesthetic"];

// "alike by": the Sampler's clusters and the models with vectors; "ranking by":
// the models that give a score (state.options: the map's model list)
function fillStrategyModels() {
  const models = state.options ? state.options.models : [];
  const similar = [["none", "–"], ...(rank.manifestClusters ? [["manifest", "the Sampler's clusters"]] : []),
    ...models.filter((m) => !SCORE_KINDS.includes(m.kind)).map((m) => [m.id, m.label])];
  const keep = $("r-similar").value;
  fillSelect($("r-similar"), similar, keep && keep !== "none" ? keep : (rank.manifestClusters ? "manifest" : "none"));
  fillSelect($("r-ranked"), [["none", "–"], ...models.filter((m) => SCORE_KINDS.includes(m.kind)).map((m) => [m.id, m.label])],
    $("r-ranked").value || "none");
  strategyChanged();
}

function strategyChanged() {
  const strategy = $("r-strategy").value;
  $("r-similar-row").hidden = !["mixed", "similar"].includes(strategy);
  $("r-ranked-row").hidden = !["mixed", "ranking"].includes(strategy);
  const ranked = state.options && state.options.models.find((m) => m.id === $("r-ranked").value);
  $("r-ranked-prompt-row").hidden = $("r-ranked-row").hidden || !(ranked && ranked.prompt);
  if (rank.open) updateHelpers();
  else describeStrategy();
}

function describeStrategy(progress = "") {
  const strategy = $("r-strategy").value;
  const clusters = rank.byCluster ? `${rank.byCluster.size} clusters of alike images` : null;
  const ranking = rank.rankOrder ? `a ranking of ${rank.rankOrder.length} images` : null;
  const missing = (what) => `${what} not set: random pairs meanwhile`;
  $("r-strategy-info").textContent = progress || (
    strategy === "random" ? "Any pair not answered yet."
      : strategy === "similar" ? (clusters ? `Two images of the same cluster (${clusters}).` : missing("alike by"))
        : strategy === "ranking" ? (ranking ? `Two images at most 5 places apart in ${ranking}.` : missing("ranking by"))
          : `Half random, half ${[clusters && "within clusters", ranking && "close in the ranking"].filter(Boolean).join(" or ") || "random too (set alike by or ranking by)"}.`);
}

// Fetch (once per dataset and model) what the chosen strategy needs
async function updateHelpers() {
  const run = ++rank.helperRun;
  const strategy = $("r-strategy").value;
  const similar = ["mixed", "similar"].includes(strategy) ? $("r-similar").value : "none";
  const ranked = ["mixed", "ranking"].includes(strategy) ? $("r-ranked").value : "none";
  rank.byCluster = similar === "manifest" ? groupClusters(rank.manifestClusters) : rank.byCluster;
  if (similar === "none") rank.byCluster = null;
  if (ranked === "none") rank.rankOrder = null;
  describeStrategy();
  try {
    if (similar !== "none" && similar !== "manifest") {
      const { clusters } = await helper("regions", similar, "", run);
      if (run !== rank.helperRun) return;
      rank.byCluster = groupClusters(clusters);
    }
    if (ranked !== "none") {
      const { scores } = await helper("scores", ranked, $("r-ranked-prompt").value.trim(), run);
      if (run !== rank.helperRun) return;
      rank.rankOrder = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    }
    describeStrategy();
  } catch (e) {
    if (run === rank.helperRun) $("r-strategy-info").textContent = e.message.replace(/^.*?: \d+ /, "");
  }
}

// Clusters of the dataset's images, or the scores, by a model: computed on the
// server (embedding every image the first time), polled until done
async function helper(kind, model, prompt, run) {
  const key = [rank.dataset, kind, model, prompt].join("|");
  if (helpers.has(key)) return helpers.get(key);
  const { job } = await postJSON("/api/ranking/helper", { dataset: rank.dataset, kind, model, prompt });
  for (;;) {
    const st = await api(`/api/job/${job}`);
    if (st.error) throw new Error(st.error);
    if (st.finished) {
      helpers.set(key, st.result);
      return st.result;
    }
    if (run === rank.helperRun) describeStrategy(`${kind === "regions" ? "clustering" : "scoring"} the images: ${st.stage}…`);
    await sleep(600);
  }
}

// cluster of every image -> Map cluster -> its images, clusters of 2 or more
function groupClusters(clusters) {
  if (!clusters) return null;
  const grouped = new Map();
  clusters.forEach((c, i) => {
    if (!grouped.has(c)) grouped.set(c, []);
    grouped.get(c).push(i);
  });
  const usable = new Map([...grouped].filter(([, members]) => members.length > 1));
  return usable.size ? usable : null;
}

for (const id of ["r-strategy", "r-similar", "r-ranked", "r-ranked-prompt"]) $(id).addEventListener("change", strategyChanged);

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

const rankingBody = () => ({ dataset: rank.dataset, file: rank.file, question: rank.question, created: rank.created,
                             pairs: rank.pairs, repeats: rank.repeats });

async function saveRanking() {
  if (!rank.open || rank.saving) return;
  rank.saving = true;
  const unsaved = rank.unsaved;
  updateRankingCounts();
  try {
    const saved = await postJSON("/api/ranking/save", rankingBody());
    rank.file = saved.file;         // an old-format file is saved to a new ranking_ file
    rank.unsaved -= unsaved;
  } catch (e) {
    $("r-form-status").textContent = `not saved: ${e.message}`;
  } finally {
    rank.saving = false;
    updateRankingCounts();
  }
}

$("r-done").addEventListener("click", async () => {
  await saveRanking();
  if (rank.unsaved) return;         // the error is shown: keep the answers on screen
  rank.open = false;
  $("r-session").hidden = true;
  $("r-empty").hidden = false;
  await loadRankingFiles(rank.file);
});

// closing or reloading the page: send what wasn't saved yet
window.addEventListener("pagehide", () => {
  if (rank.open && rank.unsaved) {
    navigator.sendBeacon("/api/ranking/save", new Blob([JSON.stringify(rankingBody())], { type: "application/json" }));
  }
});
