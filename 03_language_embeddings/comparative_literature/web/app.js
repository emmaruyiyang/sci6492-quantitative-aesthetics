const SVG_NS = "http://www.w3.org/2000/svg";
const RIM = 1.14; // radius (in ring units) where words absent from a book are drawn

const form = document.querySelector("#query-form");
const wordInput = document.querySelector("#word");
const topN = document.querySelector("#top_n");
const topNOut = document.querySelector("#top_n_out");
const sourceControl = document.querySelector("#source");
const layoutControl = document.querySelector("#layout");
const modeControl = document.querySelector("#mode");
const axisControl = document.querySelector("#axis");
const main = document.querySelector("main");
const legend = document.querySelector("#legend");
const tooltip = document.querySelector("#tooltip");
const message = document.querySelector("#message");
const trailNav = document.querySelector("#trail");

let data = null;   // last response from /api/compare
let trail = [];    // words visited, in order
let requestId = 0; // only the latest request is allowed to draw

// ---------------------------------------------------------------- helpers

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function lerp(a, b, t) { return a + (b - a) * t; }

function mixHex(a, b, t) {
  const pa = a.match(/\w\w/g).map((h) => parseInt(h, 16));
  const pb = b.match(/\w\w/g).map((h) => parseInt(h, 16));
  return "#" + pa.map((v, i) => Math.round(lerp(v, pb[i], t)).toString(16).padStart(2, "0")).join("");
}

// normalised similarity on the scale shared by both books: 0 = least similar word shown, 1 = most similar
function simT(sim) {
  const [lo, hi] = data.sim_range;
  return hi > lo ? Math.min(1, Math.max(0, (sim - lo) / (hi - lo))) : 1;
}

// axis encoding: the "from" end in warm yellow, the "to" end in blue, neutral gray in between
const AXIS_FROM = "#f2c94c", AXIS_MID = "#6e6e73", AXIS_TO = "#3d7fe0";
function axisColor(v) {
  return v < 0 ? mixHex(AXIS_MID, AXIS_FROM, Math.min(1, -v)) : mixHex(AXIS_MID, AXIS_TO, Math.min(1, v));
}

function segmentedValue(control) { return control.dataset.value; }

function formatCount(n) { return n >= 1000 ? `${Math.round(n / 1000)}k` : `${n}`; }

// ---------------------------------------------------------------- data

async function compare(word, { pushTrail = true } = {}) {
  word = word.trim().toLowerCase();
  if (!word) return;
  wordInput.value = word;
  message.textContent = "reading both books…";

  const id = ++requestId;
  const params = new URLSearchParams({
    word,
    source: segmentedValue(sourceControl),
    top_n: topN.value,
    layout: segmentedValue(layoutControl),
    axis: segmentedValue(axisControl),
  });
  const response = await fetch(`/api/compare?${params}`);
  const json = await response.json();
  if (id !== requestId) return; // a newer query was sent while this one was loading

  if (!response.ok) {
    message.textContent = json.error;
    return;
  }
  message.textContent = "";
  data = json;

  if (pushTrail && trail[trail.length - 1] !== word) trail.push(word);
  renderTrail();
  render();
}

function renderTrail() {
  trailNav.innerHTML = "";
  trail.forEach((w, i) => {
    if (i > 0) trailNav.insertAdjacentHTML("beforeend", '<span class="sep">/</span>');
    const a = document.createElement("a");
    a.textContent = w;
    a.addEventListener("click", () => {
      trail = trail.slice(0, i + 1);
      compare(w, { pushTrail: false });
    });
    trailNav.appendChild(a);
  });
}

// ---------------------------------------------------------------- layout

// push overlapping labels apart while a weak spring pulls each back to its semantic position
function relax(items, fixed) {
  const pad = 6;
  for (let iter = 0; iter < 300; iter++) {
    let moved = false;
    const all = [fixed, ...items];
    for (let i = 0; i < all.length; i++) {
      for (let j = i + 1; j < all.length; j++) {
        const a = all[i], b = all[j];
        const ox = (a.w + b.w) / 2 + pad - Math.abs(a.x - b.x);
        const oy = (a.h + b.h) / 2 + pad - Math.abs(a.y - b.y);
        if (ox > 0 && oy > 0) {
          moved = true;
          // separate along the axis with the smaller overlap
          const dx = a.x - b.x || Math.random() - 0.5;
          const dy = a.y - b.y || Math.random() - 0.5;
          const wa = a === fixed ? 0 : 1, wb = b === fixed ? 0 : 1;
          const share = 1 / (wa + wb);
          if (ox < oy) {
            const s = Math.sign(dx) * ox * share;
            a.x += s * wa; b.x -= s * wb;
          } else {
            const s = Math.sign(dy) * oy * share;
            a.y += s * wa; b.y -= s * wb;
          }
        }
      }
    }
    for (const it of items) {
      it.x += (it.tx - it.x) * 0.02;
      it.y += (it.ty - it.y) * 0.02;
    }
    if (!moved && iter > 20) break;
  }
}

// ---------------------------------------------------------------- drawing

function render() {
  if (!data) return;
  const [holmes, abyss] = data.panels;
  // an absent word is placed in the direction it has in the other book
  renderPanel(holmes, abyss);
  renderPanel(abyss, holmes);
  renderLegend();
  if (highlighted) highlight(highlighted);
  if (hoverWord) simulateHover(hoverWord);
}

// ?hover=word shows the hover state without a mouse (used for screenshots):
// once the words are in place, hover the word in the first panel that has it.
// every render replaces the words, so a render cancels the pending hover and schedules a new one
let hoverTimer = null;
function simulateHover(word) {
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    const g = document.querySelector(`.panel svg .word:not(.ghost)[data-word="${CSS.escape(word)}"]`);
    if (!g) return;
    const box = g.getBoundingClientRect();
    g.dispatchEvent(new MouseEvent("mouseenter"));
    g.dispatchEvent(new MouseEvent("mousemove", { clientX: box.right, clientY: box.bottom }));
  }, 1200);
}

function renderPanel(panel, other) {
  const section = document.querySelector(`#panel-${panel.key}`);
  const svg = section.querySelector("svg");
  const mode = segmentedValue(modeControl);

  // panel header: title and how often the centre word occurs in this book
  let colorNote = "";
  if (mode === "color") {
    colorNote = ` · nearest color <b>${panel.query_color_term}</b>`;
  }
  section.querySelector(".panel-head").innerHTML =
    `<div class="title">${panel.title}</div>` +
    `<div class="meta">${panel.author}, ${panel.years} · ${formatCount(panel.words)} words · ` +
    `<b>${data.word}</b> × ${panel.query_count}${colorNote}</div>`;

  const { width, height } = svg.getBoundingClientRect();
  const cx = width / 2, cy = height / 2;
  const R = Math.min(width, height) / 2 / 1.26; // leave room for absent words beyond the rim
  const [lo, hi] = data.sim_range;

  svg.innerHTML = "";

  // similarity rings on the shared scale: radius = 0.28 + 0.72 * normalised distance
  if (data.layout === "radial") {
    for (const r of [0.28, 0.52, 0.76, 1.0]) {
      el("circle", { class: "ring", cx, cy, r: r * R }, svg);
      const sim = hi - ((r - 0.28) / 0.72) * (hi - lo);
      const label = el("text", { class: "ring-label", x: cx + 4, y: cy - r * R - 4 }, svg);
      label.textContent = sim.toFixed(2);
    }
  }
  if (panel.absent.length) {
    el("circle", { class: "rim", cx, cy, r: RIM * R }, svg);
    const label = el("text", { class: "ring-label", x: cx + 4, y: cy - RIM * R - 4 }, svg);
    label.textContent = "not in this book";
  }

  const linkLayer = el("g", {}, svg);
  const wordLayer = el("g", {}, svg);

  // query word, fixed at the centre
  const q = el("g", { class: "word query", transform: `translate(${cx},${cy})` }, wordLayer);
  const qText = el("text", {}, q);
  qText.textContent = data.word;
  const qBox = qText.getBBox();
  const fixed = { x: cx, y: cy, w: qBox.width, h: qBox.height };

  // words present in this book
  const items = panel.results.map((r) => {
    const t = simT(r.similarity);
    const g = el("g", { class: "word", "data-word": r.word, opacity: 0 }, wordLayer);
    const attrs = { "font-size": lerp(12, 30, t).toFixed(1), fill: mixHex("#48484a", "#f5f5f7", t) };

    if (mode === "shared") {
      // both books have it as a neighbour / only this book / only the other book
      if (r.provenance === "both") { attrs.fill = "#f5f5f7"; attrs["font-weight"] = 600; }
      else if (r.provenance === "own") { attrs.fill = "#a1a1a6"; }
      else if (r.provenance === "other") { attrs.fill = "#6e6e73"; attrs["font-style"] = "italic"; }
    } else if (mode === "axis") {
      attrs.fill = axisColor(r.axis);
    }

    const text = el("text", attrs, g);
    text.textContent = r.word;
    let box = text.getBBox();
    if (mode === "color") {
      // a small dot in this book's nearest color word, left of the label
      const size = Math.max(3, box.height * 0.16);
      el("circle", {
        class: "dot", cx: -box.width / 2 - size - 4, cy: 0, r: size,
        fill: panel.palette[r.color_term],
      }, g);
      box = g.getBBox();
    }
    const tx = cx + r.x * R, ty = cy - r.y * R;
    return { r, g, t, tx, ty, x: tx, y: ty, w: box.width, h: box.height };
  });

  relax(items, fixed);

  // words the comparison picked that this book never uses: beyond the rim,
  // in the direction they point to in the other book
  const otherByWord = Object.fromEntries(other.results.map((r) => [r.word, r]));
  const ghosts = panel.absent.map((w, i) => {
    const o = otherByWord[w];
    const angle = o ? Math.atan2(o.y, o.x) : (i / panel.absent.length) * 2 * Math.PI;
    const g = el("g", { class: "word ghost", "data-word": w, opacity: 0 }, wordLayer);
    const text = el("text", {}, g);
    text.textContent = w;
    const box = text.getBBox();
    const tx = cx + Math.cos(angle) * RIM * R, ty = cy - Math.sin(angle) * RIM * R;
    return { r: { word: w, absent: true }, g, tx, ty, x: tx, y: ty, w: box.width, h: box.height };
  });
  // ghosts only avoid each other, then get pinned back onto the rim
  relax(ghosts, { x: -1e6, y: -1e6, w: 0, h: 0 });
  for (const it of ghosts) {
    const a = Math.atan2(it.y - cy, it.x - cx);
    it.x = cx + Math.cos(a) * RIM * R;
    it.y = cy + Math.sin(a) * RIM * R;
  }

  for (const it of items) {
    el("line", {
      class: "link", x1: cx, y1: cy, x2: it.x, y2: it.y,
      "stroke-opacity": lerp(0.03, 0.18, it.t).toFixed(2),
    }, linkLayer);
  }
  for (const it of ghosts) {
    el("line", { class: "ghost-link", x1: cx, y1: cy, x2: it.x, y2: it.y }, linkLayer);
  }

  for (const it of [...items, ...ghosts]) {
    if (!it.r.absent) it.g.addEventListener("click", () => compare(it.r.word));
    it.g.addEventListener("mouseenter", () => highlight(it.r.word));
    it.g.addEventListener("mousemove", (e) => showTooltip(e, it.r, panel, other));
    it.g.addEventListener("mouseleave", () => { highlight(null); tooltip.hidden = true; });
  }

  // animate words outwards from the query: start at the centre, then move to the final position
  const all = [...items, ...ghosts];
  for (const it of all) it.g.style.transform = `translate(${cx}px, ${cy}px)`;
  if (all.length) getComputedStyle(all[0].g).transform;
  for (const it of all) {
    it.g.style.transform = `translate(${it.x}px, ${it.y}px)`;
    it.g.setAttribute("opacity", 1);
  }
}

// hovering a word in one book highlights the same word in the other book
let highlighted = null; // kept so a re-render (e.g. on resize) can restore the highlight
function highlight(word) {
  highlighted = word;
  main.classList.toggle("linking", !!word);
  document.querySelectorAll(".panel svg .word.hl").forEach((n) => n.classList.remove("hl"));
  if (word) {
    document.querySelectorAll(`.panel svg .word[data-word="${CSS.escape(word)}"]`)
      .forEach((n) => n.classList.add("hl"));
  }
}

function showTooltip(e, r, panel, other) {
  const rect = main.getBoundingClientRect();
  const short = (p) => (p.key === "holmes" ? "Holmes" : "Abyss");
  if (r.absent) {
    const o = other.results.find((x) => x.word === r.word);
    tooltip.innerHTML =
      `<b>${r.word}</b>` +
      `not in ${short(panel)} (fewer than 5 uses)<br>` +
      (o ? `similarity in ${short(other)}<span class="num">${o.similarity.toFixed(3)}</span><br>` +
           `uses in ${short(other)}<span class="num">${o.count}</span>` : "");
  } else {
    const provenance = {
      both: "neighbour in both books",
      own: `neighbour in ${short(panel)} only`,
      other: `neighbour in ${short(other)} only`,
      list: "Gehl word list",
    }[r.provenance];
    tooltip.innerHTML =
      `<b>${r.word}</b>` +
      `similarity to ${data.word}<br>` +
      `&nbsp; in ${short(panel)}<span class="num">${r.similarity.toFixed(3)}</span><br>` +
      `&nbsp; in ${short(other)}<span class="num">${r.other_similarity === null ? "absent" : r.other_similarity.toFixed(3)}</span><br>` +
      `uses (${short(panel)} / ${short(other)})<span class="num">${r.count} / ${r.other_count}</span><br>` +
      `${provenance}<br>` +
      `color word<span class="num">${r.color_term} · ${r.color_term_sim.toFixed(2)}</span><br>` +
      `${data.axis.from} → ${data.axis.to}<span class="num">${r.axis >= 0 ? "+" : ""}${r.axis.toFixed(2)}</span>`;
  }
  tooltip.style.left = `${Math.min(e.clientX - rect.left + 14, rect.width - 240)}px`;
  tooltip.style.top = `${e.clientY - rect.top + 14}px`;
  tooltip.hidden = false;
}

function renderLegend() {
  const mode = segmentedValue(modeControl);
  let html = "";
  if (mode === "shared") {
    html = '<div class="title">Whose neighbour is it</div>' +
      '<span class="shared-both">both books</span><br>' +
      '<span class="shared-own">this book only</span><br>' +
      '<span class="shared-other">the other book only</span>';
    if (data.source === "list") html += '<br><span style="color:#86868b">(Gehl list: all words chosen by hand)</span>';
  } else if (mode === "color") {
    // each book is colored only with the color words it contains
    html = '<div class="title">Nearest color word · each book\'s own palette</div>';
    for (const p of data.panels) {
      const counts = {};
      p.results.forEach((r) => {
        const hex = p.palette[r.color_term];
        counts[hex] = counts[hex] || { names: new Set(), n: 0 };
        counts[hex].names.add(r.color_term);
        counts[hex].n++;
      });
      const swatches = Object.values(p.palette).filter((v, i, a) => a.indexOf(v) === i)
        .map((hex) => `<span class="swatch" style="background:${hex};opacity:${counts[hex] ? 1 : 0.25}"></span>`).join("");
      html += `${p.key === "holmes" ? "Holmes" : "Abyss"} &nbsp;${swatches}<br>`;
    }
  } else if (mode === "axis") {
    html = '<div class="title">Semantic axis · projection in each book</div>' +
      `${data.axis.from}<span class="axis-ramp" style="background:linear-gradient(90deg, ${AXIS_FROM}, ${AXIS_MID}, ${AXIS_TO})"></span>${data.axis.to}`;
  } else {
    html = '<div class="title">Similarity · same scale in both books</div>less<span class="ramp"></span>more';
  }
  html += '<br><span class="ghost-sample">italic beyond the rim</span> = not in this book';
  html += '<br><span style="color:#86868b">Size = similarity to the centre word</span>';
  legend.innerHTML = html;
}

// ---------------------------------------------------------------- events

form.addEventListener("submit", (e) => {
  e.preventDefault();
  compare(wordInput.value);
});
topN.addEventListener("input", () => (topNOut.textContent = topN.value));
topN.addEventListener("change", () => data && compare(data.word, { pushTrail: false }));

function initSegmented(control, onChange) {
  const buttons = control.querySelectorAll("button");
  const sync = () => buttons.forEach((b) => b.classList.toggle("active", b.dataset.value === control.dataset.value));
  buttons.forEach((b) => b.addEventListener("click", () => {
    if (control.dataset.value === b.dataset.value) return;
    control.dataset.value = b.dataset.value;
    sync();
    onChange();
  }));
  sync();
}

function syncControls() {
  document.querySelector("#axis_control").hidden = segmentedValue(modeControl) !== "axis";
  document.querySelector("#top_n_control").style.opacity = segmentedValue(sourceControl) === "list" ? 0.35 : 1;
}

// optional URL parameters, e.g. /?word=night&source=list&mode=axis&axis=clean-dirty&n=20&trail=street,night
const params = new URLSearchParams(location.search);
if (params.get("n")) topN.value = topNOut.textContent = params.get("n");
if (params.get("source")) sourceControl.dataset.value = params.get("source");
if (params.get("layout")) layoutControl.dataset.value = params.get("layout");
if (params.get("mode")) modeControl.dataset.value = params.get("mode");
if (params.get("axis")) axisControl.dataset.value = params.get("axis");
if (params.get("trail")) trail = params.get("trail").split(",");
const hoverWord = params.get("hover");
// screenshot mode: no fly-out animation, so the words are already in place when the hover happens
if (hoverWord) document.body.classList.add("no-anim");

const refetch = () => data && compare(data.word, { pushTrail: false });
initSegmented(sourceControl, () => { syncControls(); refetch(); });
initSegmented(layoutControl, refetch);
initSegmented(modeControl, () => { syncControls(); render(); });
initSegmented(axisControl, refetch);
syncControls();
window.addEventListener("resize", render);

compare(params.get("word") || wordInput.value);
