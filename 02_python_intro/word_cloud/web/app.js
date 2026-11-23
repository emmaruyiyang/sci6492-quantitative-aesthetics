const SVG_NS = "http://www.w3.org/2000/svg";

const form = document.querySelector("#query-form");
const wordInput = document.querySelector("#word");
const topN = document.querySelector("#top_n");
const topNOut = document.querySelector("#top_n_out");
const layoutControl = document.querySelector("#layout");
const modeControl = document.querySelector("#mode");
const svg = document.querySelector("#cloud");
const legend = document.querySelector("#legend");
const tooltip = document.querySelector("#tooltip");
const message = document.querySelector("#message");
const trailNav = document.querySelector("#trail");

let data = null;   // last response from /api/cloud
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

// normalised similarity: 0 = least similar word shown, 1 = most similar
function simT(sim) {
  const sims = data.results.map((r) => r.similarity);
  const lo = Math.min(...sims), hi = Math.max(...sims);
  return hi > lo ? (sim - lo) / (hi - lo) : 1;
}

// the interface stays black and white:
// similarity -> gray level, clusters -> typographic style, color words -> a small colored dot
const CLUSTER_STYLES = [
  { weight: 600, style: "normal", sample: "Bold" },
  { weight: 400, style: "italic", sample: "Italic" },
  { weight: 300, style: "normal", sample: "Light" },
  { weight: 500, style: "normal", caps: true, sample: "CAPS" },
  { weight: 400, style: "normal", underline: true, sample: "Underline" },
  { weight: 700, style: "italic", sample: "Bold italic" },
];

function grayFor(r) {
  return mixHex("#48484a", "#f5f5f7", simT(r.similarity));
}

function segmentedValue(control) { return control.dataset.value; }

// ---------------------------------------------------------------- data

async function explore(word, { pushTrail = true } = {}) {
  word = word.trim().toLowerCase();
  if (!word) return;
  wordInput.value = word;
  message.textContent = "listening to the embedding…";

  const id = ++requestId;
  const params = new URLSearchParams({ word, top_n: topN.value, layout: segmentedValue(layoutControl) });
  const response = await fetch(`/api/cloud?${params}`);
  const json = await response.json();
  if (id !== requestId) return; // a newer query was sent while this one was loading

  if (!response.ok) {
    message.textContent = json.error;
    return;
  }
  message.textContent = "";
  data = json;
  document.querySelector("#model-name").textContent = data.model;

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
      explore(w, { pushTrail: false });
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
  const { width, height } = svg.getBoundingClientRect();
  const cx = width / 2, cy = height / 2;
  const R = Math.min(width, height) / 2 - 50;

  svg.innerHTML = "";
  const mode = segmentedValue(modeControl);

  const sims = data.results.map((r) => r.similarity);
  const lo = Math.min(...sims), hi = Math.max(...sims);

  // similarity rings (radial layout only): radius = 0.28 + 0.72 * normalised distance
  if (data.layout === "radial") {
    for (const r of [0.28, 0.52, 0.76, 1.0]) {
      el("circle", { class: "ring", cx, cy, r: r * R }, svg);
      const sim = hi - ((r - 0.28) / 0.72) * (hi - lo);
      const label = el("text", { class: "ring-label", x: cx + 4, y: cy - r * R - 4 }, svg);
      label.textContent = sim.toFixed(2);
    }
  }

  const linkLayer = el("g", {}, svg);
  const wordLayer = el("g", {}, svg);

  // query word, fixed at the centre
  const q = el("g", { class: "word query", transform: `translate(${cx},${cy})` }, wordLayer);
  const qText = el("text", {}, q);
  qText.textContent = data.word;
  const qBox = qText.getBBox();
  const fixed = { x: cx, y: cy, w: qBox.width, h: qBox.height };

  // neighbours: measure each label, then resolve overlaps
  const items = data.results.map((r) => {
    const t = simT(r.similarity);
    const g = el("g", { class: "word", transform: `translate(${cx},${cy})`, opacity: 0 }, wordLayer);
    const attrs = { "font-size": lerp(12, 30, t).toFixed(1), fill: grayFor(r) };
    if (mode === "cluster") {
      const cs = CLUSTER_STYLES[r.cluster % CLUSTER_STYLES.length];
      attrs.fill = "#f5f5f7";
      attrs["fill-opacity"] = lerp(0.45, 1, t).toFixed(2);
      attrs["font-weight"] = cs.weight;
      attrs["font-style"] = cs.style;
      if (cs.underline) attrs["text-decoration"] = "underline";
    }
    const text = el("text", attrs, g);
    text.textContent = mode === "cluster" && CLUSTER_STYLES[r.cluster % CLUSTER_STYLES.length].caps
      ? r.word.toUpperCase() : r.word;
    let box = text.getBBox();
    if (mode === "semantic") {
      // a small dot in the color named by the nearest color word, left of the label
      const size = Math.max(3, box.height * 0.16);
      el("circle", {
        class: "dot", cx: -box.width / 2 - size - 4, cy: 0, r: size,
        fill: data.color_terms[r.color_term],
      }, g);
      box = g.getBBox();
    }
    const tx = cx + r.x * R, ty = cy - r.y * R;
    return { r, g, t, tx, ty, x: tx, y: ty, w: box.width, h: box.height };
  });

  relax(items, fixed);

  for (const it of items) {
    el("line", {
      class: "link", x1: cx, y1: cy, x2: it.x, y2: it.y,
      "stroke-opacity": lerp(0.03, 0.18, it.t).toFixed(2),
    }, linkLayer);

    it.g.addEventListener("click", () => explore(it.r.word));
    it.g.addEventListener("mousemove", (e) => showTooltip(e, it.r));
    it.g.addEventListener("mouseleave", () => (tooltip.hidden = true));
  }

  // animate words outwards from the query: start them at the centre, force the browser
  // to register that position, then set the final one so the CSS transition runs
  for (const it of items) it.g.style.transform = `translate(${cx}px, ${cy}px)`;
  getComputedStyle(wordLayer.lastChild).transform;
  for (const it of items) {
    it.g.style.transform = `translate(${it.x}px, ${it.y}px)`;
    it.g.setAttribute("opacity", 1);
  }

  renderLegend();
}

function showTooltip(e, r) {
  const rect = svg.getBoundingClientRect();
  const cluster = data.clusters[r.cluster].name;
  tooltip.innerHTML =
    `<b>${r.word}</b>` +
    `similarity to ${data.word}<span class="num">${r.similarity.toFixed(3)}</span><br>` +
    `cluster<span class="num">${cluster}</span><br>` +
    `color word<span class="num">${r.color_term} · ${r.color_term_sim.toFixed(2)}</span>`;
  tooltip.style.left = `${e.clientX - rect.left + 14}px`;
  tooltip.style.top = `${e.clientY - rect.top + 14}px`;
  tooltip.hidden = false;
}

function renderLegend() {
  const mode = segmentedValue(modeControl);
  let html = "";
  if (mode === "cluster") {
    html = '<div class="title">Clusters · k-means on word vectors</div>' +
      data.clusters.map((c, i) => {
        const cs = CLUSTER_STYLES[i % CLUSTER_STYLES.length];
        const style = `font-weight:${cs.weight};font-style:${cs.style};${cs.underline ? "text-decoration:underline;" : ""}`;
        const name = cs.caps ? c.name.toUpperCase() : c.name;
        return `<span style="${style};color:#f5f5f7">${name}</span>`;
      }).join("<br>");
  } else if (mode === "semantic") {
    const counts = {};
    data.results.forEach((r) => (counts[r.color_term] = (counts[r.color_term] || 0) + 1));
    html = `<div class="title">Nearest color word · ${data.word} → ${data.query_color_term}</div>` +
      Object.entries(counts).sort((a, b) => b[1] - a[1])
        .map(([c, n]) => `<span class="swatch" style="background:${data.color_terms[c]}"></span>${c} &nbsp;${n}`).join("<br>");
  } else {
    html = '<div class="title">Similarity</div>less<span class="ramp"></span>more';
  }
  html += '<br><span style="color:#86868b">Size = similarity to the centre word</span>';
  legend.innerHTML = html;
}

// ---------------------------------------------------------------- events

form.addEventListener("submit", (e) => {
  e.preventDefault();
  explore(wordInput.value);
});
topN.addEventListener("input", () => (topNOut.textContent = topN.value));
topN.addEventListener("change", () => data && explore(data.word, { pushTrail: false }));
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
// optional URL parameters, e.g. /?word=ocean&n=40&layout=pca&mode=cluster&trail=architecture,ocean
const params = new URLSearchParams(location.search);
if (params.get("n")) topN.value = topNOut.textContent = params.get("n");
if (params.get("layout")) layoutControl.dataset.value = params.get("layout");
if (params.get("mode")) modeControl.dataset.value = params.get("mode");
if (params.get("trail")) trail = params.get("trail").split(",");

initSegmented(layoutControl, () => data && explore(data.word, { pushTrail: false }));
initSegmented(modeControl, render);
window.addEventListener("resize", render);

explore(params.get("word") || wordInput.value);
