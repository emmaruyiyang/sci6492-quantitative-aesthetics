// Spring forces: a projection computed live in the page (Projection > spring
// forces). The server sends the vectors (spring_vectors in the result: at most
// 64 numbers per image) and a PCA map to start from, or the images keep their
// place on the map shown before. From there every image is a mass on springs,
// simulated here frame after frame.
//
// A spring between two images pulls or pushes them toward its rest length, a
// function of their similarity (cosine of their centered vectors, -1..1):
//   equal     L whatever the similarity: only which pairs have springs counts
//   inverse   L / sim (sim taken as 0..1, L at 0.5): similar close, others far
//   linear    L × (1 − sim): 0 for twins, 2L for opposites
//   distance  L × their distance in vector space / the average distance
// and never shorter than `spacing`, so similar images don't pile up.
//
// Unrelated images get no spring (it would also pull them together, tying
// every cluster to every other) but a push that never pulls: `repulsion` when
// they touch, fading like 1/d² beyond `reach`. Between the two, a smooth step
// over the similarity: below `cutoff` − transition/2 pure push, above
// `cutoff` + transition/2 pure spring, a blend of both in between.
//
// Which pairs have springs:
//   all       every pair, every step. The N×N similarities are computed once,
//             a few rows per frame, and cached (N² numbers).
//   knn       each image's k most similar, found once at the start (a few
//             rows per frame), plus `random / step` random pairs every step.
//   dynamic   each image tracks k neighbours, random at first. Every step it
//             also gets springs to `random / step` images picked at random,
//             and any of them more similar than its least similar neighbour
//             takes its place. `tests / frame` more random pairs are compared
//             between frames, only to find neighbours sooner. Nothing is
//             computed in advance.
//
// Weights, so stiffness and dt mean about the same in every mode: all pairs
// 1/(N−1) each. Otherwise the k neighbours 1/k each, and the random pairs
// together `random weight` (random weight / M each): M only sets how many
// samples stand for all the other images, more is less noisy, not stronger.
// Pinned images (and the one dragged) don't move. Positions are map
// coordinates (0..1: the map at zoom 1), gravity pulls toward the middle.
//
// Pair lines (optional): every image to its `lines / image` most similar
// partners among its springs, each pair once, drawn under the images.

const spring = (() => {
  const ALL_MAX = 5000;       // all pairs: N² cached similarities (100 MB at 5000)
  const PREPARE_MS = 12;      // per frame, for the similarities computed at the start
  const MAX_MOVE = 0.02;      // per step, in map widths: a too large dt doesn't explode
  const SLIDERS = ["sp-k", "sp-m", "sp-weight", "sp-l", "sp-spacing", "sp-cutoff", "sp-transition", "sp-repulsion",
                   "sp-reach", "sp-stiffness", "sp-damping", "sp-dt", "sp-steps", "sp-gravity", "sp-lines-n"];

  let sim = null;             // the simulation of the current result
  let running = true;
  let frame = null;
  let held = null;            // the image being dragged: { i, row, x, y, moved }
  const pinned = new Set();   // image indices

  const params = () => ({
    length: $("sp-length").value,
    pairs: $("sp-pairs").value,
    k: Number($("sp-k").value),
    m: Number($("sp-m").value),
    weight: Number($("sp-weight").value),
    tests: Math.max(0, Math.floor(Number($("sp-tests").value) || 0)),
    L: Number($("sp-l").value),
    spacing: Number($("sp-spacing").value),
    cutoff: Number($("sp-cutoff").value),
    transition: Number($("sp-transition").value),
    repulsion: Number($("sp-repulsion").value),
    reach: Number($("sp-reach").value),
    stiffness: Number($("sp-stiffness").value),
    damping: Number($("sp-damping").value),
    dt: Number($("sp-dt").value),
    steps: Number($("sp-steps").value),
    gravity: Number($("sp-gravity").value),
    pinDrop: $("sp-pin-drop").checked,
    lines: $("sp-lines").checked,
    lineCount: Number($("sp-lines-n").value),
    lineColor: $("sp-lines-color").value,
  });
  let p = params();

  // -------------------------------------------------------------------------
  // The simulation
  // -------------------------------------------------------------------------

  function decode({ shape: [n, d], data }) {
    const text = atob(data);
    const bytes = new Uint8Array(text.length);
    for (let k = 0; k < text.length; k++) bytes[k] = text.charCodeAt(k);
    return { n, d, values: new Float32Array(bytes.buffer) };
  }

  const keyOf = (result) => [result.model, result.dataset, result.prep, result.indices.join(",")].join("|");

  function createSim(result) {
    const { n, d, values } = decode(result.spring_vectors);
    // centered, then length 1: the similarity of two images is the dot product of their rows
    const mean = new Float64Array(d);
    for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) mean[k] += values[i * d + k] / n;
    const unit = new Float32Array(n * d), norm = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let k = 0; k < d; k++) sum += (values[i * d + k] - mean[k]) ** 2;
      norm[i] = Math.sqrt(sum);
      for (let k = 0; k < d; k++) unit[i * d + k] = (values[i * d + k] - mean[k]) / (norm[i] || 1);
    }
    const s = {
      key: keyOf(result), ids: result.indices, n, d, unit, norm,
      pos: new Float64Array(2 * n), vel: new Float64Array(2 * n), force: new Float64Array(2 * n),
      fixed: new Uint8Array(n), steps: 0, cursor: 0,
    };
    // the average distance of two images, from a sample of pairs (for "vector distance")
    let total = 0;
    const samples = Math.min(2000, n * (n - 1));
    for (let t = 0; t < samples; t++) {
      const i = Math.floor(Math.random() * n), j = randomOther(i, n);
      total += vectorDistance(s, i, j, similarity(s, i, j));
    }
    s.meanDistance = total / Math.max(1, samples) || 1;
    setPairs(s);
    return s;
  }

  function similarity(s, i, j) {
    const { unit, d } = s;
    let dot = 0;
    for (let k = 0, a = i * d, b = j * d; k < d; k++) dot += unit[a + k] * unit[b + k];
    return dot;
  }

  const vectorDistance = (s, i, j, v) =>
    Math.sqrt(Math.max(0, s.norm[i] ** 2 + s.norm[j] ** 2 - 2 * v * s.norm[i] * s.norm[j]));

  // any image but i
  function randomOther(i, n) {
    const j = Math.floor(Math.random() * (n - 1));
    return j >= i ? j + 1 : j;
  }

  // (Re)starts finding the pairs: the cache of all similarities, or each
  // image's neighbours. Positions and speeds stay.
  function setPairs(s) {
    s.mode = p.pairs;
    s.k = Math.max(1, Math.min(p.k, s.n - 1));
    s.matrix = s.nbr = s.nbrSim = s.top = s.lines = null;
    s.prepared = 0;
    s.error = null;
    if (s.n < 2) {
      s.error = "springs need two images or more";
    } else if (s.mode === "all") {
      if (s.n > ALL_MAX) s.error = `all pairs: ${s.n} images are too many (at most ${ALL_MAX}), pick other pairs`;
      else s.matrix = new Float32Array(s.n * s.n);
    } else {
      s.nbr = new Int32Array(s.n * s.k);
      s.nbrSim = new Float32Array(s.n * s.k).fill(-Infinity);
      if (s.mode === "dynamic") {
        // k different random neighbours each, to start with
        for (let i = 0; i < s.n; i++) {
          const base = i * s.k;
          for (let t = 0; t < s.k; t++) {
            let j;
            do j = randomOther(i, s.n); while (s.nbr.subarray(base, base + t).includes(j));
            s.nbr[base + t] = j;
            s.nbrSim[base + t] = similarity(s, i, j);
          }
        }
        s.prepared = s.n;
      }
    }
    s.ready = !s.error && s.prepared >= s.n;
  }

  // A few more rows of the cache or of the k most similar, until the deadline
  function prepare(s, deadline) {
    const { n, k } = s;
    while (s.prepared < n && performance.now() < deadline) {
      const i = s.prepared++;
      if (s.mode === "all") {
        s.matrix[i * n + i] = 1;
        for (let j = i + 1; j < n; j++) s.matrix[i * n + j] = s.matrix[j * n + i] = similarity(s, i, j);
        continue;
      }
      // knn: kept sorted, most similar first
      let filled = 0;
      for (let j = 0; j < n; j++) {
        if (j !== i) filled = insertTop(s.nbr, s.nbrSim, i * k, k, filled, j, similarity(s, i, j));
      }
    }
    s.ready = s.prepared >= n;
  }

  // j, of similarity v, into the list of `size` places at base, kept sorted
  // most similar first, `filled` places taken so far. -> places taken now
  function insertTop(list, sims, base, size, filled, j, v) {
    if (filled === size && v <= sims[base + size - 1]) return filled;
    let t = Math.min(filled, size - 1);
    while (t > 0 && sims[base + t - 1] < v) {
      sims[base + t] = sims[base + t - 1];
      list[base + t] = list[base + t - 1];
      t--;
    }
    sims[base + t] = v;
    list[base + t] = j;
    return Math.min(size, filled + 1);
  }

  // dynamic: j takes the place of i's least similar neighbour if it is more similar
  function consider(s, i, j, v) {
    const { nbr, nbrSim, k } = s;
    const base = i * k;
    let worst = base;
    for (let t = base; t < base + k; t++) {
      if (nbr[t] === j) return;
      if (nbrSim[t] < nbrSim[worst]) worst = t;
    }
    if (v > nbrSim[worst]) {
      nbr[worst] = j;
      nbrSim[worst] = v;
    }
  }

  // rest length of the spring between i and j, of similarity v: never under
  // the spacing, so even twins don't end up on top of each other
  function restLength(s) {
    const { L, spacing } = p;
    if (p.length === "equal") return () => Math.max(L, spacing);
    if (p.length === "inverse") return (i, j, v) => Math.max(spacing, L * 0.5 / Math.max(0.05, (1 + v) / 2));
    if (p.length === "linear") return (i, j, v) => Math.max(spacing, L * (1 - v));
    const scale = L / s.meanDistance;
    return (i, j, v) => Math.max(spacing, scale * vectorDistance(s, i, j, v));
  }

  function step(s, rest) {
    const { n, pos, vel, force } = s;
    force.fill(0);
    const { cutoff, repulsion } = p;
    const width = Math.max(0.01, p.transition), reach2 = p.reach * p.reach;
    const pull = (i, j, v, w) => {
      const dx = pos[2 * j] - pos[2 * i], dy = pos[2 * j + 1] - pos[2 * i + 1];
      const dist = Math.sqrt(dx * dx + dy * dy) || 1e-9;
      // how much of a spring (1) or of a push (0) this pair is
      const t = Math.min(1, Math.max(0, (v - cutoff) / width + 0.5));
      const related = t * t * (3 - 2 * t);
      let f = 0;
      if (related > 0) f += related * (dist - rest(i, j, v));
      if (related < 1) f -= (1 - related) * repulsion * reach2 / (dist * dist + reach2);
      f *= w / dist;
      force[2 * i] += f * dx;
      force[2 * i + 1] += f * dy;
      force[2 * j] -= f * dx;
      force[2 * j + 1] -= f * dy;
    };
    if (s.mode === "all") {
      const w = 1 / (n - 1), matrix = s.matrix;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) pull(i, j, matrix[i * n + j], w);
      }
    } else {
      const k = s.k, m = Math.min(p.m, n - 1);
      const w = 1 / k, random = m ? p.weight / m : 0;
      for (let i = 0; i < n; i++) {
        for (let t = i * k; t < i * k + k; t++) pull(i, s.nbr[t], s.nbrSim[t], w);
        for (let r = 0; r < m; r++) {
          const j = randomOther(i, n);
          const v = similarity(s, i, j);
          pull(i, j, v, random);
          if (s.mode === "dynamic") consider(s, i, j, v);
        }
      }
    }
    const { stiffness, damping, dt, gravity } = p;
    const keep = 1 - damping;
    for (let a = 0; a < 2 * n; a++) {
      if (s.fixed[a >> 1]) {
        vel[a] = 0;
        continue;
      }
      vel[a] = (vel[a] + (stiffness * force[a] - gravity * (pos[a] - 0.5)) * dt) * keep;
      pos[a] += Math.max(-MAX_MOVE, Math.min(MAX_MOVE, vel[a] * dt));
    }
    s.steps++;
  }

  // dynamic: more random pairs compared, round the images in turn
  function backgroundTests(s, count) {
    for (let t = 0; t < count; t++) {
      const i = s.cursor;
      s.cursor = (i + 1) % s.n;
      const j = randomOther(i, s.n);
      consider(s, i, j, similarity(s, i, j));
    }
  }

  // The pair lines: image ids, two per line. Every image's `count` most
  // similar partners (all: from its row of the cache, sorted once; knn: its
  // neighbours, kept sorted; dynamic: its tracked ones, sorted now), each pair
  // once. Made again at most once per step.
  function pairLines(s) {
    const count = Math.min(p.lineCount, s.mode === "all" ? s.n - 1 : s.k);
    const key = `${s.steps}|${s.mode}|${count}|${s.k}`;
    if (s.lines && s.lines.key === key) return s.lines.ids;
    const { n, ids } = s;
    const seen = new Set(), out = [];
    const add = (i, j) => {
      const pair = i < j ? i * n + j : j * n + i;
      if (seen.has(pair)) return;
      seen.add(pair);
      out.push(ids[i], ids[j]);
    };
    if (s.mode === "all") {
      if (!s.top || s.top.count < count) s.top = topOfRows(s, count);
      for (let i = 0; i < n; i++) for (let t = 0; t < count; t++) add(i, s.top.rows[i * s.top.count + t]);
    } else {
      const order = Array.from({ length: s.k }, (_, t) => t);
      for (let i = 0; i < n; i++) {
        const base = i * s.k;
        if (s.mode === "dynamic") order.sort((a, b) => s.nbrSim[base + b] - s.nbrSim[base + a]);
        for (let t = 0; t < count; t++) add(i, s.nbr[base + order[t]]);
      }
    }
    s.lines = { key, ids: out };
    return out;
  }

  // all pairs: every image's `count` most similar, from the cache
  function topOfRows(s, count) {
    const { n, matrix } = s;
    const rows = new Int32Array(n * count), sims = new Float32Array(n * count);
    for (let i = 0; i < n; i++) {
      let filled = 0;
      for (let j = 0; j < n; j++) {
        if (j !== i) filled = insertTop(rows, sims, i * count, count, filled, j, matrix[i * n + j]);
      }
    }
    return { count, rows };
  }

  // -------------------------------------------------------------------------
  // Frames
  // -------------------------------------------------------------------------

  const info = { time: 0, steps: 0, rate: 0 };

  function tick() {
    frame = null;
    const s = sim;
    if (!s) return;
    if (!s.ready && !s.error) {
      prepare(s, performance.now() + PREPARE_MS);
    } else if (running && s.ready) {
      const rest = restLength(s);
      for (let k = 0; k < p.steps; k++) step(s, rest);
      if (s.mode === "dynamic") backgroundTests(s, p.tests);
      place(s);
    }
    showInfo(s);
    if ((running && s.ready) || (!s.ready && !s.error)) frame = requestAnimationFrame(tick);
  }

  function schedule() {
    if (frame === null && sim) frame = requestAnimationFrame(tick);
  }

  // the simulation's positions onto the map
  function place(s) {
    s.ids.forEach((i, row) => {
      const c = state.coords.get(i);
      c[0] = s.pos[2 * row];
      c[1] = s.pos[2 * row + 1];
    });
    quantizeCoords();
    instantly(layout);
  }

  // the line under the buttons, about 3 times a second (now: at once)
  function showInfo(s, now = false) {
    const time = performance.now();
    if (!now && time - info.time < 300 && s.ready && !s.error) return;
    if (time - info.time >= 300) {
      info.rate = Math.round((s.steps - info.steps) * 1000 / (time - info.time));
      info.time = time;
      info.steps = s.steps;
    }
    let text;
    if (s.error) text = s.error;
    else if (!s.ready) text = `${s.mode === "all" ? "similarities of all pairs" : `${s.k} most similar of every image`}: `
      + `${Math.round(100 * s.prepared / s.n)}%`;
    else {
      text = `${s.n} images, ${s.d} numbers each, step ${s.steps}` + (running ? ` (${info.rate}/s)` : ", paused");
      if (s.nbrSim) {
        let sum = 0;
        for (const v of s.nbrSim) sum += v;
        text += `, neighbours' mean similarity ${(sum / s.nbrSim.length).toFixed(3)}`;
      }
    }
    if (pinned.size) text += `, ${pinned.size} pinned`;
    $("sp-info").textContent = text;
  }

  // -------------------------------------------------------------------------
  // Pinning, selecting, dragging
  // -------------------------------------------------------------------------

  function updateFixed() {
    if (!sim) return;
    sim.fixed.fill(0);
    for (const i of pinned) {
      const row = state.rows.get(i);
      if (row !== undefined) sim.fixed[row] = 1;
    }
    if (held) sim.fixed[held.row] = 1;
  }

  function updateButtons() {
    const selected = state.selected;
    $("sp-pin").disabled = !sim || selected === null;
    $("sp-pin").textContent = selected !== null && pinned.has(selected) ? "Unpin selected" : "Pin selected";
    $("sp-unpin").disabled = !pinned.size;
    $("sp-run").textContent = running ? "Pause" : "Run";
  }

  // pins changed: the map shows them
  function refresh() {
    updateFixed();
    updateButtons();
    if (sim) showInfo(sim, true);
    layout();
  }

  function togglePin(i) {
    if (i === null) return;
    if (pinned.has(i)) pinned.delete(i);
    else pinned.add(i);
    refresh();
  }

  // map pixels (mouse) -> map coordinates, the inverse of toScreen
  function fromScreen(clientX, clientY) {
    const r = map.getBoundingClientRect();
    const side = mapWidth();
    const { scale, tx, ty } = state.view;
    return [((clientX - r.left - tx) / (side * scale) - MARGIN) / (1 - 2 * MARGIN),
            ((clientY - r.top - ty) / (side * scale) - MARGIN) / (1 - 2 * MARGIN)];
  }

  // pointerdown on the map: true when it takes an image (then the map doesn't pan)
  function grab(event) {
    if (!sim || state.picking) return false;
    const mark = markAt(event.clientX, event.clientY);
    if (!mark || mark.i === undefined || !state.rows.has(mark.i)) return false;
    held = { i: mark.i, row: state.rows.get(mark.i), x: event.clientX, y: event.clientY, moved: false };
    updateFixed();
    map.setPointerCapture(event.pointerId);
    return true;
  }

  // pointermove: false (no image held), "held" (not moved yet) or "moved"
  function drag(event) {
    if (!held) return false;
    if (!held.moved && Math.hypot(event.clientX - held.x, event.clientY - held.y) < 3) return "held";
    held.moved = true;
    const [x, y] = fromScreen(event.clientX, event.clientY);
    sim.pos[2 * held.row] = x;
    sim.pos[2 * held.row + 1] = y;
    if (!running || !sim.ready) place(sim);     // otherwise the next frame shows it
    return "moved";
  }

  function release() {
    if (!held) return;
    if (held.moved && p.pinDrop) pinned.add(held.i);
    held = null;
    refresh();
  }

  // -------------------------------------------------------------------------
  // A new result
  // -------------------------------------------------------------------------

  function stop() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    sim = null;
    held = null;
    pinned.clear();
    map.classList.remove("springs");
    updateButtons();
  }

  // Called by showResult before it draws the new map: a simulation of other
  // images (or another dataset's pins) must not draw on it
  function beforeResult(result) {
    if (state.result && baseDataset(state.result.dataset) !== baseDataset(result.dataset)) pinned.clear();
    if (sim && sim.key !== keyOf(result)) {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      sim = null;
      held = null;
    }
  }

  // Called by showResult: the images are where the map has put them (where
  // they were, new ones on the PCA map). The same vectors keep the pairs
  // found so far and the speeds.
  function onResult(result) {
    if (result.projection !== "spring" || !result.spring_vectors) {
      stop();
      return;
    }
    map.classList.add("springs");
    const key = keyOf(result);
    if (!sim || sim.key !== key) sim = createSim(result);
    result.indices.forEach((i, row) => {
      const [x, y] = state.coords.get(i);
      sim.pos[2 * row] = x;
      sim.pos[2 * row + 1] = y;
    });
    for (const i of pinned) if (!state.rows.has(i)) pinned.delete(i);
    refresh();
    schedule();
  }

  // -------------------------------------------------------------------------
  // Controls
  // -------------------------------------------------------------------------

  function updateSpringControls() {
    for (const id of SLIDERS) $(`${id}-out`).textContent = $(id).value;
    const pairs = $("sp-pairs").value;
    $("sp-m-row").hidden = $("sp-weight-row").hidden = pairs === "all";
    $("sp-k-row").hidden = pairs === "all";
    $("sp-tests-row").hidden = pairs !== "dynamic";
    $("sp-lines-n-row").hidden = $("sp-lines-color-row").hidden = !$("sp-lines").checked;
  }

  for (const id of [...SLIDERS, "sp-length", "sp-tests", "sp-pin-drop"]) {
    $(id).addEventListener("input", () => {
      p = params();
      updateSpringControls();
    });
  }
  // the lines are drawn again at once (also while paused)
  for (const id of ["sp-lines", "sp-lines-n", "sp-lines-color"]) {
    $(id).addEventListener("input", () => {
      p = params();
      updateSpringControls();
      if (sim) layout();
    });
  }
  // other pairs: found again (the slider's neighbours only once it is let go)
  for (const id of ["sp-pairs", "sp-k"]) {
    $(id).addEventListener("change", () => {
      p = params();
      updateSpringControls();
      if (!sim) return;
      setPairs(sim);
      showInfo(sim, true);
      schedule();
    });
  }
  $("sp-run").addEventListener("click", () => {
    running = !running;
    updateButtons();
    schedule();
  });
  $("sp-shake").addEventListener("click", () => {
    if (!sim) return;
    for (let a = 0; a < sim.vel.length; a++) if (!sim.fixed[a >> 1]) sim.vel[a] += (Math.random() - 0.5) * 0.1;
    schedule();
  });
  $("sp-pin").addEventListener("click", () => togglePin(state.selected));
  $("sp-unpin").addEventListener("click", () => {
    pinned.clear();
    refresh();
  });
  // P: pin / unpin the selected image (not while typing)
  document.addEventListener("keydown", (event) => {
    if (!sim || state.selected === null || event.key.toLowerCase() !== "p" || event.ctrlKey || event.metaKey) return;
    if (event.target instanceof Element && event.target.closest("input, select, textarea")) return;
    togglePin(state.selected);
  });
  updateSpringControls();
  updateButtons();

  return {
    beforeResult, onResult, grab, drag, release,
    selectionChanged: updateButtons,        // the pin button follows the selection (embedding_map.js)
    isPinned: (i) => sim !== null && pinned.has(i),
    // the pair lines to draw: { ids (two image ids per line), color }, or null
    lines: () => (sim && sim.ready && p.lines ? { ids: pairLines(sim), color: p.lineColor } : null),
  };
})();
