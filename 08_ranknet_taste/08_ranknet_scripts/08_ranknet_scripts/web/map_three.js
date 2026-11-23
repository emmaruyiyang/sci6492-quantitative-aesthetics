// three.js renderer of the Embedding Map (Display > renderer).
//
// embedding_map.js describes every mark of the map the way the stylesheet
// draws an .item (looks(): center and size in pixels, border, ring of map
// color, outline, thumbnail or flat color...) and hands the list to draw().
// Each mark is one quad: its fragment shader draws the box, round or square,
// in the same layers as the browser. Marks glide from where they are to where
// draw() puts them, like the .item transitions of the HTML renderer, and are
// painted in the order of their z, then of the list (as z-index, then the
// order of the page).
//
// Loaded as a module on first use (same three.js as raup_shell_map).

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

// Colors and thumbnails go to the screen as they are, like in the page
THREE.ColorManagement.enabled = false;

const VERTEX = /* glsl */ `
uniform float uHalf;
varying vec2 vPx;
void main() {
  vPx = position.xy * uHalf;      // pixels from the center (y up)
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

// Outward from the center: fill (thumbnail, text on ink, or flat color),
// border, then outside the box the outline (over) the ring of map color.
const FRAGMENT = /* glsl */ `
uniform float uSize, uBorder, uRing, uOutline, uOffset, uRound, uOpacity, uHasMap, uAspect;
uniform vec3 uFill, uColor, uBg, uInk;
uniform sampler2D uMap;
varying vec2 vPx;
void main() {
  float r = 0.5 * uSize;
  float d = uRound > 0.5 ? length(vPx) - r : max(abs(vPx.x), abs(vPx.y)) - r;
  float edge = max(uRing, uOutline > 0.0 ? uOffset + uOutline : 0.0);
  float soft = uRound > 0.5 ? 0.5 : 0.0;     // half a pixel of antialiasing around round marks
  if (d > edge + soft) discard;
  vec3 c;
  if (d > 0.0) {
    if (uOutline > 0.0 && d >= uOffset && d <= uOffset + uOutline + soft) c = uInk;
    else if (d <= uRing + soft) c = uBg;
    else discard;
  } else if (d > -uBorder) {
    c = uColor;
  } else {
    c = uFill;
    if (uHasMap > 0.5) {
      // background-size: cover, over the box inside the border
      vec2 uv = vPx / max(1.0, uSize - 2.0 * uBorder);
      uv *= uAspect >= 1.0 ? vec2(1.0 / uAspect, 1.0) : vec2(1.0, uAspect);
      vec4 t = texture2D(uMap, uv + 0.5);
      c = mix(c, t.rgb, t.a);
    }
  }
  float aa = uRound > 0.5 ? clamp(edge - d + soft, 0.0, 1.0) : 1.0;
  gl_FragColor = vec4(c, uOpacity * aa);
}`;

const SIZE_MS = 300;      // width, height and opacity transitions of .item

// CSS "ease": cubic-bezier(0.25, 0.1, 0.25, 1)
function ease(t) {
  const bez = (u, p1, p2) => 3 * (1 - u) * (1 - u) * u * p1 + 3 * (1 - u) * u * u * p2 + u * u * u;
  const slope = (u, p1, p2) => 3 * (1 - u) * (1 - u) * p1 + 6 * (1 - u) * u * (p2 - p1) + 3 * u * u * (1 - p2);
  let u = t;
  for (let k = 0; k < 8; k++) {
    const dx = slope(u, 0.25, 0.25);
    if (Math.abs(dx) < 1e-6) break;
    u = Math.min(1, Math.max(0, u - (bez(u, 0.25, 0.25) - t) / dx));
  }
  return bez(u, 0.1, 1);
}

// One animated number: from where it was to where it goes
const tween = (value) => ({ from: value, to: value, t0: 0, ms: 0 });
const valueAt = (tw, now) => (tw.ms ? tw.from + (tw.to - tw.from) * ease(Math.min(1, (now - tw.t0) / tw.ms)) : tw.to);
const moving = (tw, now) => tw.ms > 0 && now < tw.t0 + tw.ms;
function retarget(tw, to, ms, now) {
  if (tw.to === to && (ms || !tw.ms)) return;
  tw.from = ms ? valueAt(tw, now) : to;
  tw.to = to;
  tw.t0 = now;
  tw.ms = ms;
}

// A class anchor's name, white on the ink fill, wrapped like the HTML one
// (bold, centered, broken anywhere when a word is too long)
function textTexture(text, fontRatio, family) {
  const n = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = n;
  const ctx = canvas.getContext("2d");
  const size = fontRatio * n;
  ctx.font = `700 ${size}px ${family}`;
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const width = n * 0.94;
  const lines = [];
  let line = "";
  const add = (piece, separator) => {
    const joined = line ? line + separator + piece : piece;
    if (line && ctx.measureText(joined).width > width) {
      lines.push(line);
      line = piece;
    } else {
      line = joined;
    }
  };
  for (const word of text.split(/\s+/).filter(Boolean)) {
    if (ctx.measureText(word).width <= width) add(word, " ");
    else [...word].forEach((ch, k) => add(ch, k ? "" : " "));
  }
  if (line) lines.push(line);
  const height = size * 1.05;
  lines.forEach((l, k) => ctx.fillText(l, n / 2, n / 2 + (k - (lines.length - 1) / 2) * height));
  return new THREE.CanvasTexture(canvas);
}

export function createThreeMap(container) {
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setClearColor(0x000000, 0);     // the map's own background shows through
  renderer.domElement.className = "three";
  container.prepend(renderer.domElement);   // under #empty

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(0, 1, 0, -1, -10, 10);   // pixels, y = -top
  const quad = new THREE.PlaneGeometry(2, 2);
  const loader = new THREE.TextureLoader();
  const family = getComputedStyle(document.body).fontFamily;
  const moveMs = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--move")) * 1000 || 900;
  // shared by every mark
  const shared = { uBg: { value: new THREE.Color("#ececec") }, uInk: { value: new THREE.Color("#000000") } };

  const entries = new Map();      // key -> one mark
  let line = null;                // the interpolation line
  let segments = null;            // spring forces: the pair lines, 1 pixel wide
  let frame = null;
  let size = { w: 0, h: 0, ratio: 0 };

  function resize() {
    const w = container.clientWidth, h = container.clientHeight, ratio = window.devicePixelRatio || 1;
    if (w === size.w && h === size.h && ratio === size.ratio) return;
    size = { w, h, ratio };
    renderer.setPixelRatio(ratio);
    renderer.setSize(w, h, false);
    camera.right = w;
    camera.bottom = -h;
    camera.updateProjectionMatrix();
  }

  function createEntry(look) {
    const start = { ...look, ...look.start };
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        ...shared,
        uHalf: { value: 1 }, uSize: { value: 1 }, uBorder: { value: 0 }, uRing: { value: 0 },
        uOutline: { value: 0 }, uOffset: { value: 0 }, uRound: { value: 0 }, uOpacity: { value: 1 },
        uHasMap: { value: 0 }, uAspect: { value: 1 }, uMap: { value: null },
        uFill: { value: new THREE.Color() }, uColor: { value: new THREE.Color() },
      },
    });
    const mesh = new THREE.Mesh(quad, material);
    mesh.frustumCulled = false;
    scene.add(mesh);
    return {
      mesh, material, look,
      x: tween(start.x), y: tween(start.y), s: tween(start.s), opacity: tween(start.opacity),
      source: null, texture: null,
    };
  }

  function removeEntry(entry) {
    scene.remove(entry.mesh);
    entry.material.dispose();
    entry.texture?.dispose();
    entry.removed = true;
  }

  // the thumbnail (a URL) or the anchor's name: loaded once per change
  function setSource(entry, look) {
    const source = look.kind === "anchor" ? `text:${look.text}|${(look.fontSize / look.s).toFixed(3)}`
      : look.kind === "image" ? look.src : null;
    if (entry.source === source) return;
    entry.source = source;
    entry.texture?.dispose();
    entry.texture = null;
    const u = entry.material.uniforms;
    u.uHasMap.value = 0;
    if (!source) return;
    const use = (texture, aspect) => {
      entry.texture = texture;
      u.uMap.value = texture;
      u.uAspect.value = aspect;
      u.uHasMap.value = 1;
      requestRender();
    };
    if (look.kind === "anchor") {
      use(textTexture(look.text, look.fontSize / look.s, family), 1);
      return;
    }
    loader.load(source, (texture) => {
      if (entry.removed || entry.source !== source) {
        texture.dispose();
        return;
      }
      use(texture, texture.image.width / texture.image.height || 1);
    });
  }

  // The line A -> interpolated images -> B, 2 pixels wide: one quad per segment
  function setLine(look, order) {
    if (!line) {
      line = new THREE.Mesh(new THREE.BufferGeometry(),
        new THREE.MeshBasicMaterial({ transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide }));
      line.frustumCulled = false;
      scene.add(line);
    }
    const vertices = [];
    for (let k = 1; k < look.points.length; k++) {
      const [x1, y1] = look.points[k - 1], [x2, y2] = look.points[k];
      const length = Math.hypot(x2 - x1, y2 - y1) || 1;
      const nx = -(y2 - y1) / length, ny = (x2 - x1) / length;     // half the width across
      const a = [x1 + nx, -(y1 + ny)], b = [x1 - nx, -(y1 - ny)];
      const c = [x2 + nx, -(y2 + ny)], d = [x2 - nx, -(y2 - ny)];
      for (const [x, y] of [a, b, c, b, d, c]) vertices.push(x, y, 0);
    }
    line.geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    line.material.color.setStyle(look.color);
    line.renderOrder = look.z * 1e6 + order;
    line.visible = true;
  }

  function draw(looks, { instant = false, background, ink } = {}) {
    const now = performance.now();
    if (background) shared.uBg.value.setStyle(background);
    if (ink) shared.uInk.value.setStyle(ink);
    const seen = new Set();
    let hasLine = false, hasSegments = false;
    looks.forEach((look, order) => {
      if (look.kind === "line") {
        setLine(look, order);
        hasLine = true;
        return;
      }
      if (look.kind === "segments") {
        setSegments(look, order);
        hasSegments = true;
        return;
      }
      seen.add(look.key);
      let entry = entries.get(look.key);
      if (!entry) entries.set(look.key, entry = createEntry(look));
      entry.look = look;
      retarget(entry.x, look.x, instant ? 0 : moveMs, now);
      retarget(entry.y, look.y, instant ? 0 : moveMs, now);
      retarget(entry.s, look.s, instant ? 0 : SIZE_MS, now);
      retarget(entry.opacity, look.opacity, SIZE_MS, now);
      const u = entry.material.uniforms;
      u.uBorder.value = look.border;
      u.uRing.value = look.ring;
      u.uOutline.value = look.outline;
      u.uOffset.value = look.outlineOffset || 0;
      u.uRound.value = look.round ? 1 : 0;
      u.uColor.value.setStyle(look.color);
      u.uFill.value.setStyle(look.fill);
      setSource(entry, look);
      entry.mesh.renderOrder = look.z * 1e6 + order;
    });
    for (const [key, entry] of entries) {
      if (!seen.has(key)) {
        removeEntry(entry);
        entries.delete(key);
      }
    }
    if (line && !hasLine) line.visible = false;
    if (segments && !hasSegments) segments.visible = false;
    requestRender();
  }

  // look.points: x1 y1 x2 y2 per line, in pixels
  function setSegments(look, order) {
    if (!segments) {
      segments = new THREE.LineSegments(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ transparent: true, depthTest: false, depthWrite: false }));
      segments.frustumCulled = false;
      scene.add(segments);
    }
    const n = look.points.length / 2;
    const vertices = new Float32Array(n * 3);
    for (let k = 0; k < n; k++) {
      vertices[3 * k] = look.points[2 * k];
      vertices[3 * k + 1] = -look.points[2 * k + 1];
    }
    segments.geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
    segments.material.color.setStyle(look.color);
    segments.renderOrder = look.z * 1e6 + order;
    segments.visible = true;
  }

  function requestRender() {
    if (frame === null) frame = requestAnimationFrame(render);
  }

  function render() {
    frame = null;
    const now = performance.now();
    resize();
    let animating = false;
    for (const entry of entries.values()) {
      const x = valueAt(entry.x, now), y = valueAt(entry.y, now), s = valueAt(entry.s, now);
      const look = entry.look;
      const pad = Math.max(look.ring, look.outline ? (look.outlineOffset || 0) + look.outline : 0) + 1;
      const half = s / 2 + pad;
      entry.mesh.position.set(x, -y, 0);
      entry.mesh.scale.set(half, half, 1);
      const u = entry.material.uniforms;
      u.uHalf.value = half;
      u.uSize.value = s;
      u.uOpacity.value = valueAt(entry.opacity, now);
      animating ||= moving(entry.x, now) || moving(entry.y, now) || moving(entry.s, now) || moving(entry.opacity, now);
    }
    renderer.render(scene, camera);
    if (animating) requestRender();
  }

  // The topmost mark at (x, y), pixels in the map, where it is drawn now
  function hit(x, y) {
    const now = performance.now();
    let best = null;
    for (const entry of entries.values()) {
      const s = valueAt(entry.s, now);
      const dx = x - valueAt(entry.x, now), dy = y - valueAt(entry.y, now);
      const inside = entry.look.round ? Math.hypot(dx, dy) <= s / 2 : Math.max(Math.abs(dx), Math.abs(dy)) <= s / 2;
      if (inside && (!best || entry.mesh.renderOrder >= best.mesh.renderOrder)) best = entry;
    }
    return best ? best.look.ref : null;
  }

  function clear() {
    for (const entry of entries.values()) removeEntry(entry);
    entries.clear();
  }

  function dispose() {
    if (frame !== null) cancelAnimationFrame(frame);
    clear();
    for (const object of [line, segments]) {
      object?.geometry.dispose();
      object?.material.dispose();
    }
    quad.dispose();
    renderer.dispose();
    renderer.domElement.remove();
  }

  return { draw, hit, clear, dispose };
}
