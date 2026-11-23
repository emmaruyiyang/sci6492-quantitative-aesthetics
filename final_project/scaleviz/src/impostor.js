// P3 - impostors.
//
// A chunk that covers 8 pixels does not deserve 1,500 points and a draw call
// of its own. Below a threshold we stop drawing the chunk and draw a BAKED
// PICTURE of one instead: a few variants of each level are rendered once into
// an atlas, and every distant chunk becomes one textured point sprite. All of
// them go out in a single draw call whose positions are rewritten each frame.
//
// This is what the MIN_CHUNK_PX hook in cascade.js was reserved for. Before
// it, chunks under the threshold were skipped outright and chunks past the
// per-level budget were dropped, so whole bands of the zoom were missing
// structure that the data actually had.
//
// Impostors are STATISTICAL, not exact: a chunk is drawn as one of N baked
// variants chosen by hash, not as a picture of itself. At <= ~12 px that is
// not a distinction anyone can see.

import * as THREE from 'three';

const TILE = 128;
// Level 0 bakes bigger. A cell chunk's smallest cells are ~0.7% of the chunk
// across, so at a 128px tile they land under uMinPx INSIDE THE BAKE and get
// the same k^2 dimming that source-over under-composites - which made every
// cell impostor ~9% darker than the chunk it stood in for, and that step was
// measurable in mean frame luminance as you zoomed through the threshold.
// Level 1 bakes big for the same reason level 0 does, and more so now that
// its mark is a cell: at a 128 tile a river's cells land 1.4 px across inside
// the bake, source-over saturates the channel into a solid bar, and the
// impostor came out 1.42x BRIGHTER than the granular chunk it stood in for.
const TILE_BY_LEVEL = [288, 512, 128];
const COLS = 4;
const ROWS = 3;
export const VARIANTS = COLS * ROWS;
export function tileForLevel(i) { return TILE_BY_LEVEL[i] || TILE; }

// INSTANCED QUADS, not point sprites.
//
// gl_PointSize is capped by the driver - ALIASED_POINT_SIZE_RANGE is 511 on
// this machine, and the shader used to clamp at 480 to stay inside it. That
// is fine while an impostor stands in for something small, which was the
// whole premise. It stopped being true the moment the river's chunk budget
// started overflowing: at u=2.6 a river chunk is 771 px, the 48 nearest get
// full detail and the next 32 become impostors - which were then drawn at
// 480 px, covering 39% of the area they were standing in for. Forced
// measurement across the level: impostor/full mean luminance 0.30 at u=2.2,
// 0.53 at u=2.4, 1.05 only once chunks fall under the cap at u>=2.6.
//
// A quad has no such limit. Four vertices per impostor instead of one, still
// one draw call, and the sprite can be any size.
const IMP_VERT = /* glsl */`
  attribute vec3 iPos;         // instance centre, in the group's local space
  attribute float size;        // sprite size in PIXELS
  attribute float variant;
  attribute float orient;
  attribute vec3 tint;
  attribute float gain;
  varying vec2 vUv;
  varying float vVariant;
  varying float vOrient;
  varying float vFade;
  varying vec3 vTint;
  varying float vGain;
  uniform float uMinPx;
  uniform vec2 uViewport;      // drawing buffer size, device pixels
  void main() {
    vec4 c = projectionMatrix * modelViewMatrix * vec4(iPos, 1.0);
    float px = size;
    float a = 1.0;
    if (px < uMinPx) { float k = px / uMinPx; a = k * k; px = uMinPx; }
    // position holds the unit quad's corner in [-0.5, 0.5]
    c.xy += position.xy * px / uViewport * 2.0 * c.w;
    gl_Position = c;
    // gl_PointCoord runs +y DOWN; the quad's y runs up, so flip to keep
    // every orientation and every baked tile reading exactly as before.
    vUv = vec2(position.x + 0.5, 0.5 - position.y);
    vVariant = variant;
    vOrient = orient;
    vTint = tint;
    vGain = gain;
    vFade = a;
  }
`;

const IMP_FRAG = /* glsl */`
  uniform sampler2D uAtlas;
  uniform float uOpacity;
  uniform float uGain;
  uniform vec2 uTile;          // 1/COLS, 1/ROWS
  varying vec2 vUv;
  varying float vVariant;
  varying float vOrient;
  varying float vFade;
  varying vec3 vTint;
  varying float vGain;
  void main() {
    // 90-degree rotations and a mirror are exact and cost nothing, and they
    // turn ${COLS * ROWS} baked tiles into 8x as many apparent ones. Without this the
    // repeat is obvious wherever impostors are more than ~20px across.
    vec2 p = vUv;
    if (vOrient >= 4.0) p.x = 1.0 - p.x;
    float o = mod(vOrient, 4.0);
    vec2 q = p - 0.5;
    if (o == 1.0) q = vec2(-q.y, q.x);
    else if (o == 2.0) q = -q;
    else if (o == 3.0) q = vec2(q.y, -q.x);
    p = q + 0.5;

    float v = vVariant;
    float cx = mod(v, ${COLS}.0);
    float cy = floor(v / ${COLS}.0);
    vec2 uv = (vec2(cx, cy) + p) * uTile;
    vec4 t = texture2D(uAtlas, uv);
  #ifdef PREMULT
    // the bake already holds premultiplied rgb and real coverage in a
    // vGain scales the whole premultiplied result, rgb AND coverage: the
    // correction is "this stand-in puts too much cell on screen for the node
    // it is standing in for", which is a coverage error, not a grade.
    float k = uOpacity * vFade * vGain;
    if (t.a * k < 0.003) discard;
    gl_FragColor = vec4(t.rgb * uGain * vTint, t.a) * k;
  #else
    if (t.a <= 0.001 && dot(t.rgb, t.rgb) <= 0.000001) discard;
    gl_FragColor = vec4(t.rgb * uGain * vTint, 1.0) * uOpacity * vFade;
  #endif
  }
`;

// Render `variants` sample chunks of one level into a tiled atlas, using the
// level's own material so the baked picture is exactly what the real thing
// looks like at this size.
export function bakeAtlas(renderer, material, buildVariant, extent = 1, tile = TILE, clamp = false) {
  const TILE = tile;
  const rt = new THREE.WebGLRenderTarget(COLS * TILE, ROWS * TILE, {
    // Additive sums must not clamp at 1.0 - but a PREMULTIPLIED level is
    // different. Its fragment writes rgb = albedo*a with albedo free to
    // exceed 1, and on screen the 8-bit framebuffer clips those highlights.
    // A float bake keeps them, mip-filtering then spreads them over the
    // sprite, and the impostor comes out brighter than the thing it stands
    // in for everywhere it overlaps. Measured on the river: 1.8x.
    type: clamp ? THREE.UnsignedByteType : THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearMipmapLinearFilter,
    magFilter: THREE.LinearFilter,
    generateMipmaps: true,
    depthBuffer: false,
    stencilBuffer: false,
  });

  // The tile must cover everything the chunk draws, not just its [-1,1] box.
  const cam = new THREE.OrthographicCamera(-extent, extent, extent, -extent, -10, 10);
  cam.position.z = 1;
  const scene = new THREE.Scene();

  const prevTarget = renderer.getRenderTarget();
  const uSize = material.uniforms.uSizeToPx.value;
  const uOpac = material.uniforms.uOpacity.value;

  // local [-1,1] maps to TILE pixels, so one local unit is TILE/2 px
  material.uniforms.uSizeToPx.value = TILE / (2 * extent);
  material.uniforms.uOpacity.value = 1;

  // The tile rectangle goes on the RENDER TARGET, not on the renderer.
  // renderer.setViewport()/setScissor() are canvas-space calls: three
  // multiplies them by the canvas pixel ratio on the way to GL. At dpr 2
  // every tile was therefore drawn at twice the size into a box twice as
  // wide, so each variant sprawled across its neighbours and the ones baked
  // after it cleared holes back out of it. The atlas came out ~30% darker
  // than the chunks it was standing in for, which is what showed up as a
  // step in mean frame luminance at the impostor threshold.
  rt.viewport.set(0, 0, rt.width, rt.height);
  rt.scissorTest = false;
  renderer.setRenderTarget(rt);
  renderer.setClearColor(0x000000, 0);
  renderer.clear(true, false, false);

  const disposables = [];
  for (let v = 0; v < VARIANTS; v++) {
    const data = buildVariant(v);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(data.pos, 3));
    geom.setAttribute('size', new THREE.BufferAttribute(data.size, 1));
    geom.setAttribute('tint', new THREE.BufferAttribute(data.color, 3));
    geom.setAttribute('aux', new THREE.BufferAttribute(data.aux, 1));
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), extent * 2);
    const pts = new THREE.Points(geom, material);
    pts.frustumCulled = false;
    scene.add(pts);

    const x = (v % COLS) * TILE;
    const y = (ROWS - 1 - Math.floor(v / COLS)) * TILE;   // GL y is bottom-up
    rt.viewport.set(x, y, TILE, TILE);
    rt.scissor.set(x, y, TILE, TILE);
    rt.scissorTest = true;
    renderer.setRenderTarget(rt);            // re-applies the rect above
    renderer.render(scene, cam);

    scene.remove(pts);
    disposables.push(geom);
  }
  for (const g of disposables) g.dispose();

  material.uniforms.uSizeToPx.value = uSize;
  material.uniforms.uOpacity.value = uOpac;
  rt.scissorTest = false;
  rt.viewport.set(0, 0, rt.width, rt.height);
  renderer.setRenderTarget(prevTarget);
  renderer.setScissorTest(false);
  renderer.setClearColor(0x05060a, 1);

  return rt.texture;
}

// One draw call per level. Positions are rewritten every frame from float64
// values computed on the CPU, so the impostors inherit the cascade's
// precision guarantee exactly as the full-detail chunks do.
export class ImpostorLayer {
  constructor(scene, atlas, maxCount = 8192, premultiplied = false) {
    this.max = maxCount;
    this.n = 0;
    this.pos = new Float32Array(maxCount * 3);
    this.size = new Float32Array(maxCount);
    this.variant = new Float32Array(maxCount);
    this.orient = new Float32Array(maxCount);
    this.tint = new Float32Array(maxCount * 3).fill(1);
    this.gain = new Float32Array(maxCount).fill(1);

    const geom = new THREE.InstancedBufferGeometry();
    // the unit quad every instance is stamped from
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.5, -0.5, 0,   0.5, -0.5, 0,   0.5, 0.5, 0,   -0.5, 0.5, 0,
    ]), 3));
    geom.setIndex([0, 1, 2, 0, 2, 3]);
    this.aPos = new THREE.InstancedBufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage);
    this.aSize = new THREE.InstancedBufferAttribute(this.size, 1).setUsage(THREE.DynamicDrawUsage);
    this.aVar = new THREE.InstancedBufferAttribute(this.variant, 1).setUsage(THREE.DynamicDrawUsage);
    this.aOri = new THREE.InstancedBufferAttribute(this.orient, 1).setUsage(THREE.DynamicDrawUsage);
    this.aTint = new THREE.InstancedBufferAttribute(this.tint, 3).setUsage(THREE.DynamicDrawUsage);
    this.aGain = new THREE.InstancedBufferAttribute(this.gain, 1).setUsage(THREE.DynamicDrawUsage);
    geom.setAttribute('iPos', this.aPos);
    geom.setAttribute('size', this.aSize);
    geom.setAttribute('variant', this.aVar);
    geom.setAttribute('orient', this.aOri);
    geom.setAttribute('tint', this.aTint);
    geom.setAttribute('gain', this.aGain);
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9);
    geom.instanceCount = 0;
    this.geom = geom;

    this.material = new THREE.ShaderMaterial({
      defines: premultiplied ? { PREMULT: '' } : {},
      uniforms: {
        uAtlas:   { value: atlas },
        uOpacity: { value: 1 },
        uGain:    { value: 1 },
        uMinPx:   { value: 1.0 },
        uTile:    { value: new THREE.Vector2(1 / COLS, 1 / ROWS) },
        uViewport: { value: new THREE.Vector2(1, 1) },
      },
      vertexShader: IMP_VERT,
      fragmentShader: IMP_FRAG,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      ...(premultiplied ? {
        blending: THREE.CustomBlending,
        blendEquation: THREE.AddEquation,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneMinusSrcAlphaFactor,
        blendEquationAlpha: THREE.AddEquation,
        blendSrcAlpha: THREE.OneFactor,
        blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      } : { blending: THREE.AdditiveBlending }),
    });

    this.points = new THREE.Mesh(geom, this.material);
    this.points.frustumCulled = false;
    this.points.matrixAutoUpdate = false;
    scene.add(this.points);
  }

  begin(opacity, vpW, vpH) {
    this.n = 0;
    this.material.uniforms.uOpacity.value = opacity;
    if (vpW) this.material.uniforms.uViewport.value.set(vpW, vpH);
  }

  add(x, y, z, px, variant, orient, tint, gain) {
    const i = this.n;
    if (i >= this.max) return false;
    this.pos[i * 3] = x; this.pos[i * 3 + 1] = y; this.pos[i * 3 + 2] = z;
    this.size[i] = px;
    this.variant[i] = variant;
    this.orient[i] = orient;
    if (tint) {
      this.tint[i * 3] = tint[0]; this.tint[i * 3 + 1] = tint[1]; this.tint[i * 3 + 2] = tint[2];
    } else {
      this.tint[i * 3] = this.tint[i * 3 + 1] = this.tint[i * 3 + 2] = 1;
    }
    this.gain[i] = gain === undefined ? 1 : gain;
    this.n++;
    return true;
  }

  end() {
    this.points.visible = this.n > 0;
    if (this.n > 0) {
      this.aPos.needsUpdate = true;
      this.aSize.needsUpdate = true;
      this.aVar.needsUpdate = true;
      this.aOri.needsUpdate = true;
      this.aTint.needsUpdate = true;
      this.aGain.needsUpdate = true;
      this.geom.instanceCount = this.n;
    }
    return this.n;
  }
}
