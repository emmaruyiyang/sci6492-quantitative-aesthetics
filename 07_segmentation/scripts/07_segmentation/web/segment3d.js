// Segmentation Explorer, 3D model mode: renders the meshes of a Rhino file
// (as /api/scene/model sends them) from N cameras with three.js.
//
// Everything stays in Rhino's coordinates, Z up: the cameras recorded with
// every frame are in the model's coordinates, ready to be drawn back into it
// (Save frustum). A camera is {position, target, up, fov (vertical, degrees),
// aspect, near, far, width, height}, as three.js's PerspectiveCamera has it.
//
// The page script (segment.js, not a module) uses it as window.Scene3D.

import * as THREE from "three";

let meshes = [];            // three.js meshes of the loaded model
let bounds = null;          // {center: Vector3, radius}

function build(data) {
  for (const mesh of meshes) mesh.geometry.dispose();
  meshes = data.meshes.map((m) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(m.vertices, 3));
    if (m.colors) geometry.setAttribute("color", new THREE.Float32BufferAttribute(m.colors, 3));
    geometry.setIndex(m.triangles);
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry);
    mesh.userData = { color: new THREE.Color(...m.color), vertexColors: Boolean(m.colors), name: m.name };
    return mesh;
  });
  const [lo, hi] = data.bounds.map((p) => new THREE.Vector3(...p));
  bounds = { center: lo.clone().add(hi).multiplyScalar(0.5), radius: Math.max(hi.distanceTo(lo) / 2, 1e-9) };
}

function material(mode, mesh) {
  const side = THREE.DoubleSide;
  if (mode === "normals") return new THREE.MeshNormalMaterial({ side });
  if (mode === "colors") {
    return new THREE.MeshStandardMaterial({
      side, roughness: 0.75, metalness: 0,
      vertexColors: mesh.userData.vertexColors,
      color: mesh.userData.vertexColors ? 0xffffff : mesh.userData.color,
    });
  }
  return new THREE.MeshStandardMaterial({ side, roughness: 0.85, metalness: 0, color: 0xe6e3dd });
}

// Unit vectors from the center towards the cameras
function directions(n, layout, elevation) {
  const out = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    let z, a = (2 * Math.PI * i) / n;
    if (layout === "orbit") {
      z = Math.sin(elevation * Math.PI / 180);
    } else if (layout === "spiral") {
      // from 10 degrees below the horizon up to 80 above, going around twice
      z = Math.sin((-10 + 90 * i / Math.max(n - 1, 1)) * Math.PI / 180);
      a *= 2;
    } else {
      // evenly spread (Fibonacci): dome from the horizon up, sphere all around
      z = layout === "dome" ? 1 - (i + 0.5) / n : 1 - 2 * (i + 0.5) / n;
      a = i * golden;
    }
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    out.push(new THREE.Vector3(r * Math.cos(a), r * Math.sin(a), z));
  }
  return out;
}

// -> [{blob (PNG), camera}], one per view
async function render({ views, layout, elevation, fov, size, mode, background, onProgress }) {
  if (!bounds) throw new Error("no model loaded");
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(size, size);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(background);
  const sky = new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.6);
  sky.position.set(0, 0, 1);                 // Z up
  scene.add(sky);
  const materials = meshes.map((m) => material(mode, m));
  meshes.forEach((m, i) => { m.material = materials[i]; scene.add(m); });

  // the model's bounding sphere fills the frame
  const distance = bounds.radius / Math.sin((fov / 2) * Math.PI / 180) * 1.05;
  const near = Math.max(distance - bounds.radius * 1.5, distance * 0.01);
  const far = distance + bounds.radius * 1.5;
  const camera = new THREE.PerspectiveCamera(fov, 1, near, far);
  const key = new THREE.DirectionalLight(0xffffff, 1.4);  // from above left of the camera
  key.position.set(-1, 1, 1);
  camera.add(key);
  scene.add(camera);

  const frames = [];
  const all = directions(views, layout, elevation);
  for (let i = 0; i < all.length; i++) {
    const d = all[i];
    // looking straight up or down: Z can't be up, Y is
    const up = Math.abs(d.z) > 0.999 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
    camera.position.copy(bounds.center).addScaledVector(d, distance);
    camera.up.copy(up);
    camera.lookAt(bounds.center);
    camera.updateMatrixWorld();
    renderer.render(scene, camera);
    const blob = await new Promise((resolve) => renderer.domElement.toBlob(resolve, "image/png"));
    frames.push({
      blob,
      camera: {
        position: camera.position.toArray(), target: bounds.center.toArray(), up: up.toArray(),
        fov, aspect: 1, near, far, width: size, height: size,
      },
    });
    if (onProgress) onProgress((i + 1) / all.length);
  }
  meshes.forEach((m) => scene.remove(m));
  materials.forEach((m) => m.dispose());
  renderer.dispose();
  renderer.forceContextLoss();
  return frames;
}

// Distance from the camera plane, per pixel: a float render target
const depthMaterial = new THREE.ShaderMaterial({
  side: THREE.DoubleSide,
  vertexShader: `varying float depth;
    void main() {
      vec4 p = modelViewMatrix * vec4(position, 1.0);
      depth = -p.z;
      gl_Position = projectionMatrix * p;
    }`,
  fragmentShader: `varying float depth;
    void main() { gl_FragColor = vec4(depth, 0.0, 0.0, 1.0); }`,
});

function cameraOf(c) {
  const camera = new THREE.PerspectiveCamera(c.fov, c.aspect, c.near, c.far);
  camera.position.fromArray(c.position);
  camera.up.fromArray(c.up);
  camera.lookAt(new THREE.Vector3(...c.target));
  camera.updateMatrixWorld();
  camera.updateProjectionMatrix();
  return camera;
}

// Where every vertex lands in a frame, if the camera sees it.
// -> per mesh {uv: [x, y] in the frame's pixels per vertex (-1, -1: not
// seen), facing: |cos| of the angle between the surface and the view, per vertex}
// Seen: inside the frame and not behind another surface (its distance is at
// most the depth buffer's around its pixel, plus a little).
let depthRenderer = null;       // one for every projection: browsers allow only a few WebGL contexts

function project(c) {
  const { width: W, height: H } = c;
  depthRenderer = depthRenderer || new THREE.WebGLRenderer();
  const renderer = depthRenderer;
  renderer.setPixelRatio(1);
  renderer.setSize(W, H);
  const target = new THREE.WebGLRenderTarget(W, H, { type: THREE.FloatType });
  const scene = new THREE.Scene();
  meshes.forEach((m) => { m.material = depthMaterial; scene.add(m); });
  const camera = cameraOf(c);
  renderer.setClearColor(0x000000, 0);
  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  const depth = new Float32Array(W * H * 4);
  renderer.readRenderTargetPixels(target, 0, 0, W, H, depth);
  meshes.forEach((m) => scene.remove(m));
  renderer.setRenderTarget(null);
  target.dispose();

  // depth at pixel (x, y), y from the top; 0 = nothing there (far away)
  const at = (x, y) => {
    const d = depth[4 * ((H - 1 - y) * W + x)];
    return d > 0 ? d : Infinity;
  };
  const tolerance = bounds.radius * 0.01;
  const view = camera.matrixWorldInverse;
  const p = new THREE.Vector3(), n = new THREE.Vector3(), toCamera = new THREE.Vector3();
  return meshes.map((m) => {
    const positions = m.geometry.attributes.position, normals = m.geometry.attributes.normal;
    const count = positions.count;
    const uv = new Float32Array(2 * count).fill(-1);
    const facing = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      p.fromBufferAttribute(positions, i);
      const distance = -p.clone().applyMatrix4(view).z;
      toCamera.copy(camera.position).sub(p).normalize();
      p.project(camera);
      if (Math.abs(p.x) > 1 || Math.abs(p.y) > 1 || p.z < -1 || p.z > 1) continue;
      const x = (p.x + 1) / 2 * W, y = (1 - p.y) / 2 * H;
      const px = Math.min(W - 1, Math.floor(x)), py = Math.min(H - 1, Math.floor(y));
      let nearest = 0;      // the farthest surface around the pixel: forgiving at edges
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const qx = px + dx, qy = py + dy;
          if (qx >= 0 && qy >= 0 && qx < W && qy < H) nearest = Math.max(nearest, at(qx, qy));
        }
      }
      if (distance > nearest + tolerance) continue;
      uv[2 * i] = x;
      uv[2 * i + 1] = y;
      facing[i] = Math.abs(n.fromBufferAttribute(normals, i).dot(toCamera));
    }
    return { uv, facing };
  });
}

const vertexCounts = () => meshes.map((m) => m.geometry.attributes.position.count);

window.Scene3D = { build, render, project, vertexCounts };
