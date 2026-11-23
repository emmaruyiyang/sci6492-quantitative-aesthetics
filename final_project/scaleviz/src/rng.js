// Deterministic hashing + PRNG so every chunk can be regenerated on demand
// from its key alone (no global state, no stored point data).

export function hashStr(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller, cached second sample.
export function gaussian(rnd) {
  let u = 0, v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Cheap smooth 1D value noise (for river meander headings).
export function makeNoise1D(rnd, octaves = 4) {
  const tables = [];
  for (let o = 0; o < octaves; o++) {
    const n = 64;
    const t = new Float32Array(n);
    for (let i = 0; i < n; i++) t[i] = rnd() * 2 - 1;
    tables.push(t);
  }
  return function (x) {
    let sum = 0, amp = 1, freq = 1, norm = 0;
    for (let o = 0; o < octaves; o++) {
      const t = tables[o], n = t.length;
      const p = x * freq;
      const i0 = Math.floor(p);
      const f = p - i0;
      const a = t[((i0 % n) + n) % n];
      const b = t[((i0 + 1) % n + n) % n];
      const u = f * f * (3 - 2 * f);
      sum += (a + (b - a) * u) * amp;
      norm += amp;
      amp *= 0.5;
      freq *= 2.07;
    }
    return sum / norm;
  };
}
