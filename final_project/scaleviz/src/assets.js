// Where an asset lives.
//
// Normally: next to the page, under Vite's BASE_URL, so the build works at a
// domain root and under a GitHub Pages project path alike.
//
// In the single-file build (tools/make_single_html.mjs) there is nothing next
// to the page - the page IS the whole thing - so every asset has been inlined
// as a data: URI and handed over on globalThis under its usual relative path.
// A data: URI is a URL like any other: fetch() reads it and an <img> loads it,
// which is why this can be one lookup instead of a loader shim.
export function asset(base, path) {
  const inlined = globalThis.__SCALEVIZ_INLINE__;
  if (inlined && inlined[path]) return inlined[path];
  return base + path;
}
