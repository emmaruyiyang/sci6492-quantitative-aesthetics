// Fold the built site into ONE html file you can double-click.
//
// Run `vite build` first, then `node tools/make_single_html.mjs`.
//
// Every data file becomes a data: URI handed to the page on
// globalThis.__SCALEVIZ_INLINE__, keyed by the same relative path the source
// asks for (see src/assets.js). The bundle goes in as an inline module, so
// the page fetches nothing and works from file:// as well as from a server.
//
// Base64 costs 4 bytes per 3, so the file is about a third larger than the
// folder build. That is the price of having no folder.

import { readFile, writeFile, readdir, stat, mkdir } from 'node:fs/promises';
import { join, relative, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

// .pathname keeps %20, and this project lives under a path with a space
const ROOT = fileURLToPath(new URL('..', import.meta.url));
const DIST = join(ROOT, 'dist');
const OUT_DIR = join(ROOT, 'dist-single');
const OUT = join(OUT_DIR, 'scaleviz.html');

const MIME = {
  '.json': 'application/json',
  '.bin': 'application/octet-stream',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
};

async function walk(dir) {
  const out = [];
  for (const name of await readdir(dir)) {
    const p = join(dir, name);
    if ((await stat(p)).isDirectory()) out.push(...await walk(p));
    else out.push(p);
  }
  return out;
}

const files = await walk(DIST);

// the data files, keyed by their path relative to dist/
const inline = {};
for (const f of files) {
  const rel = relative(DIST, f).split(/[\\/]/).join('/');
  const ext = extname(f);
  if (!MIME[ext]) continue;                  // index.html and the js go elsewhere
  const b64 = (await readFile(f)).toString('base64');
  inline[rel] = `data:${MIME[ext]};base64,${b64}`;
}

let html = await readFile(join(DIST, 'index.html'), 'utf8');
const m = html.match(/<script[^>]*src="([^"]+)"[^>]*><\/script>/);
if (!m) throw new Error('no bundle <script> in dist/index.html');
const bundlePath = join(DIST, m[1].replace(/^\.\//, ''));
const bundle = await readFile(bundlePath, 'utf8');

// A literal </script> anywhere inside either payload would close the tag
// early. Shader source and minified strings are exactly where that hides.
const safe = (s) => s.replace(/<\/script/gi, '<\\/script');

// A replacer FUNCTION, not a replacement string. In a replacement string
// `$&`, `$'` and '$`' are substitution patterns, and minified JS is full of
// them - the first attempt spliced four copies of the original HTML into the
// middle of the bundle, which the parser met as "Unexpected token '<'".
html = html.replace(m[0], () =>
  '<script>window.__SCALEVIZ_INLINE__ = ' + safe(JSON.stringify(inline)) + ';</script>\n' +
  '    <script type="module">' + safe(bundle) + '</script>');

await mkdir(OUT_DIR, { recursive: true });
await writeFile(OUT, html);

const bytes = Buffer.byteLength(html);
console.log(`${relative(ROOT, OUT)}  ${(bytes / 1048576).toFixed(2)} MB`);
for (const [k, v] of Object.entries(inline)) {
  console.log(`  inlined ${k.padEnd(28)} ${(v.length / 1024).toFixed(0)} KB as data:`);
}
console.log(`  inlined bundle${''.padEnd(22)} ${(bundle.length / 1024).toFixed(0)} KB`);
