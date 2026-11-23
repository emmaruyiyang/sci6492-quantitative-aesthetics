// A copy of dist/ for a host that serves .json but not .bin.
//
// Run `vite build` first, then `node tools/make_artifact_build.mjs`.
//
// The five density fields (245 KB in all) are folded into fields.json as
// base64 and their .bin files dropped; src/fields.js decodes them with atob.
// The two PNGs stay real files, because they are 3.6 MB and because <img>
// loads them, which has never been the problem.

import { readFile, writeFile, rm, cp, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const DIST = join(ROOT, 'dist');
const OUT = join(ROOT, 'dist-artifact');

await rm(OUT, { recursive: true, force: true });
await mkdir(OUT, { recursive: true });
await cp(DIST, OUT, { recursive: true });

const manifestPath = join(OUT, 'fields', 'fields.json');
const meta = JSON.parse(await readFile(manifestPath, 'utf8'));

const entries = [meta.galaxy.diffuse, meta.galaxy.stars, meta.galaxy.color,
                 meta.river.mask, meta.river.dist];
let folded = 0;
for (const e of entries) {
  const buf = await readFile(join(OUT, e.file));
  e.b64 = buf.toString('base64');
  await rm(join(OUT, e.file));
  folded += buf.length;
  delete e.file;
}

// river_spine.json is already a served type, but inlining it saves a request
// and costs 14 KB.
const spine = JSON.parse(await readFile(join(OUT, meta.river.spine.file), 'utf8'));
await rm(join(OUT, meta.river.spine.file));
meta.river.spine.data = spine;
delete meta.river.spine.file;

await writeFile(manifestPath, JSON.stringify(meta));
const after = (await readFile(manifestPath)).length;
console.log(`dist-artifact/  ${folded} bytes of .bin -> ${(after / 1024).toFixed(0)} KB fields.json`);
