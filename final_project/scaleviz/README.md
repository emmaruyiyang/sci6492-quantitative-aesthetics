# scaleviz — P0 + P1 + P2 + P3 + P4 + P5 + P6

Continuous zoom across ~4.5 decades: **cell → river → galaxy**.
P0 built the scaffold: numerical stability and the LOD plumbing, the two
things that are expensive to retrofit. P1 replaced the placeholder geometry
with density fields extracted from the two reference photographs, and gave
each scale its own mark instead of one shared dot. P3 added impostors, which
is what makes the far field exist at all: the galaxy is visibly made of
rivers. P2 turned the river level from 1,600 unrelated squiggles into one
drainage network that spans the whole galaxy.

## Run

```bash
npm install
npm run dev     # http://localhost:5273
```

The fields in `public/fields/` are committed, so the app runs as-is. To
re-extract them from `refs/`:

```bash
uv run --no-project --python 3.12 --with numpy --with pillow --with scipy python tools/extract_fields.py
```

wheel = zoom (toward the cursor) · drag = pan · `space` = auto tour ·
`d` = dive to the deepest anchor · `r` = reset

The scale bar in the bottom-left corner is also the zoom control: drag it,
click anywhere on its track, or click a level name to jump there. Focused,
it takes the arrow keys (0.05 in u, 0.25 with shift) and `Home` / `End`.

The wheel is a local gesture - it moves you a little, from wherever you
already are - so there was no way to address the range as a whole; getting
from a cell to the galaxy meant scrolling and hoping. The bar was already
drawing that range, so it is now grabbable. Dragging it changes the target
zoom only and leaves the focus point alone, so the slider zooms about the
centre of the view, the way `d` and `r` do; the wheel keeps its own
cursor-pinning behaviour. The level ticks sit where each level's cross-fade
weight actually peaks (u = 0.93 / 2.27 / 3.55), not at even thirds: the
steps are 10^1.5 and 10^2, so evenly spaced labels would misstate the
hierarchy.

## The one idea

The camera never moves. It is an orthographic camera with a fixed half-height
of 1 view unit. The whole zoom state is two float64 numbers:

- `H` — the view half-height expressed in **absolute units** (1 unit ≈ one cell)
- `F` — the focus point in absolute coordinates

Every chunk is placed by

```
position = (chunkOrigin - F) / H      // float64 on the CPU
scale    = chunkUnit / H
```

computed in JS (float64) and only then handed to the GPU. A chunk that is on
screen has `|position| <= ~2` and vertex data local to `[-1,1]`, so float32
never holds a large number and never has to cancel one.

The naive alternative — pushing the camera back to see the galaxy — loses
most of the float32 mantissa and the near view shimmers. Measured drift of
the float32 pipeline against an exact float64 computation, swept across the
full zoom range: **max 6.8e-13 px**.

Zoom is linear in `u = log10(H)`, so the apparent rate of change is constant
at every scale.

## P1 — the density fields

`tools/extract_fields.py` turns the two reference photographs into the fields
the runtime samples. 254 KB in total, in `public/fields/`.

**galaxy_ref.jpg** is split into the two things a photograph of the Milky Way
actually contains:

| field | how | drives |
|---|---|---|
| `galaxy_diffuse` | low-pass luminance, σ=6, γ=1.45 | where the band's light is, and where river chunks exist at all |
| `galaxy_stars` | high-pass luminance, σ=1.6 | the resolved stars on top |
| `galaxy_color` | heavily blurred RGB, σ=9 | each point's colour — warm core, cool rim |

The frame is feathered to zero at the border, or the galaxy would render as a
visible rectangle.

**river_ref.jpg** is traced rather than sampled:

| field | how | drives |
|---|---|---|
| mask | saturation + value threshold, largest blob | — |
| `dist` | distance transform | channel width |
| `river_spine.json` | per-row ridge of the distance transform, resampled to 220 stations along arc length | the **curvature signature**: sinuosity 1.694, curvature RMS 0.1176, mean half-width 0.1205, arc length 2.522 |

Copying the traced centreline 1,600 times would be obvious, so each river
chunk instead **integrates the heading from the real per-station curvature**,
starting at a random station, optionally mirrored, with gain 0.74–1.04 and a
little noise. Every river is different and every river meanders like the real
one. Tributaries use the same signature, shorter and narrower, translated onto
a junction — that is what makes the level a network rather than a line.

Two proportion bugs worth recording, both found by looking at the screen:

- `halfWidth` was measured against the photograph's arc length and applied to
  a path with a different one, so every mark came out a quarter of the chunk
  wide and the level rendered as a solid white sheet. Fixed by exporting
  `arcLength` and rescaling.
- Normalising each river to **fill** its chunk stretched knotted paths and
  fattened their channels. Now scaled by arc length, with a bbox fit only as a
  fallback, so the generated channel keeps the reference's width-to-length
  ratio.

### Placement

River chunks are 200 absolute units wide. Sampling their positions straight
from the density field gave a **median nearest-neighbour spacing of 78**, so
every chunk sat 2–3 deep and the whole level saturated to white. Density now
decides where rivers are *likely*; blue-noise rejection (min distance 168)
decides that they do not land on top of each other. Result: 1,600 rivers,
median spacing 192, minimum 168.

## P1 — one mark per scale

A scale is not a different size of the same dot. `MARK` is a per-level
`#define` on one shader (`cascade.js`):

- **cell** — a membrane rim, cytoplasm falloff and an offset nucleus, all
  drawn procedurally. *Replaced in P4 by a photograph of a real cell.*
- **river** — the sprite is stretched 2.6× along the local flow direction,
  passed per point as `aux`, so the channel shows a current.
- **galaxy** — two populations in one buffer: faint wide glow at 3% alpha that
  must overlap to read as light, and sharp stars with a faint halo.

Cell chunks also had to stop being discs: they were rendering as separate
circular pills on the water. They now fill a soft-edged square and sit across
the channel as well as along it, so they merge into continuous tissue.

## P2 — one drainage network across all chunks

Before this, every river chunk grew its own private meander and neighbours
knew nothing about each other. The level was 1,600 separate squiggles.

Now the river origins are the **nodes of one global graph**. Each node flows
to its nearest neighbour with higher density — the same galaxy field that
placed them — which gives a forest of catchments converging on the bright
parts of the band. Strictly increasing potential along every link means the
graph cannot contain a cycle.

### Half-edges

An edge spans two nodes, so it belongs to neither chunk. Chunks must still
build independently from nothing but their own id.

The fix: both endpoints generate the **identical** edge geometry from a seed
derived from the *sorted* node pair, and each draws only the half nearer to
itself. Every edge is drawn exactly once, and the two halves meet at the
midpoint exactly — measured mismatch across 500 edges is **0.0**.

The channel between two nodes is a meander integrated from the reference
river's own curvature, then pinned to the two node positions with a
similarity transform. Sinuosity is a ratio, so it survives that transform
untouched: the channel between nodes 200 apart comes out ~1.69 × 200 long,
like the traced river.

### Depression filling

"Flow to the brightest neighbour" leaves one basin per local maximum. Sampled
raw, the density field gave 355 basins and 142 nodes with no link at all,
which rendered as empty chunks. Two changes: the potential is **smoothed**
over a 9-tap disc, and a second pass joins each remaining basin to the nearest
node in a *different* component. Linking only across components is what keeps
cycles impossible, so the potential rule can be dropped there.

| | before | after |
|---|---|---|
| basins | 355 | 265 |
| edges | 1,245 | 1,335 |
| largest drainage area | 151 | 205 |

122 nodes still have no neighbour within `EDGE_MAX` at all. They are marked
inactive rather than drawn as empty chunks. Channel half-width follows
drainage area as `area^0.35`, giving 5.2 at a headwater to 33.5 at a trunk.

### Three calibration bugs this surfaced

- **Coverage, not count.** Budgeting points by channel *area* makes coverage
  grow as width², so wide trunks were painted ~100× over, clipped the 8-bit
  framebuffer to white, and stood out against their own impostors. Points are
  now budgeted for constant coverage.
- **Lattice, not scatter.** Cell chunks scattered at random need ~3.6×
  oversampling before the Poisson holes close. A jittered lattice covers the
  channel evenly at the same count.
- **Interpolate the polyline.** Cell stations snapped to the nearest edge
  vertex, which put a floor of one segment length (~2 units) under their
  spacing — wider than a cell chunk, so the channel came out striped.

## P3 — impostors

A chunk covering 8 pixels does not deserve 1,500 points and a draw call. Below
`FULL_MIN_PX = 34` it is replaced by a baked picture: `tools`-free, done at
startup in `impostor.js`, which renders 12 representative chunks per level
into a 512×384 half-float atlas using **that level's own material**, so the
baked tile is exactly what the real thing looks like at that size. Half-float
matters — additive sums must not clamp at 1.0 during the bake.

Every distant chunk then becomes one textured point sprite, and all of them go
out in **one draw call per level** whose positions are rewritten each frame
from the same float64 computation the full-detail chunks use. Impostors
inherit the precision guarantee rather than working around it.

Impostors are **statistical, not exact**: a chunk is drawn as one of the baked
variants picked by hash, not as a picture of itself. 12 tiles would repeat
visibly, so each sprite also gets one of 8 orientations — four 90° rotations
and a mirror, which are exact, cost nothing and do not clip corners. 96
apparent variants.

Two things this fixed that were not about speed:

- **Nothing is dropped any more.** The old budget drew the nearest N chunks
  and discarded the rest; a full zoom sweep used to drop up to 1,563 chunks in
  a single frame. It now drops **zero** across the whole range.
- **The galaxy is made of rivers.** At u≈3.7, 1,113 river impostors are
  scattered through the band. Before, they were below `MIN_CHUNK_PX` and
  simply not drawn.

Picking full-detail chunks by sorting every candidate by distance became far
too expensive once the impostor path started visiting thousands of chunks per
frame. Full detail is now a disc around the focus whose radius is derived from
the budget, `2·unit·√(budget/π)` — no sort, and steadier frame to frame.
`content.js` also grew an allocation-free `forEachChunk`, because an object
per candidate made the garbage collector the most expensive thing in the frame.

### The bottleneck was somewhere else

Impostors removed all the dropping and cut peak draw calls from 67 to 45, but
the worst frame barely moved: 11.9 → 12.5 ms. The cost at u≈3.7 was never the
chunks. It was the galaxy's own 78,000 glow sprites, which grow as you zoom
*into* the level — they reached 129 px each.

A mark should never be larger on screen than it is at its own level's native
scale. Past that you have zoomed into the level and the child level is
carrying the detail, so letting the parent's sprites keep growing buys nothing.
Capping sprite growth at `MARK_MAX_LOCAL · halfPxY` took the worst frame from
12.5 ms to **8.3 ms**, and is invisible at the home view (46 px, under the cap).

## P4 — the cell is a photograph

Up to P3 a cell was procedural: a membrane, some cytoplasm, an offset
nucleus, drawn by the fragment shader. From P4 a cell is one cell cut out of
one of the 500 micrographs in `sample/topography/cell/`.

`tools/extract_cells.py` does the cutting, offline:

```
uv run --no-project --python 3.12 --with numpy --with pillow --with scipy \
  python tools/extract_cells.py
```

The 500 photographs are not one kind of image — bright-field histology (dark
cells on white), fluorescence (bright cells on black), phase contrast, stained
sections — so nothing in the extractor may assume "the cells are the bright
bits":

- **background** = median of a border ring, which is white for a slide and
  black for a fluorescence field;
- **foregroundness** = colour distance from that background, normalised;
- **blobs** = multiscale LoG maxima on foregroundness, scale → radius;
- **score** = the blob's contrast against its own annulus × saturation ×
  detail (Laplacian energy inside the blob, normalised). Detail is what
  rejects soft out-of-focus balls, which were half of the first atlas;
  circularity (the blob must fill its own disc) rejects flat patches of
  background between two real cells;
- **alpha** = foregroundness × a radial window, so a crop taken out of dense
  tissue still comes out as one round cell;
- **exposure** = each tile scaled toward a common mean luminance.

At most one tile is kept per photograph, so the atlas is 100 cells from 100
different photographs — `public/cells/cell_atlas.png`, 10×10 tiles of 128 px,
2.5 MB, plus `cells.json` with each tile's source filename and its
alpha-weighted mean colour.

**Exposure has to roll off, not clip.** The first atlas came out in neon
magenta and electric cyan. Equalising exposure by multiplying and clipping is
a hue rotation: a pale lavender histology cell multiplied by 3.2 pins its red
and blue channels at 1.0 while green stays put. The gain is now capped at
1.9, the top 20% is rolled off exponentially instead of clipped, and anything
still above 0.80 saturation is pulled back toward its own luminance.

**Matter covers; light adds.** The cell level is the only one that does not
blend additively. Three overlapping photographed cells summed to white
immediately. Level 0 writes premultiplied `rgb*a, a` and blends source-over,
so cells occlude each other. Levels 1 and 2 are light and still add.

**A palette is a window on the hue circle, not a lottery.** Every cell chunk
hanging off one river node draws from the same 6 tiles, so a field of cells
reads as one tissue instead of a hundred unrelated organisms shuffled
together. But the first version picked those 6 uniformly out of an atlas that
runs the whole spectrum, and six uniform draws from a rainbow average to grey
— so *every* tissue averaged to the same grey and the level read as
colourless haze wherever chunks overlapped (which is everywhere: chunks sit
0.62 units apart and are 2.28 wide, so ~13 stack on each pixel). Tiles are
now sorted by hue and a palette is a 15-tile window on that order.

### Three measured bugs, one of them old

**A cell chunk leaves the full path while its cells are still bigger than a
pixel.** `FULL_MIN_PX` is now per level — `[170, 34, 0]`. An additive mark
that falls under `uMinPx` is clamped up and dimmed by `k²`, which conserves
its energy exactly; that is why 34 px was right for the river. A cell is drawn
*over*, and n overlapping sprites at alpha a composite to `1-(1-a)ⁿ`, not
`n·a`, so the same dimming comes out visibly darker. At 60 px per chunk the
full path was drawing 2,400 sub-pixel cells and producing a dark, banded
sheet next to its own impostors.

**The impostor atlas had been baked wrong since P3.** With a dev switch
(`__dbg.cascade.forceImpostor = level`) that renders the identical frame
entirely out of impostors, a cell impostor measured **0.68×** the brightness
of the chunk it replaced — and the deficit survived changing the bake
resolution, the bake extent and the variant seeds, which ruled out the three
plausible explanations in turn. The cause was `renderer.setViewport()` /
`setScissor()`: those are *canvas-space* calls and three multiplies them by
the canvas pixel ratio on the way to GL. At dpr 2 every tile was drawn at
twice the size into a box twice as wide, so each variant sprawled over its
neighbours and the variants baked after it cleared holes back out of it. The
tile rectangle now goes on the render target (`rt.viewport` / `rt.scissor`),
which three does not scale. Ratio after the fix: **1.01**.

**An impostor cannot be the colour of a tissue it was not baked from.** There
are 12 baked variants and 1,600 tissues, and cells are impostors over most of
their visible range, so without help every chunk past the threshold goes the
same average colour and a node's colour pops into existence as you zoom in.
Each impostor sprite now carries `paletteMean(node) / paletteMean(variant)`
as a per-sprite tint — three floats, memoised per (node, variant). Per-channel
agreement between the two paths is then 0.94 / 1.05 / 1.08.

River impostors were checked with the same switch and are also matched by
*shape* now: the variant is chosen by the chunk's drainage-area quantile
rather than by hash, so a trunk is no longer drawn as a headwater. That
changed measured luminance by about 1% — it is a fidelity fix, not a
brightness fix.

## P5 — making the scales actually compose

The levels looked right one at a time and did not add up: where the galaxy was
bright the rivers were not noticeably denser, and the tissue's colour had
nothing to do with the colour of the galaxy it sat in. Each scale was
independently plausible and none of them was *made of* the one above.

That is measurable, so it got measured first rather than argued about.

### The density link was there, but flattened

For each active node, take the galaxy's density at that node and the number
of other nodes within 700 absolute units, and fit a log-log slope. A composed
hierarchy wants **1.0**: twice the light, twice the river.

| | before | after |
|---|---|---|
| exponent (river areal density vs galaxy density) | 0.359 | **0.951** |
| Spearman | 0.901 | 0.947 |
| range in local node count | 19× | 70× |
| bottom density decile → top | 9.2 → 31.3 nodes | 7.5 → 120 nodes |

The cause was the blue-noise pass. A **fixed** Poisson radius lets density
decide *where* rivers are but never *how many*: past the packing limit extra
brightness buys nothing, so a 250× range in galaxy density came out as a 19×
range in rivers. The radius is now per candidate,

```
r(x) = clamp(K / sqrt(density(x)), 70, 290)
```

and points per area go as `1/r² ∝ density`, which is the exponent we wanted.
The larger of the two radii wins in the rejection test, so a sparse node is
not crowded out by a dense one that happened to be sampled later.

**What the clamps cost.** `R_MAX` cannot exceed `EDGE_MAX`: a node's
half-edge is exactly what `LEVEL_EXTENT[1]` promises its chunk reaches, and
culling, the impostor sprite size and the bake camera all believe that
promise. I first let sparse nodes reach further to stay connected and that
quietly broke all three, so it is a hard cap instead. Drainage area also
tracks the galaxy slightly better as a side effect (Spearman 0.287 → 0.393),
because dense regions now feed more upstream nodes into one trunk.

### Spending the nodes that landed nowhere

Adaptive spacing pushes the faintest regions past the link range on purpose,
so 260 of the 1,600 sampled nodes had no neighbour within `EDGE_MAX` at all.
They rendered nothing — 16% of the budget bought nothing — and they left bare
patches in the band.

The useful observation is that connectivity does not need the graph:

> a node ends up rendering nothing **exactly when** no other node lies within
> `EDGE_MAX` of it.

With a neighbour in range it either flows to one that is brighter, or a
dimmer one flows into it, or — being a one-node basin — depression filling
links it across components. So the test is a spatial-hash lookup and can run
*inside* the sampler, before the graph exists.

Sampling is now a loop: place nodes, drop the ones with no neighbour in
range, put the freed budget back in, repeat. The one thing that makes it
converge instead of churn is that a node placed on a recycle pass must itself
have a neighbour within `EDGE_MAX` — otherwise the CDF keeps sending the
same budget back to the same empty country it came from. With that condition
the freed nodes can only land where the spacing rule still has room, and the
rule being proportional to density, that means the bright regions.

| | before | after |
|---|---|---|
| inactive nodes | 260 | **0** |
| nodes that render | 1,340 | 1,600 |
| passes needed | — | 1 |
| density exponent | 0.951 | 0.928 |
| max drainage area | 205 | 737 |

The exponent barely moves, which is the point: the recycled nodes are placed
by the same rule as the originals, so they fill in proportionally rather than
piling into the core. The trunk's half-width roughly doubles (33.5 → 64.5)
because the network is now fully connected and far more of it drains into one
channel.

### The colour link did not exist at all

The river took its colour from the river photograph and the tissue picked a
hue window at random, so galaxy colour and tissue colour correlated
**−0.075** — i.e. not at all.

There is now one scalar per place, `nodeWarmth(node)`: where that node sits on
the galaxy's own warm-cool axis `(r−b)/(r+b)`, percentile-stretched so the
photograph's narrow spread fills the range the levels below can use. Both
lower levels read it:

- **cells** choose their tissue with it — the hue-sorted atlas window is
  centred on the hue the galaxy is here, warm → the red end, cool → the cyan
  end, plus a ±6-tile jitter so neighbouring rivers are not identical;
- **rivers** rotate the water toward it, one luminance-preserving hue
  rotation per chunk of ±1.15 rad, so the channel keeps the photograph's
  shading and its along-channel variation and only the hue moves.

| warmth correlation with the galaxy | before | after |
|---|---|---|
| tissue | −0.075 | **+0.797** |
| river water | n/a (fixed teal) | **+0.896** |

The first version measured **−0.899** for the river: `feColorMatrix` rotates
R→G→B, and the water starts at teal, so a positive rotation walks it toward
*blue*. Warm has to come down the hue circle, not up it.

### The link that already worked

Cell count against drainage area fits an exponent of **0.608** — a bigger
channel really does carry more cells, because the station lattice fills the
channel and the channel's width is `area^0.35`. That one needed nothing.

### What the new density cost, and what paid for it

Packing bright regions three times tighter put ~176,000 cell chunks in view
at u≈2.2 and dropped 104,000 of them over the impostor cap — for a level at
6% opacity. Two changes, both measured:

- **`FADE_IN` is per level now.** The river is at full weight by u≈1.95, so
  the cell level has nothing left to contribute past it and hands over at
  `[-2.05, -1.55]` instead of `[-2.30, -1.60]`. Peak cell impostors: 72,000
  (capped, dropping) → 9,868.
- **The station lattice loosened, 0.62 → 0.82.** 0.62 was calibrated in P2
  against the old procedural marks; P4's photographed cells are 1.45× wider,
  so the same coverage needs fewer of them. Dark-pixel fraction at u=0.1 went
  0.0029 → 0.0035 (i.e. unchanged), and the deepest frame went 14.3 ms → 10.6
  ms.

### The bright core was not stacking — it was dissolving

I had written down that river chunks stack about 19 deep in the galaxy's
core, from `(2 × LEVEL_EXTENT[1] × 100) / R_MIN = 304/70 = 4.3` per axis.
That number is about chunk *boxes*, and a river chunk does not paint its box
— it paints its half-edges, which are thin. Measuring the ink instead:

```
ground coverage of a node = (Σ half-edge length × channel width) / territory
```

| | |
|---|---|
| median | 0.19 |
| 90th percentile | 0.80 |
| faintest density decile → brightest | 0.05 → 0.90 |
| half-width / local spacing, median | 0.077 |
| half-width / local spacing, max | 0.92 |

So the core is not a pile-up: its channels cover about 90% of the ground,
which is exactly what *more river where it is brighter* should look like.
The estimate I had been quoting was wrong by an order of magnitude because it
measured the wrong thing.

Looking at the widest node instead of the arithmetic showed the real defect:
a handful of trunks render as **smooth white fog with no channel inside
them**. The cause is one line — the mark radius was `MARK_FRAC × half-width`,
so a trunk 13× wider than a headwater got marks 13× bigger, and past about
`hw = 20` they overlap into a blur. Single channels were dissolving; nothing
was stacking.

A mark is the *texture of water*, not a scaled copy of the channel — a big
river's surface is not 13× coarser than a creek's. So the radius is
sub-linear now:

```js
markRadius(hw) = MARK_FRAC · HW_REF · (hw / HW_REF)^0.45
```

For the widest trunk (hw = 64.5) that is 12.9 → 4.1, i.e. **5 marks across
the channel → 15.8**. The point budget needs no adjustment at all: it is
computed from the mark area, so smaller marks buy proportionally more of
them and coverage stays at `TARGET_COVER` by construction. Measured mean
frame luminance across the river threshold, before and after:

| u | before | after |
|---|---|---|
| 3.74 | 28.32 | 28.38 |
| 3.78 | 29.16 | 29.20 |
| 3.86 | 27.89 | 27.91 |

Under 0.3% drift, and the worst frame is unchanged at 8.5–8.9 ms. Headwaters
actually get *fewer*, slightly larger marks (radius 1.0 → 1.3), which is the
same trade running the other way.

## P6 — the cells get volume, from the depth maps

`sample/topography/cell_depth/` holds a monocular depth estimate for each of
the 500 photographs, and they are good: every cell comes out as a clean
convex bump. That is a height field, so `tools/extract_cells.py` now crops
the matching depth patch alongside each colour tile and bakes a normal map,
`public/cells/cell_normal.png` (1.1 MB, RG = normal xy, B = height). 99 of
the 100 tiles get one; the odd one out sits too close to its photograph's
edge for the depth crop to land.

The colour atlas is byte-for-byte identical afterwards — the depth is only
read, it never gets a vote on which cells are chosen or where their edges
are.

**Monocular depth arrives terraced.** The maps are 8-bit and a cell often
spans only 20–40 levels, so the first bake came out as corduroy: the
gradient turned every quantisation step into a ridge. Three changes fixed
it — resample with BILINEAR rather than LANCZOS (which rings on each step),
smooth at σ = 3.5 before differentiating (a cell's bump spans ~100 px of the
tile, so this costs none of the shape), and use Sobel rather than central
differences. The gradient is also clamped by *length* rather than per
component, so `nx, ny` stay inside the unit disc and the shader's
`nz = sqrt(1 - nx² - ny²)` is real.

Each tile's relief is normalised against its own 90th-percentile gradient.
Monocular depth is relative, so two tiles from two photographs are not on a
common scale; without this, some cells would be boulders and others nearly
flat.

**The normal has to be turned back.** Sprites pick one of 8 orientations, so
the texcoord goes `q → Rot(Mirror(q))` and the sampled normal has to come
back as `Mirror(Rot⁻¹(n))`. Rotations and a mirror are orthogonal, so it is
the same branch run backwards and costs nothing. Impostors inherit the
shading for free, because the atlas is baked through this same material.

**Lighting must not change exposure.** `AMBIENT + DIFFUSE · mean(lambert) = 1`
with `mean(lambert) = 0.596` measured over the atlas, giving `AMBIENT =
0.404`. The first build put a *different* light vector in the shader than
the one that calibration used — same direction, wrong z — and took **23% off
the whole cell level**:

| u | before lighting | wrong vector | calibrated |
|---|---|---|---|
| 0.86 | 41.74 | 32.16 | 42.43 |
| 1.06 | 37.25 | 29.00 | 37.92 |
| 1.10 | 34.90 | 27.44 | 35.05 |

Final drift is +1.8% in the full-detail band and +0.5% in the impostor band —
the residue is source-over being non-linear, not the calibration.

## P7 — the river is made of cells

The complaint that started this: *the levels do not look like they compose.
A river should look like it is made of the thing below it.*

It was not a rendering problem. Measured on the old build:

| u | cell weight | river nodes in view | cell chunk | cell chunks in view |
|---|---|---|---|---|
| 2.00 | 0.03 | **3** | **23 px** | 13,158 |
| 2.40 | 0 | 21 | 9 px | 154,066 |
| 2.80 | 0 | **122** | 4 px | **703,269** |

The cell level was switched off at u=2.0 while its chunks were still 23 px
across — not too small to see, deliberately cut. And at that moment the view
held three river nodes. Three nodes is not a network; the drainage pattern
does not read as one until u≈2.8, and there the view holds 703,000 cell
chunks against an impostor cap of 72,000. A 1.2-decade gap that no budget
closes.

The cause is the step between scales. At 100× per level, one river edge is
~70 cells long and a node owns 1,000–7,700 cell chunks, so "this river is
made of cells" is a five-million-element relationship. It is true, and it is
not drawable.

Two changes, because neither alone is enough.

### The mark at river scale becomes a cell

The shader already had a branch that samples the cell atlas — it was the cell
level's. The river level now uses it too: its points stop being abstract
streaks and become cell silhouettes, continuously rotated along the local
flow with ±26° of jitter so the field reads as water rather than corduroy.

Same chunks, same enumeration, same single draw call. The composition is
carried by the brush instead of by a second level being on screen.

Three things the swap forced:

**The river stopped being light.** Marks that are matter have to composite
source-over, not additively — eleven layers of additive streaks get brighter,
eleven layers of cells just get flatter. So the level moved to premultiplied
blending, and the channel is now painted ~2.2 times over instead of ~11.

**A cell is a cell.** The sub-linear `hw^0.45` law from P5 is right for a
texture of water, which may coarsen with the channel. It is wrong for an
object with a real size. At `RIVER_CELLS` the mark is constant in absolute
units, and the point budget follows automatically because it is computed from
the mark's area.

**The sprite is not the cell.** First attempt looked like scattered dust.
Measured off the atlas: a baked cell spans 0.907 of its tile's width and inks
0.412 of its area, so sizing the sprite as if the cell filled it made every
cell 2.2× too small. Worse, the cell reaches radius 58 of a 128 tile while
the square's inscribed circle is only 45.3 — **25.6% of all the ink in the
atlas lies outside that circle**, so a free rotation cuts corners off cells
and the loss swings with the flow direction. The cell level never hit this
because its eight orientations are 90° rotations and a mirror, which are
axis-preserving. The shader now samples 1.281× wider to shrink the content
into the safe circle.

### The step from cell to river goes from 100× to 10^1.5

An edge is now ~22 cells long, a node owns ~540 cell chunks instead of
~3,000, and the view that holds 43 river nodes holds about 20,000 cell
chunks — under the impostor cap, so at that zoom **every cell in the picture
is a real one**.

Every river constant that was written in absolute units is now written in
river units (`0.70 * RIVER_UNIT` instead of `70`), so the step is the only
thing that has to change. Node count, packing and the density link are
untouched: 1,600 nodes, median spacing 1.21 river units against 1.35 before,
density-vs-galaxy slope 0.899 (r=0.988) against 0.93.

The river→galaxy step stays at 100×. That relationship was already visible
without drawing it — spacing follows the galaxy's density and the colour
chain correlates +0.88 — so only the cell→river step needed closing.

The result:

| | before | after |
|---|---|---|
| cell/river overlap window | 0.6 decades | **1.23 decades** |
| river nodes in view when cells are at 50% | ~1 | **43** |
| cell chunks in view at that zoom | ~2,000,000 | **~20,000** |
| median channel width | 12 cells | 5.0 cells (widest trunk 26) |
| total range | 4.45 decades | 3.95 decades |

### Channel width is no longer only a hydraulics question

Channels came out thin at the new step — 2.7 cells across at the median, a
string of beads rather than a river. Widening `WIDTH_K` alone does not fix
it: at 0.115 river units the headwaters reach 3.3 cells but the trunks reach
33, and in the dense core, where nodes pack at 0.70 river units, those trunks
merge into one mass and the drainage pattern stops reading.

The exponent is the right lever instead. `hw = WIDTH_K · area^WIDTH_EXP`:
`WIDTH_K` is the headwater, `WIDTH_EXP` is how fast it grows. Real channels
run roughly 0.3–0.5 of drainage area, so both 0.35 and 0.30 are defensible
hydraulics — but once the mark is a cell, the exponent also sets how many
cells wide the median channel is. Dropping it to 0.30 and lifting `WIDTH_K`
to 0.125 raises the bottom and leaves the top where it was:

| | headwater | median | p90 | widest |
|---|---|---|---|---|
| 0.090 / 0.35 | 2.6 | 3.8 | — | 26.1 |
| 0.115 / 0.35 | 3.3 | 4.9 | — | 33.3 |
| **0.125 / 0.30** | **3.6** | **5.0** | **10** | **26.0** |

Cost of the wider channels: ~800 cell chunks per node instead of ~540, and
the worst frame goes from 6–10 ms to 12.2 ms.

What actually buries the dense core at u≈2.1 is not the channels — rendering
the river level alone there shows the network perfectly clearly. It is the
**cell level's own wash** on top of it: ~53,000 cell impostors at 29%
opacity. That is arguably correct (at that zoom you are inside the tissue),
but it is worth knowing which of the two is doing it.

### A sprite cap that had been silently wrong since P3

Forcing the whole river level through its impostors and comparing mean frame
luminance gave 0.30 at u=2.2 — the stand-ins were covering a third of what
they stood in for.

`gl_PointSize` is capped by the driver. `ALIASED_POINT_SIZE_RANGE` on this
machine is **511**, and the impostor shader clamped at 480 to stay inside it.
That is harmless while an impostor stands in for something small, which was
the premise. It stopped being true once the river's chunk budget started
overflowing: a 771 px chunk was being drawn as a 480 px sprite, covering 39%
of its area. The README previously claimed this cap "never binds in
practice". It bound.

Impostors are now **instanced quads**, not point sprites: four vertices each
instead of one, still one draw call per level, no size limit.

### The stand-ins were 1.8× too bright, and why

With the cap gone the error reversed: impostors came out 1.80× brighter than
the chunks they replaced at 771 px, 1.36× at 487.

Not a grade — visibly fatter channels. An impostor is one of twelve baked
chunks chosen by drainage area, and drainage area is power-law, so the top
band holds nodes that differ by an order of magnitude. At the probe anchor,
45 of 99 chunks were being handed the three fattest stand-ins. With the old
additive streak this hardly mattered, because the mark's size tracked the
channel; with a cell of fixed size, the whole of a channel's weight is
carried by how many cells there are and how wide they are spread.

Point count is the wrong correction: a narrow channel piles its cells on top
of each other and most of them never reach the frame. The part that does is
`area × (1 − e^−coverage)`, computed per node from the graph alone — no
geometry built — and carried to the GPU as one float per sprite that scales
the whole premultiplied result, coverage included.

| river chunk | 771 px | 485 px | 385 px | 273 px | 172 px |
|---|---|---|---|---|---|
| impostor/chunk, before | 1.80 | 1.36 | — | 1.18 | 1.09 |
| after the per-sprite gain | 1.31 | 0.78 | 0.87 | 0.93 | 0.96 |
| after the 1.10 calibration | — | 0.84 | **0.94** | 0.99 | 1.03 |

`riverVis` is an analytic estimate of what the rasteriser does, so the ratio
of two of them is close but not exact; measured against forced-impostor
frames it ran about 10% low, and `GAIN_CAL` is that correction rather than a
free parameter. The hand-off sits at 400 px, where the two agree to 6%. A
reading taken while the cell level is still at meaningful opacity is
worthless here — the cell contribution is a large constant in both frames and
drags the ratio toward 1.0, which is how 485 px once measured 1.13 and really
was 0.78.

A clamped bake was tried first, on the theory that a float render target
keeps highlights the 8-bit framebuffer clips and mip-filtering then spreads
them. It changed nothing — the output was byte-identical — so that was not
the mechanism. The clamp was kept anyway, since a premultiplied level has no
use for unclamped sums, and the level-1 bake tile went 128 → 512 so a 500 px
impostor is not a 4× stretch.

### The two levels disagreed about colour

The river's cells take their hue from the water and their shape and internal
structure from the photograph. First pass put that mix at 30% photograph —
and at the one zoom where both levels are on screen, the same node's river
came out pale green while its cell chunks came out red. The one place the
composition has to be legible was the one place the levels contradicted each
other.

Now 60% photograph. The galaxy→river colour chain survives the swap because
the tissue palette is itself chosen by the galaxy's warmth: +0.85 for the
palette against +0.88 for the water.

## Levels

| level | unit (absolute) | chunk | points/chunk |
|---|---|---|---|
| 2 galaxy | 10^3.5 ≈ 3162 | 1, sampled from the photograph | 100,000 |
| 1 river | 10^1.5 ≈ 31.6 | one per graph node (1,600, all active), drawing its half-edges as cells | by coverage |
| 0 cell | 1 | a jittered lattice tiling each node's channel | 2,400 |

The two steps are not equal. Cell→river is 10^1.5 and river→galaxy is 100;
P7 says why. Zoom runs `u = log10(H)` from −0.3 to 3.65, 3.95 decades.

A chunk's content does not have to fit its `[-1,1]` box: `LEVEL_EXTENT` says
how far each level reaches, and culling, impostor sprite size and the bake
camera all use it. River chunks reach 1.52 because they own half of each
incident edge; cell chunks reach 1.14 because a cell on the edge of the box
hangs over it.

Chunks are generated on demand from a seeded PRNG (no stored dataset), cached
LRU, and at most 6 are built per frame so fast zooms do not hitch.

Levels cross-fade on `s = log10(unit / H)`: fade in over `[-2.3, -1.6]` (the
cell level over `[-2.18, -1.90]`, see P7), fade out over `[0.05, 0.55]`, so
adjacent levels overlap rather than popping. The fade-out used to run to
0.85; a level at 3% opacity still pays full fill rate and its marks keep
growing as you zoom past it, which cost 21 ms/frame around u=3.4 for a galaxy
nobody could see.

A level's opacity does not reduce its cost — the sprite count depends only on
the zoom — so the cell level's fade-in tail is short on purpose. Carrying it
to u=2.30 at 4% opacity cost 17.2 ms/frame for something invisible.

## Measurements (2560×1720 buffer)

| | |
|---|---|
| max float32 transform error, full sweep | 4.1e-12 px |
| half-edge seam mismatch, 500 edges | 0.0 |
| river density vs galaxy density, log-log slope | 0.899, r=0.988 (1.0 = composed) |
| galaxy↔tissue warmth correlation | +0.85 |
| galaxy↔river warmth correlation | +0.88 |
| river nodes that render nothing | 0 of 1,600 |
| cell/river overlap window | 1.23 decades (was 0.6) |
| river nodes in view when cells are at 50% | 43 (was ~1) |
| cells across a channel: headwater / median / p90 / widest | 3.6 / 5.0 / 10 / 26 |
| cell chunks, per node / total | 796 median / ~1.55M (was 3,000 / ~5M) |
| cell impostor vs the chunk it replaces | 1.00–1.02 |
| river impostor vs the chunk it replaces, at hand-off | 0.94 (was 1.80) |
| peak points on screen | 297,803 |
| peak impostors | 69,644 (in 1 draw call) |
| chunks dropped, full sweep | 0 |
| peak draw calls | 178 |
| worst amortised frame cost | 7–12 ms (GPU flush forced) |
| cell lighting exposure drift | +1.8% |

Frame cost was measured with `gl.readPixels` forcing a flush, not from
requestAnimationFrame — the dev browser pane was backgrounded and throttled to
~2 fps, so a live fps reading from it would be meaningless. **60 fps is not
confirmed on a visible window.**

The worst frame is the cell/river overlap at u≈2.15, where ~70,000 cell
impostors sit over a river at full detail. Five repeats: 9.6 (cold, still
building), 11.8, 11.8, 11.9, 12.2 ms. The galaxy/river crossover at u≈3.2
runs 7.4–9.7. Repeats of one measurement spread about ±20% on an otherwise
idle machine, so this is a range, not a number: **roughly 7–12 ms against a
16.7 ms budget**. The single figures quoted in P5 and earlier sit inside that
spread and the differences between them were never real.

The first reading after a zoom is always higher — it includes chunk building,
which is throttled to 6 per frame and amortises away. Quoting it as the
steady-state cost is a mistake this README has made before; the cold figure
is listed above rather than dropped.

**A correction to the previous figure.** P5 first reported 10.6 ms at the
deepest zoom, u = −0.3. That reading is not stable: `__probe` pins the view to
one anchor, and at the bottom of the range the frame contains only a handful
of chunks, so the cost depends entirely on whether that anchor sits mid-channel
or out at its edge. Repeated at the default anchor it measures 1.8–4.9 ms (8
chunks in view); re-anchored on the centre of the widest trunk — which is the
honest worst case down there — it measures 5.6 ms with 31,200 points. The
10.6 ms was a single reading at an anchor that happened to be busy, and it
should not have been quoted as *the* worst frame without repeats.

Full↔impostor fidelity is measured by rendering the identical frame twice —
once normally, once with the whole level forced through its impostors — and
comparing mean luminance. Both paths must be unconstrained for the comparison
to mean anything: with the chunk budget left at its shipping value the "full"
frame is itself part impostor, which is how an earlier reading of 1.05 came
out of a level that was really 1.80× off.

| | cell level | river level |
|---|---|---|
| hand-off threshold | 170 px | 400 px |
| impostor/chunk at hand-off | 1.00–1.02 | 0.94 |
| further out | 1.00 at u=1.4, 1.9 | 0.99 / 1.03 / 1.01 at 273 / 172 / 109 px |

Swept across the river hand-off (u = 2.22 → 2.38), mean frame luminance runs
61.2, 59.2, 56.9 | 50.6, 49.6 — a −7% residual step over a −4%-per-step
trend. That is the 0.94 above, and it is the largest threshold step in the
piece.

The cell level's hand-off to the river at u≈2.18 is clean: cell weight runs
0.71 → 0.20 → 0.03 → 0 over u = 2.00…2.18 while the river is already at
weight 1, and mean frame luminance rises monotonically through it.

**A correction to the previous claim.** The README used to say the 480 px
sprite cap in the impostor vertex shader "never binds in practice". It did,
from P3 onward, wherever a level's chunk budget overflowed while its chunks
were still large — for the river that is u ∈ [2.5, 2.9]. Impostors are
instanced quads now and have no cap. See P7.

Dev probes in the console: `__probe(u)`, `__sweep()`, `__bench(seconds)`,
`__dbg`, `__dbg.content`, `__dbg.fields`, and
`__dbg.cascade.forceImpostor = <level | -1>`.

## Known gaps (deliberate)

- **Impostors are statistical, and for a network that now shows.** A chunk's
  impostor is a picture of some *other* chunk, so its half-edges radiate in
  the wrong directions and do not meet its neighbours'. Tissue is forgiving
  of this — any patch of tissue looks like any other patch. A drainage
  network is not: its whole meaning is connectivity. Between u≈2.45 and 2.9,
  where every river chunk is a stand-in 100–350 px across, the network reads
  as disconnected worms rather than as a river system. This was true from P3
  on and was invisible while the marks were thin additive streaks. Sprites
  now carry their chunk's colour, drainage area and visible-ink gain, which
  covers every attribute except the one that matters here. Fixes, none cheap:
  bake per-chunk impostors lazily into an LRU atlas; or choose the sprite's
  orientation to align its dominant edge bearing with the node's own.
- **No refinement inside a river.** A river chunk's point count is fixed once
  built, so zooming between u=2.2 and u=1.4 reveals no new cells, just bigger
  ones.
- **The river's cells are not the cell level's cells.** They are drawn from
  the same atlas, with the same tissue palette and the same lighting, and at
  the P7 step they are the same absolute size — but they are a different set
  of points, placed by the river's own coverage budget rather than by the
  cell level's lattice. In the overlap the two are superimposed, not
  identified.
- **u≈1.6–2.0 is still sparse** away from the bright regions. Inside them the
  adaptive spacing has largely closed it; out in the faint country it is now
  sparse *because the galaxy is faint there*, which is the intended answer
  rather than a gap.
- **The cell level is impostors over most of its range.** Full detail needs
  170 px per chunk, which is u ≲ 0.6; above that the whole tissue is
  stand-ins. That is the right trade (it is also what removed the banding),
  but it means the photographs are only resolved in the bottom third of the
  cell level's zoom range.
- **The cell assets are 3.6 MB** (2.5 colour + 1.1 normals) and load before
  the first frame. The river level now binds the same two textures, so the
  cost is shared rather than doubled.
- **Enumeration, not drawing, is the cell level's ceiling.** The per-frame
  walk over cell stations costs roughly 0.4 µs each, so ~30,000 impostors is
  ~12 ms of CPU before a single one is drawn. That is what sets the cell
  level's fade-out, not fill rate or the impostor cap.
- **The level-1 bake atlas is 2048×1536 half-float** (~25 MB of VRAM), built
  once at startup.
- **Mipmaps bleed between atlas tiles** at small sprite sizes. The tiles have
  near-transparent borders so it reads as a faint fringe rather than a wrong
  cell.
- **The brightest decile's channels cover ~90% of the ground.** That is the
  density relationship working, not a bug, and the marks have texture again —
  but there is no headroom left there for more river.
- **Wide trunks overlap their neighbours on the ground**, and in the dense
  core they merge. Capping the width would buy tidiness at the cost of the
  `area^0.30` hydraulic scaling, so it is left alone and recorded instead —
  but it is the reason channel width could not simply be scaled up.
- **The total range lost half a decade**, 4.45 → 3.95, which is what closing
  the cell→river step costs. Adding a fourth level would buy it back, but
  there is no fourth photograph set to build one from.
- **Faint regions are empty, not sparse.** Nodes can no longer land where
  they would not connect, so the outskirts have *no* rivers rather than a few
  isolated ones. That is the intended reading of a faint galaxy, but it is a
  choice, not a measurement.
- **Orthographic, no parallax.** Points carry a `z` and `F` tracks the sheet
  height, so switching to perspective is a camera change, not a rewrite.
- **The depth maps only light; they do not displace.** Cells are still flat
  sprites with a normal, so there is no silhouette change and no
  self-occlusion — at these sprite sizes neither would be visible, but it is
  a shading trick, not geometry.
- **One key light, no shadows between cells.** A cell never darkens its
  neighbour, so a thick field reads as evenly lit however deep it is.
- **`galaxy/` and `river/` are unused.** Those two levels are still driven by
  the single hand-picked reference photograph each, not by their 500-image
  sets.

## Files

```
tools/extract_fields.py  offline: photographs -> density fields
tools/extract_cells.py   offline: 500 micrographs + their depth maps
                         -> one cell sprite atlas + its normal map
refs/                    the two reference photographs
public/fields/           the extracted fields (254 KB, committed)
public/cells/            cell_atlas.png + cell_normal.png + cells.json
                         (3.6 MB, committed)

src/rng.js       seeded hash / PRNG / 1D value noise
src/fields.js    field loading + CDF samplers
src/cells.js     cell atlas loading + tile metadata
src/impostor.js  atlas baking + the one-draw-call instanced-quad impostors
src/content.js   the 3-level hierarchy + chunk queries
src/cascade.js   the scale cascade, float64 transforms, LOD, marks, cache
src/main.js      renderer, log-zoom controls, HUD, scale slider, dev probes
```
