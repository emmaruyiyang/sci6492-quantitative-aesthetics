"""The Embedding Map as a PDF (the Export PDF button of the page, /api/export/pdf).

The page sends what it draws, measured on the screen in its own pixels (y
down, 0,0 = top left corner of the map), so the PDF shows exactly the current
view: zoom, pan, quantized tiles, highlighted cluster, interpolation...

  spec = {"width", "height",            size of the map, px
          "background",                 map color, "#rrggbb" or "rgb(r, g, b)"
          "title", "subtitle",          text above the map
          "legend": [{"color", "label"}],   below the map
          "marks": [...]}               in drawing order, later ones on top

  mark = {"kind": "image" | "dot" | "anchor" | "line",
          "x", "y", "w", "h",           box on the map, px
          "jpeg": bytes,                image: its picture, already cropped to the box's shape
          "color", "border",            border (anchor: frame) color and width, px
          "ring",                       px of map color around the mark (where marks overlap)
          "round",                      drawn as a circle
          "outline",                    px of black outline outside the box (picked images)
          "opacity", "text",            anchor: class name
          "points"}                     line: [[x, y], ...]

The PDF is written directly (no library): every image is a JPEG stored as it
is (DCTDecode), shapes and text are vector, the font is Courier like the page.
"""

import io
import math
import re

from PIL import Image

PT = 0.75               # PDF points per CSS pixel (72 / 96)
MARGIN = 36             # page margin, points
INK = (0, 0, 0)
KAPPA = 0.5523          # Bezier control distance for a quarter circle of radius 1


def rgb(color: str) -> tuple[float, float, float]:
    """"#rrggbb" or "rgb(r, g, b)" -> 0..1 each"""
    color = color.strip()
    if color.startswith("#"):
        return tuple(int(color[k:k + 2], 16) / 255 for k in (1, 3, 5))
    r, g, b = (float(v) for v in re.findall(r"[\d.]+", color)[:3])
    return r / 255, g / 255, b / 255


def text_bytes(text: str) -> bytes:
    """A PDF string: latin-1 (the standard fonts), with ( ) \\ escaped"""
    raw = text.encode("latin-1", "replace")
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def wrap(text: str, size: float, width: float) -> list[str]:
    """Lines of Courier text (0.6 em per character) at most `width` points
    long, broken between words"""
    per_line = max(1, int(width / (0.6 * size)))
    lines = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= per_line:
            lines[-1] += " " + word
        else:
            lines += [word[k:k + per_line] for k in range(0, len(word), per_line)]
    return lines


def jpeg_size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


class Canvas:
    """The content stream of the page, and the resources it uses"""

    def __init__(self):
        self.ops = []
        self.images = []        # JPEG bytes, named /Im0, /Im1...
        self.alphas = {}        # opacity -> /GS name
        self.forms = []         # (content, bbox) of the transparency groups, named /Fm0, /Fm1...

    def __call__(self, *parts):
        self.ops.append(" ".join(f"{p:.2f}" if isinstance(p, float) else str(p) for p in parts))

    def fill(self, color):
        self(*map(float, color), "rg")

    def stroke(self, color, width):
        self(*map(float, color), "RG", float(width), "w")

    def opacity(self, alpha):
        if alpha < 0.999:
            name = self.alphas.setdefault(round(alpha, 3), f"GS{len(self.alphas)}")
            self(f"/{name}", "gs")

    def group(self, start, alpha, bbox):
        """Everything drawn since len(self.ops) was `start` becomes one
        transparency group painted with `alpha`: faded as a whole, like CSS
        opacity (its overlapping parts don't add up)"""
        name = f"Fm{len(self.forms)}"
        self.forms.append((self.ops[start:], bbox))
        del self.ops[start:]
        self("q")
        self.opacity(alpha)
        self(f"/{name}", "Do", "Q")

    def rect(self, x, y, w, h):
        self(float(x), float(y), float(w), float(h), "re")

    def ellipse(self, x, y, w, h):
        rx, ry, cx, cy = w / 2, h / 2, x + w / 2, y + h / 2
        kx, ky = KAPPA * rx, KAPPA * ry
        self(float(cx + rx), float(cy), "m")
        self(float(cx + rx), float(cy + ky), float(cx + kx), float(cy + ry), float(cx), float(cy + ry), "c")
        self(float(cx - kx), float(cy + ry), float(cx - rx), float(cy + ky), float(cx - rx), float(cy), "c")
        self(float(cx - rx), float(cy - ky), float(cx - kx), float(cy - ry), float(cx), float(cy - ry), "c")
        self(float(cx + kx), float(cy - ry), float(cx + rx), float(cy - ky), float(cx + rx), float(cy), "c")
        self("h")

    def shape(self, mark, grow=0.0):
        """The mark's box (or circle), grown by `grow` px on every side"""
        x, y, w, h = mark["x"] - grow, mark["y"] - grow, mark["w"] + 2 * grow, mark["h"] + 2 * grow
        (self.ellipse if mark.get("round") else self.rect)(x, y, w, h)

    def image(self, data, x, y, w, h):
        """In the map's coordinates (y down): the matrix flips the image upright"""
        name = f"Im{len(self.images)}"
        self.images.append(data)
        self("q", float(w), 0, 0, float(-h), float(x), float(y + h), "cm", f"/{name}", "Do", "Q")

    def text(self, string, x, y, size, bold=False, color=INK):
        """Baseline at x, y in page coordinates (y up)"""
        self.fill(color)
        self("BT", "/F2" if bold else "/F1", float(size), "Tf", float(x), float(y), "Td",
             text_bytes(string).decode("latin-1"), "Tj", "ET")


def draw_mark(c: Canvas, mark: dict, background) -> None:
    kind = mark["kind"]
    if kind == "line":
        if len(mark["points"]) >= 2:
            c.stroke(INK, 2.0)
            c("1 j 1 J")                    # round joins and caps
            (x0, y0), *rest = mark["points"]
            c(float(x0), float(y0), "m")
            for x, y in rest:
                c(float(x), float(y), "l")
            c("S")
        return
    c("q")
    if mark.get("ring"):
        # box-shadow: a ring of map color just outside the mark
        c.stroke(background, mark["ring"])
        c.shape(mark, mark["ring"] / 2)
        c("S")
    if kind == "image" and mark.get("jpeg"):
        c("q")
        c.shape(mark)
        c("W n")                            # clip: round images stay round
        c.image(mark["jpeg"], mark["x"], mark["y"], mark["w"], mark["h"])
        c("Q")
    elif kind in ("dot", "anchor", "image"):
        c.fill(rgb(mark["color"]) if kind == "dot" else (INK if kind == "anchor" else (1, 1, 1)))
        c.shape(mark)
        c("f")
    border = mark.get("border", 0)
    if border > 0:
        c.stroke(rgb(mark["color"]), border)
        c.shape(mark, -border / 2)
        c("S")
    if kind == "anchor" and mark.get("text"):
        draw_anchor_text(c, mark)
    if mark.get("outline"):
        c.stroke(INK, mark["outline"])
        c.shape(mark, mark["outline"] / 2)
        c("S")
    c("Q")


def draw_anchor_text(c: Canvas, mark: dict) -> None:
    """The class name, white, centered, broken anywhere to fit (as on the page)"""
    size = mark.get("font_size", 9.0)
    inner = mark["w"] - 2 * mark.get("border", 0) - 2
    per_line = max(1, int(inner / (0.6 * size)))        # Courier: every character is 0.6 em wide
    name = mark["text"]
    lines = [name[k:k + per_line] for k in range(0, len(name), per_line)]
    lines = lines[:max(1, int((mark["h"] - 2 * mark.get("border", 0)) / (1.05 * size)))]
    top = mark["y"] + mark["h"] / 2 - len(lines) * 1.05 * size / 2
    c("q")
    c.shape(mark)
    c("W n")
    for k, line in enumerate(lines):
        x = mark["x"] + (mark["w"] - 0.6 * size * len(line)) / 2
        baseline = top + (k + 0.8) * 1.05 * size
        # the map is flipped (y down): flip the text back upright around its baseline
        c("q", 1, 0, 0, -1, float(x), float(baseline), "cm")
        c.text(line, 0.0, 0.0, size, bold=True, color=(1, 1, 1))
        c("Q")
    c("Q")


def make_pdf(spec: dict) -> bytes:
    width, height = float(spec["width"]), float(spec["height"])
    background = rgb(spec.get("background", "#ececec"))
    legend = spec.get("legend") or []

    page_w = width * PT + 2 * MARGIN
    # legend: as many entries per row as fit
    entry_w = max([10 + 0.6 * 8 * len(e["label"]) + 14 for e in legend] or [0])
    per_row = max(1, int((page_w - 2 * MARGIN) / entry_w)) if legend else 1
    legend_h = 14 * math.ceil(len(legend) / per_row) + 8 if legend else 0
    # the text above the map, broken into lines that fit the page
    header = [(line, 11.0, True) for line in wrap(spec.get("title", ""), 11.0, page_w - 2 * MARGIN)]
    header += [(line, 7.5, False) for line in wrap(spec.get("subtitle", ""), 7.5, page_w - 2 * MARGIN)]
    header_h = sum(1.3 * size for _, size, _ in header) + 8
    page_h = 2 * MARGIN + header_h + height * PT + legend_h
    map_top = page_h - MARGIN - header_h              # page y of the map's top edge

    c = Canvas()
    y = page_h - MARGIN
    for line, size, bold in header:
        y -= 1.3 * size
        c.text(line, MARGIN, y + 0.3 * size, size, bold=bold, color=INK if bold else (0.27, 0.27, 0.27))

    # the map, clipped to its square, in its own pixels with y down
    c("q")
    c.rect(MARGIN, map_top - height * PT, width * PT, height * PT)
    c("W n")
    c.fill(background)
    c.rect(MARGIN, map_top - height * PT, width * PT, height * PT)
    c("f")
    c(PT, 0, 0, -PT, float(MARGIN), float(map_top), "cm")
    for mark in spec["marks"]:
        start = len(c.ops)
        draw_mark(c, mark, background)
        if mark.get("opacity", 1.0) < 0.999:
            c.group(start, mark["opacity"], (-50, -50, width + 50, height + 50))
    c("Q")

    for k, entry in enumerate(legend):
        x = MARGIN + (k % per_row) * entry_w
        y = map_top - height * PT - 14 - 14 * (k // per_row)
        c.fill(rgb(entry["color"]))
        c.ellipse(x, y - 1, 7, 7)
        c("f")
        c.text(entry["label"], x + 10, y, 8.0, color=(0.27, 0.27, 0.27))

    return write_pdf(c, page_w, page_h)


def write_pdf(c: Canvas, page_w: float, page_h: float) -> bytes:
    objects = []                    # bytes of object 1, 2, ...

    def add(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    def stream(header: str, data: bytes) -> bytes:
        return f"<< {header} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"

    images = []
    for data in c.images:
        w, h = jpeg_size(data)
        images.append(add(stream(f"/Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
                                 f"/BitsPerComponent 8 /Filter /DCTDecode", data)))
    fonts = [add(f"<< /Type /Font /Subtype /Type1 /BaseFont /{name} /Encoding /WinAnsiEncoding >>".encode())
             for name in ("Courier", "Courier-Bold")]
    states = " ".join(f"/{name} << /ca {alpha} /CA {alpha} >>" for alpha, name in c.alphas.items())

    def resources(xobjects: list[str]) -> int:
        return add(f"<< /Font << /F1 {fonts[0]} 0 R /F2 {fonts[1]} 0 R >> /XObject << {' '.join(xobjects)} >> "
                   f"/ExtGState << {states} >> >>".encode())

    # the groups use the images, the page uses the images and the groups
    image_names = [f"/Im{k} {n} 0 R" for k, n in enumerate(images)]
    inner = resources(image_names)
    forms = [add(stream(f"/Type /XObject /Subtype /Form /BBox [{' '.join(f'{v:.2f}' for v in bbox)}] "
                        f"/Group << /S /Transparency >> /Resources {inner} 0 R",
                        "\n".join(ops).encode("latin-1")))
             for ops, bbox in c.forms]
    page_resources = resources(image_names + [f"/Fm{k} {n} 0 R" for k, n in enumerate(forms)])
    content = add(stream("", "\n".join(c.ops).encode("latin-1")))
    pages = len(objects) + 2        # the page refers to its parent, written right after it
    page = add(f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {page_w:.2f} {page_h:.2f}] "
               f"/Resources {page_resources} 0 R /Contents {content} 0 R >>".encode())
    add(f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>".encode())
    catalog = add(f"<< /Type /Catalog /Pages {pages} 0 R >>".encode())

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, data in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(f"{n} 0 obj\n".encode() + data + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()
