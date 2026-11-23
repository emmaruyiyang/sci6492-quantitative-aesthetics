import sys
from pathlib import Path
from PIL import Image, ImageDraw
root = Path(sys.argv[1]); out = Path(sys.argv[2]); n = 100; cols = 10; C = 150
files = sorted(root.glob("*.jpg"))
step = max(1, len(files) // n)
pick = files[::step][:n]
rows = (len(pick) + cols - 1) // cols
canvas = Image.new("RGB", (cols * C, 20 + rows * C), "white")
ImageDraw.Draw(canvas).text((4, 5), f"{root.name}: {len(pick)} sampled of {len(files)}", fill="black")
for i, f in enumerate(pick):
    im = Image.open(f).convert("RGB"); im.thumbnail((C, C))
    canvas.paste(im, ((i % cols) * C + (C - im.width)//2, 20 + (i // cols) * C + (C - im.height)//2))
canvas.save(out, quality=85); print(out)
