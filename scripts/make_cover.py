"""Render the Data Voyage book cover from the author's photograph.

Usage:
    python scripts/make_cover.py path/to/photo.jpg

Writes four files:
    assets/cover.jpg              2400 x 3600 master (6 x 9 in at 400 dpi)
    assets/cover-1600x2400.jpg    Leanpub, Google Play, Apple Books (2:3)
    assets/cover-kdp.jpg          1600 x 2560 for Amazon KDP (its ideal 1.6:1)
    assets/cover-web.jpg          600 x 900 for the README

The photograph is cropped to a 2:3 portrait, darkened at the top and bottom
for legibility, and set with Avenir Next (macOS). Pass --font to use another
TrueType collection on other systems.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

W, H = 2400, 3600
ACCENT = (226, 38, 44)  # the red of the road barriers in the photograph
WHITE = (250, 248, 245)
MUTED = (205, 205, 210)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets"


def font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=index)


def crop_portrait(img: Image.Image, center_x: float, size: tuple[int, int]) -> Image.Image:
    """Crop the largest portrait window with *size*'s ratio, centred near *center_x* (0-1)."""
    w, h = img.size
    cw = int(h * size[0] / size[1])
    x0 = int(min(max(center_x * w - cw / 2, 0), w - cw))
    return img.crop((x0, 0, x0 + cw, h)).resize(size, Image.LANCZOS)


def vertical_shade(size: tuple[int, int], stops: list[tuple[float, int]]) -> Image.Image:
    """Black overlay whose alpha is interpolated between (position, alpha) stops."""
    w, h = size
    column = Image.new("L", (1, h))
    px = column.load()
    for y in range(h):
        t = y / (h - 1)
        for (t0, a0), (t1, a1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                px[0, y] = int(a0 + (a1 - a0) * (t - t0) / (t1 - t0))
                break
    alpha = column.resize((w, h))
    overlay = Image.new("RGBA", (w, h), (8, 8, 12, 0))
    overlay.putalpha(alpha)
    return overlay


def spaced(draw: ImageDraw.ImageDraw, xy, text, fnt, fill, tracking: float, anchor="mm"):
    """Draw letter-spaced text centred on *xy* (tracking is a fraction of the font size)."""
    gap = fnt.size * tracking
    widths = [draw.textlength(ch, font=fnt) for ch in text]
    total = sum(widths) + gap * (len(text) - 1)
    x = xy[0] - total / 2 if anchor == "mm" else xy[0]
    for ch, w in zip(text, widths):
        draw.text((x, xy[1]), ch, font=fnt, fill=fill, anchor="lm")
        x += w + gap


def render(photo: Path, font_path: str, H: int = H) -> Image.Image:
    """Render at width W and height *H*; the title block is fixed, the footer follows H."""
    img = Image.open(photo).convert("RGB")
    img = crop_portrait(img, center_x=0.56, size=(W, H))
    img = ImageEnhance.Contrast(img).enhance(1.08)

    canvas = img.convert("RGBA")
    canvas.alpha_composite(
        vertical_shade(
            (W, H), [(0.0, 235), (0.30, 190), (0.46, 40), (0.72, 0), (0.86, 120), (1.0, 235)]
        )
    )
    d = ImageDraw.Draw(canvas)

    heavy = font(font_path, 8, 330)
    medium = font(font_path, 5, 92)
    demi = font(font_path, 2, 64)
    small = font(font_path, 5, 50)
    author = font(font_path, 2, 118)

    cx = W / 2
    spaced(d, (cx, 330), "PRODUCTION AI ENGINEERING", demi, ACCENT, 0.32)

    spaced(d, (cx, 700), "DATA", heavy, WHITE, 0.10)
    spaced(d, (cx, 1030), "VOYAGE", heavy, WHITE, 0.10)

    d.rectangle((cx - 170, 1255, cx + 170, 1267), fill=ACCENT)

    d.text((cx, 1390), "Building Real AI Systems", font=medium, fill=WHITE, anchor="mm")
    d.text((cx, 1510), "from Data to Deployment", font=medium, fill=WHITE, anchor="mm")

    spaced(
        d,
        (cx, H - 470),
        "ONE PROJECT  ·  24 CHAPTERS  ·  RAW DATA TO DEPLOYED API",
        small,
        MUTED,
        0.12,
    )
    spaced(d, (cx, H - 290), "IRFAN ALI", author, WHITE, 0.28)

    return canvas.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", type=Path)
    ap.add_argument("--font", default="/System/Library/Fonts/Avenir Next.ttc")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    cover = render(args.photo, args.font)
    cover.save(OUT_DIR / "cover.jpg", quality=92, dpi=(400, 400))
    cover.resize((1600, 2400), Image.LANCZOS).save(OUT_DIR / "cover-1600x2400.jpg", quality=92)
    cover.resize((600, 900), Image.LANCZOS).save(OUT_DIR / "cover-web.jpg", quality=85)
    kdp = render(args.photo, args.font, H=3840)  # 2400 x 3840 is 1.6:1
    kdp.resize((1600, 2560), Image.LANCZOS).save(OUT_DIR / "cover-kdp.jpg", quality=92)
    print(f"Wrote cover files to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
