"""Build the FID_v2 thumbnail dataset from the FID500 product images.

Each FID500 product shot is composited into a deterministic influencer-style
thumbnail: a gradient background derived from the product's own dominant colour,
the product on a white card with a soft shadow, a category badge, a "NEW" corner
marker, the product name and sub-category text, and a rounded frame. Every rule
is fixed, so the same input manifest always yields byte-identical thumbnails.

These thumbnails become the FID_v2 real set X. The evaluation reconstructs each
thumbnail with the existing img2img pipeline (thumbnail + per-image prompt) and
measures FID(X, reconstructed). The thumbnails are NOT AI-generated — they are a
rule-based composite independent of the SDXL pipeline under test.

Source: product images derived from AI Hub 「상품 이미지」 (aihub.or.kr/aidata/34145,
NIA). When publishing, identify the source as an outcome of the NIA AI training
-data program (NIA 사업결과).
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "fid"
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "fid_v2"
FONT_BOLD = Path.home() / ".fonts" / "NotoSansKR-Bold.otf"
FONT_REGULAR = Path.home() / ".fonts" / "NotoSansKR-Regular.otf"

CANVAS = 1024
RULE_VERSION = "fid_v2-thumbnail-v2-productcolor"

# 대분류(원 manifest) -> 통일 카테고리명. 폴더/배지/화면 모두 이 이름을 쓴다.
CATEGORY_MAP = {"이/미용": "뷰티"}


def category_name(raw: str) -> str:
    return CATEGORY_MAP.get(raw, raw)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dominant_hsv(image: Image.Image) -> tuple[float, float]:
    """Saturation-weighted circular-mean hue of the product's coloured pixels,
    ignoring the near-white studio background and greyscale text/shadow."""
    small = image.convert("RGB").resize((96, 96))
    pairs = []
    saturation_sum = 0.0
    for r, g, b in small.getdata():
        high, low = max(r, g, b), min(r, g, b)
        if high > 235 and (high - low) < 18:          # near-white background
            continue
        h, s, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if s < 0.12:                                    # greyscale
            continue
        pairs.append((h, s))
        saturation_sum += s
    if not pairs:
        return 0.58, 0.20                               # muted-blue fallback
    x = sum(math.cos(2 * math.pi * h) * s for h, s in pairs)
    y = sum(math.sin(2 * math.pi * h) * s for h, s in pairs)
    hue = (math.atan2(y, x) / (2 * math.pi)) % 1.0
    saturation = min(0.55, saturation_sum / len(pairs))
    return hue, saturation


def hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def vertical_gradient(top, bottom, width, height):
    column = Image.new("RGB", (1, height))
    for y in range(height):
        t = y / (height - 1)
        column.putpixel((0, y), tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return column.resize((width, height))


def fit_font(text, font_path, max_width, start_size):
    size = start_size
    while size > 20:
        font = ImageFont.truetype(str(font_path), size)
        if font.getbbox(text)[2] <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(str(font_path), 20)


def resolve_product_path(source_root: Path, raw_category: str, item_no: str) -> Path:
    """Product folders on disk use the unified category name (뷰티 etc.)."""
    folder = category_name(raw_category).replace("/", "_")
    return source_root / "input" / folder / f"{item_no}.jpg"


def compose(row, source_root: Path) -> Image.Image:
    display = category_name(row["대분류"])
    name = " ".join(row["상품명"].split())
    sub = f"{row['중분류']} · {row['소분류']}"

    product = Image.open(resolve_product_path(source_root, row["대분류"], row["item_no"])).convert("RGB")
    hue, sat = dominant_hsv(product)
    bg_top = hsv(hue, sat * 0.35, 0.97)
    bg_bottom = hsv(hue, sat * 0.70, 0.86)
    badge = hsv(hue, min(0.75, sat + 0.25), 0.52)
    accent = hsv(hue, min(0.85, sat + 0.30), 0.34)

    canvas = Image.new("RGB", (CANVAS, CANVAS))
    canvas.paste(vertical_gradient(bg_top, bg_bottom, CANVAS, CANVAS), (0, 0))
    draw = ImageDraw.Draw(canvas)

    product.thumbnail((620, 560), Image.LANCZOS)
    px = (CANVAS - product.width) // 2
    py = 150
    shadow = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [px + 12, py + 18, px + product.width + 12, py + product.height + 18], radius=18, fill=(0, 0, 0, 60))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    canvas.paste(shadow, (0, 0), shadow)
    draw.rounded_rectangle(
        [px - 16, py - 16, px + product.width + 16, py + product.height + 16], radius=18, fill=(255, 255, 255))
    canvas.paste(product, (px, py))

    badge_font = ImageFont.truetype(str(FONT_BOLD), 30)
    badge_width = badge_font.getbbox(display)[2]
    badge_box = [64, 64, 64 + badge_width + 48, 64 + 52]
    badge_cy = (badge_box[1] + badge_box[3]) / 2
    draw.rounded_rectangle(badge_box, radius=26, fill=badge)
    draw.text(((badge_box[0] + badge_box[2]) / 2, badge_cy), display,
              font=badge_font, fill=(255, 255, 255), anchor="mm")

    text_y = py + product.height + 70
    draw.rounded_rectangle(
        [(CANVAS - 120) // 2, text_y - 26, (CANVAS + 120) // 2, text_y - 18], radius=4, fill=badge)
    name_font = fit_font(name, FONT_BOLD, CANVAS - 160, 62)
    draw.text(((CANVAS - name_font.getbbox(name)[2]) // 2, text_y), name, font=name_font, fill=accent)
    sub_font = ImageFont.truetype(str(FONT_REGULAR), 30)
    draw.text(((CANVAS - sub_font.getbbox(sub)[2]) // 2, text_y + name_font.size + 18), sub, font=sub_font, fill=(0x55, 0x5F, 0x5B))
    draw.rounded_rectangle([18, 18, CANVAS - 18, CANVAS - 18], radius=24, outline=badge, width=4)
    return canvas


def build(source_root: Path, out_root: Path) -> dict:
    rows = list(csv.DictReader((source_root / "manifest.csv").open(encoding="utf-8")))
    if not rows:
        raise SystemExit("source manifest is empty")
    for font in (FONT_BOLD, FONT_REGULAR):
        if not font.is_file():
            raise SystemExit(f"font missing: {font}")

    input_dir = out_root / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)

    manifest_rows = []
    counts: dict[str, int] = {}
    for row in rows:
        display = category_name(row["대분류"])
        folder = display.replace("/", "_")
        target_dir = input_dir / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{row['item_no']}.png"
        compose(row, source_root).save(out_path, "PNG")
        counts[display] = counts.get(display, 0) + 1
        manifest_rows.append({
            "item_no": row["item_no"],
            "대분류": display,
            "중분류": row["중분류"],
            "소분류": row["소분류"],
            "상품명": row["상품명"],
            "source_product": str(Path("input") / folder / f"{row['item_no']}.jpg"),
            "thumbnail": str(Path("input") / folder / f"{row['item_no']}.png"),
            "prompt": row["prompt"],
            "sha256": sha256(out_path),
        })

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    selection = {
        "dataset": "FID_v2 thumbnail set",
        "derived_from": "dataset/fid (FID500 product images)",
        "source_dataset": {
            "name": "AI Hub 상품 이미지",
            "url": "https://aihub.or.kr/aidata/34145",
            "attribution": "NIA AI 학습용 데이터 구축사업 결과",
            "builder": "NIA",
            "year": 2020,
        },
        "thumbnail_rule_version": RULE_VERSION,
        "thumbnail_rule": "deterministic composite: gradient background derived from "
                          "the product's dominant colour, product on white card with "
                          "soft shadow, category badge, NEW marker, product-name + "
                          "sub-category text, rounded frame",
        "category_map": CATEGORY_MAP,
        "count": len(manifest_rows),
        "category_counts": counts,
        "canvas": f"{CANVAS}x{CANVAS} RGB PNG",
        "not_ai_generated": True,
        "manifest_sha256": sha256(manifest_path),
    }
    (out_root / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build(args.source_root.expanduser().absolute(), args.out.expanduser().absolute())
    print(json.dumps(summary, ensure_ascii=False))
    print(f"FID_V2_THUMBNAILS_OK count={summary['count']} counts={summary['category_counts']}")


if __name__ == "__main__":
    main()
