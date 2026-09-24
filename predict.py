"""Scan images (any size) for numbers.

    python predict.py ilan_foto.jpg
    python predict.py klasor/ --threshold 0.6 --vis cikti/

Large images are scanned as overlapping 128x128 tiles at several scales, so
both a big number across the whole photo and a small one in a corner are seen.
The image score is the max tile score.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

TILE, STRIDE = 128, 96
SCALES = (128, 384, 768)          # longer side after resize
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "digit_detector_cnn.h5")


def tiles(img):
    """Yield (tile_uint8[128,128], box_in_original_coords)."""
    W, H = img.size
    for side in SCALES:
        if side > 128 and max(W, H) < side * 0.6:   # don't blow tiny images up
            continue
        s = side / max(W, H)
        im = img.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
        w, h = max(im.width, TILE), max(im.height, TILE)
        canvas = Image.new("L", (w, h), int(np.asarray(im).mean()))
        canvas.paste(im, (0, 0))
        xs = list(range(0, w - TILE + 1, STRIDE)) or [0]
        ys = list(range(0, h - TILE + 1, STRIDE)) or [0]
        if xs[-1] != w - TILE:
            xs.append(w - TILE)
        if ys[-1] != h - TILE:
            ys.append(h - TILE)
        arr = np.asarray(canvas)
        for y in ys:
            for x in xs:
                yield arr[y:y + TILE, x:x + TILE], (x / s, y / s, (x + TILE) / s, (y + TILE) / s)


class Detector:
    def __init__(self, model_path=DEFAULT_MODEL):
        import gpu_setup  # noqa: F401
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)

    def scan(self, img):
        img = ImageOps.exif_transpose(img).convert("L")
        t = list(tiles(img))
        x = np.stack([a for a, _ in t])[..., None]
        p = self.model.predict(x, batch_size=128, verbose=0)[:, 0]
        return p, [b for _, b in t]

    def score(self, path):
        p, boxes = self.scan(Image.open(path))
        i = int(p.argmax())
        return float(p[i]), boxes[i], p, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--vis", help="klasör: numara bulunan bölgeleri işaretli kopyaları kaydet")
    a = ap.parse_args()

    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in EXTS)
        else:
            files.append(p)
    if not files:
        sys.exit("görsel bulunamadı")

    det = Detector(a.model)
    if a.vis:
        os.makedirs(a.vis, exist_ok=True)
    for f in files:
        best, box, p, boxes = det.score(f)
        verdict = "NUMARA VAR" if best >= a.threshold else "numara yok"
        print(f"{verdict:11s} {best:.3f}  {f}")
        if a.vis:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            d = ImageDraw.Draw(im)
            for s, b in zip(p, boxes):
                if s >= a.threshold:
                    d.rectangle(b, outline=(255, int(255 * (1 - s)), 0), width=max(2, im.width // 300))
            im.save(os.path.join(a.vis, os.path.splitext(os.path.basename(f))[0] + "_vis.jpg"), quality=90)


if __name__ == "__main__":
    main()
