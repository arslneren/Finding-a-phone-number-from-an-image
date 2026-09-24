"""Belirli bir numarayı (varsayılan 05340012445) her Unicode rakam setiyle
1024x720 fotoğraflara yazar ve predict.Detector ile yakalanıp yakalanmadığını ölçer.

    python test_phone.py --number 05340012445
"""
import argparse
import glob
import os
import random

from PIL import Image, ImageDraw

from fonts import build_coverage, load_font
from predict import Detector
from synth import charset_table
from unicode_sets import DIGIT_SETS, LOOKALIKES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "samples", "phone_test")


def variants(number):
    """(isim, metin) - her set için düz, gruplu ve karışık yazımlar."""
    groups = [number[:4], number[4:7], number[7:9], number[9:]]
    out = []
    for name, table in DIGIT_SETS.items():
        conv = "".join(table[int(c)] or DIGIT_SETS["ascii"][int(c)] for c in number)
        out.append((name, "bitisik", conv))
        g = [conv[:4], conv[4:7], conv[7:9], conv[9:]]
        out.append((name, "gruplu", " ".join(g)))
    r = random.Random(0)
    names = list(DIGIT_SETS)
    mixed = "".join((DIGIT_SETS[r.choice(names)][int(c)] or c) for c in number)
    out.append(("karisik_setler", "bitisik", mixed))
    look = "".join(r.choice(LOOKALIKES[int(c)]) if i % 3 == 1 else c for i, c in enumerate(number))
    out.append(("harf_taklidi", "bitisik", look))
    out.append(("ascii", "tire", "-".join(groups)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--number", default="05340012445")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cov = build_coverage(charset_table())
    photos = sorted(glob.glob(os.path.join(HERE, "data", "flower_photos", "*", "*.jpg")))
    r = random.Random(1)
    det = Detector()

    rows = []
    for dset, style, text in variants(a.number):
        faces = cov.get("d_" + dset) or cov["d_ascii"]
        if not cov.get("d_" + dset) and dset not in ("karisik_setler", "harf_taklidi"):
            continue  # sistemde font yok (ör. segmented)
        for size in (24, 48):  # küçük ve büyük yazı
            face = tuple(r.choice(faces if dset in DIGIT_SETS else cov["d_ascii"]))
            if dset == "karisik_setler":
                face = tuple(r.choice(cov["d_math_bold"] or faces))
            im = Image.open(r.choice(photos)).convert("RGB").resize((1024, 720))
            d = ImageDraw.Draw(im)
            f = load_font(face[0], face[1], size)
            x, y = r.randint(20, 400), r.randint(20, 650)
            d.text((x, y), text, font=f, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            path = os.path.join(OUT, f"{dset}_{style}_{size}.jpg")
            im.save(path, quality=88)
            score = det.score(path)[0]
            rows.append((dset, style, size, score))

    hit = sum(s >= a.threshold for *_, s in rows)
    print(f"\n'{a.number}' testi: {hit}/{len(rows)} görselde yakalandı (%{100 * hit / len(rows):.1f})\n")
    for dset, style, size, s in sorted(rows, key=lambda t: t[3]):
        mark = "OK  " if s >= a.threshold else "KAÇTI"
        print(f"  {mark} {s:.3f}  {dset:20s} {style:8s} {size}px")


if __name__ == "__main__":
    main()
