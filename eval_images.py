"""GÖRSEL DÜZEYİNDE test: tam boyutlu fotoğraflarda uçtan uca analiz.

Eğitimde / madencilikte kullanılmayan fotoğraflar (her 5. fotoğraf) üzerinden
her biri için 1 numaralı + 1 numarasız görsel üretilir ve Analyzer ile test edilir.

    python eval_images.py --n 734
"""
import argparse
import glob
import io
import json
import os
import random
import time

import numpy as np
from PIL import Image, ImageDraw

from analyzer import Analyzer, classify
from fonts import build_coverage, load_font
from mine_negatives import words_overlay
from synth import Synth, charset_table

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def number_overlay(im, S, r):
    """Fotoğrafa rastgele Unicode setinde numara (+bazen kelime) yazar; doğru rakamları döner."""
    if r.random() < 0.5:
        words_overlay(im, S, r)
    dset = "ascii" if r.random() < 0.2 else r.choice(S.digit_sets)
    pieces, _ = S.number_pieces(dset)
    size = r.randint(20, 72)
    width = sum(load_font(f[0], f[1], size).getlength(ch) for ch, f, *_ in pieces)
    x = r.randint(10, max(11, int(im.width - width - 10)))
    y = r.randint(10, im.height - size - 10)
    d = ImageDraw.Draw(im)
    color = r.choice([0, 255])
    for ch, face, *_ in pieces:
        f = load_font(face[0], face[1], size)
        d.text((x, y), ch, font=f, fill=color, stroke_width=r.choice([0, 2]), stroke_fill=255 - color)
        x += f.getlength(ch)
    return "".join(p[2] for p in pieces if len(p) > 2), dset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=734, help="kaç fotoğraf (her biri 1 pozitif + 1 negatif)")
    a = ap.parse_args()
    r = random.Random(2024)
    S = Synth(build_coverage(charset_table()), seed=2024)
    photos = [p for i, p in enumerate(sorted(glob.glob(os.path.join(DATA, "flower_photos", "*", "*.jpg"))))
              if i % 5 == 0][:a.n]
    A = Analyzer(threshold=0.9)          # karar eşiğini aşağıda skor üzerinden kendimiz tarıyoruz
    rows, times = [], []
    for k, p in enumerate(photos):
        for label in (1, 0):
            im = Image.open(p).convert("RGB")
            s = r.uniform(1.5, 3.0)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
            truth, dset = "", "negatif"
            if label:
                truth, dset = number_overlay(im, S, r)
            elif r.random() < 0.7:
                words_overlay(im, S, r)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=r.randint(60, 95))
            t_ = time.perf_counter()
            res = A.analyze(buf.getvalue(), detect_th=0.5)
            times.append((time.perf_counter() - t_) * 1000)
            rows.append({"label": label, "set": dset, "truth": truth, "score": res["score"], "status": res["status"],
                         "read": res["number_raw"] or "", "read_conf": res["read_conf"] or 0.0})
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(photos)}", flush=True)

    y = np.array([q["label"] for q in rows])
    sc = np.array([q["score"] for q in rows])
    has_read = np.array([len(q["read"]) >= 3 for q in rows])   # okuyucu en az 3 rakam okudu mu

    def report(pred, name):
        acc = (pred == y).mean()
        tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
        fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        print(f"  {name:38s} doğruluk={acc:.4f}  yakalama={tp / max(1, tp + fn):.4f}  "
              f"yanlış alarm={fp / max(1, fp + tn):.4f}   [TP={tp} FN={fn} FP={fp} TN={tn}]")
        return acc

    print(f"\n=== GÖRSEL DÜZEYİNDE TEST: {len(y)} görsel ({y.sum()} numaralı, {(1 - y).sum()} numarasız) ===")
    best = (0, 0.5)
    for th in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
        acc = report((sc >= th).astype(int), f"sadece tespit, eşik {th}")
        best = max(best, (acc, th))
    print()
    for th in (0.5, 0.8, 0.9, 0.95):
        report(((sc >= th) & has_read).astype(int), f"tespit {th} + okuyucu doğrulaması")

    pos = [q for q in rows if q["label"] == 1]
    for q in pos:   # telefonlar öz numara (10 hane) olarak döner: gerçeği de aynı biçime getir
        kind, core = classify(q["truth"])
        if kind and q["read"] == core:
            q["read"] = q["truth"]
    exact = np.mean([q["read"] == q["truth"] for q in pos])
    phone_pos = [q for q in pos if classify(q["truth"])[0] == "cep"]
    other_pos = [q for q in pos if classify(q["truth"])[0] is None]
    print(f"\nCep numarası içeren görsellerde 'cep' dendi: {np.mean([q['status'] == 'cep' for q in phone_pos]):.4f} "
          f"(n={len(phone_pos)})")
    print(f"Telefon olmayan numaralarda yanlışlıkla 'cep' dendi: {np.mean([q['status'] == 'cep' for q in other_pos]):.4f} "
          f"(n={len(other_pos)})")
    WARN = ("cep", "sabit", "olasi_telefon", "okunamadi")   # arayüzde kırmızı/turuncu uyarı alanlar
    print(f"Cep numaralı görsellerde HERHANGİ bir uyarı verildi: {np.mean([q['status'] in WARN for q in phone_pos]):.4f}")
    print(f"Telefon olmayan numaralarda uyarı verildi: {np.mean([q['status'] in WARN for q in other_pos]):.4f}")
    neg = [q for q in rows if q["label"] == 0]
    print(f"Numarasız görsellerde uyarı verildi: {np.mean([q['status'] in WARN for q in neg]):.4f}")
    print(f"Numarasız görsellerde yanlışlıkla 'cep' dendi: {np.mean([q['status'] == 'cep' for q in neg]):.4f} (n={len(neg)})")
    contains = np.mean([q["truth"] in q["read"] or q["read"] in q["truth"] and len(q["read"]) >= 7 for q in pos])
    print(f"\nAnaliz süresi (süre_ms): ortalama={np.mean(times):.0f}  medyan={np.median(times):.0f}  p95={np.percentile(times, 95):.0f}  en yavaş={np.max(times):.0f}")
    print(f"\nNumara okuma (numaralı görseller): tam doğru={exact:.4f}  büyük ölçüde doğru={contains:.4f}")
    print("Örnekler:")
    for q in pos[:12]:
        print(f"  {q['set']:20s} gerçek={q['truth']:14s} okunan={q['read']:14s} {'OK' if q['read'] == q['truth'] else ''}")
    with open(os.path.join(HERE, "models", "metrics_images.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
