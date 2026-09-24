"""Download real-photo backgrounds + MNIST, then render the synthetic dataset.

    python prepare_data.py --train 80000 --val 8000
"""
import argparse
import glob
import os
import tarfile
import time
import urllib.request
from multiprocessing import Pool

import numpy as np
from PIL import Image

from fonts import build_coverage
from synth import Synth, charset_table, IMG

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# TensorFlow's public example dataset: 3,670 real-world photos (~218 MB).
FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
BG_SIDE = 256
CUSTOM_BG = os.path.join(DATA, "custom_backgrounds")   # kendi arka plan fotoğraflarınız (numarasız)


def download(url, dst):
    if os.path.exists(dst):
        return dst
    print(f"indiriliyor: {url}")
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)
    return dst


def list_images(folder):
    """Klasördeki (alt klasörler dahil) tüm görseller."""
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    return sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                  if p.lower().endswith(exts))


def prepare_backgrounds(use_flowers=True):
    """Arka plan havuzu = TF flower_photos (indirilir) + data/custom_backgrounds/ içindeki
    KENDİ fotoğraflarınız (ör. gerçek ilan fotoğrafları; üzerlerinde numara OLMAMALI).
    Klasöre yeni fotoğraf eklenince backgrounds.npy otomatik yeniden oluşturulur."""
    out = os.path.join(DATA, "backgrounds.npy")
    custom = list_images(CUSTOM_BG)
    newest = max([os.path.getmtime(p) for p in custom], default=0)
    if os.path.exists(out) and os.path.getmtime(out) >= newest:
        return out
    paths = list(custom)
    if use_flowers:
        tgz = download(FLOWERS_URL, os.path.join(DATA, "flower_photos.tgz"))
        root = os.path.join(DATA, "flower_photos")
        if not os.path.isdir(root):
            with tarfile.open(tgz) as t:
                t.extractall(DATA)
        paths += sorted(glob.glob(os.path.join(root, "*", "*.jpg")))
    if not paths:
        raise SystemExit(f"arka plan yok: {CUSTOM_BG} klasörüne fotoğraf koyun ya da --no_flowers kullanmayın")
    arrs = []
    for p in paths:
        try:
            im = Image.open(p).convert("L")
        except Exception as e:
            print(f"  atlandı ({e}): {p}")
            continue
        s = BG_SIDE / min(im.size)
        im = im.resize((max(BG_SIDE, round(im.width * s)), max(BG_SIDE, round(im.height * s))), Image.BILINEAR)
        l, t = (im.width - BG_SIDE) // 2, (im.height - BG_SIDE) // 2
        arrs.append(np.asarray(im.crop((l, t, l + BG_SIDE, t + BG_SIDE))))
    np.save(out, np.stack(arrs))
    print(f"{len(arrs)} arka plan fotoğrafı ({len(custom)} tanesi {CUSTOM_BG} klasöründen) -> {out}")
    return out


def prepare_mnist():
    out = os.path.join(DATA, "mnist_digits.npy")
    if not os.path.exists(out):
        with np.load(download(MNIST_URL, os.path.join(DATA, "mnist.npz"))) as d:
            np.save(out, np.concatenate([d["x_train"], d["x_test"]]))
    return out


_SYNTH = None


def _init_worker(coverage, bg_path, mnist_path, seed):
    global _SYNTH
    bgs = np.load(bg_path, mmap_mode="r") if bg_path else None
    mn = np.load(mnist_path, mmap_mode="r") if mnist_path else None
    _SYNTH = Synth(coverage, bgs, mn, seed=seed + os.getpid())


def _make(args):
    label, force = args
    img, meta = _SYNTH.sample(label, force_set=force)
    return img, label, meta.get("set", "")


def generate(n, coverage, bg, mn, seed, pool_size, forced=None):
    labels = [i % 2 for i in range(n)] if forced is None else [1] * n
    jobs = [(lab, forced) for lab in labels]
    with Pool(pool_size, _init_worker, (coverage, bg, mn, seed)) as pool:
        res = pool.map(_make, jobs, chunksize=256)
    x = np.stack([r[0] for r in res])[..., None]
    y = np.array([r[1] for r in res], np.uint8)
    s = np.array([r[2] for r in res])
    return x, y, s


def _make_line(_):
    return _SYNTH.sample_line()


def generate_lines(n, coverage, bg, pool_size, chunk=30000):
    """Okuyucu (OCR) verisi: lines_x.npy (N,64,256,1) + etiketler (rakam dizisi)."""
    from synth import LINE_W, LINE_H, MAX_LABEL
    X = np.lib.format.open_memmap(os.path.join(DATA, "lines_x.npy"), "w+", np.uint8, (n, LINE_H, LINE_W, 1))
    L = np.full((n, MAX_LABEL), -1, np.int8)
    t = time.time()
    for i, start in enumerate(range(0, n, chunk)):
        m = min(chunk, n - start)
        with Pool(pool_size, _init_worker, (coverage, bg, None, 80_000_000 + i * 1_000_003)) as pool:
            res = pool.map(_make_line, range(m), chunksize=256)
        for j, (img, lab) in enumerate(res):
            X[start + j, ..., 0] = img
            L[start + j, :len(lab)] = [int(c) for c in lab]
        print(f"  lines: {start + m}/{n}  ({time.time() - t:.0f}s)", flush=True)
    X.flush()
    np.save(os.path.join(DATA, "lines_y.npy"), L)


def generate_big(n, coverage, bg, mn, pool_size, chunk=40000):
    """n görseli RAM'e sığdırmadan diske (memmap .npy) chunk chunk yazar."""
    X = np.lib.format.open_memmap(os.path.join(DATA, "big_x.npy"), "w+", np.uint8, (n, IMG, IMG, 1))
    Y, S = np.zeros(n, np.uint8), np.empty(n, dtype="U24")
    t = time.time()
    for i, start in enumerate(range(0, n, chunk)):
        m = min(chunk, n - start)
        x, y, s = generate(m, coverage, bg, mn, 50_000_000 + i * 1_000_003, pool_size)
        X[start:start + m], Y[start:start + m], S[start:start + m] = x, y, s
        print(f"  big: {start + m}/{n}  ({time.time() - t:.0f}s)", flush=True)
    X.flush()
    np.save(os.path.join(DATA, "big_y.npy"), Y)
    np.save(os.path.join(DATA, "big_s.npy"), S)


def save_preview(x, y, path, n=64):
    cols = 8
    grid = Image.new("L", (cols * (IMG + 4), (n // cols) * (IMG + 18)), 255)
    from PIL import ImageDraw
    d = ImageDraw.Draw(grid)
    for i in range(n):
        cx, cy = (i % cols) * (IMG + 4), (i // cols) * (IMG + 18)
        grid.paste(Image.fromarray(x[i, ..., 0]), (cx, cy))
        d.text((cx + 2, cy + IMG + 2), "NUMARA VAR" if y[i] else "yok", fill=0)
    grid.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=80000)
    ap.add_argument("--val", type=int, default=8000)
    ap.add_argument("--per_set_test", type=int, default=300, help="positives per digit set for the per-set report")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--no_flowers", action="store_true",
                    help="flower_photos indirme; sadece data/custom_backgrounds/ kullanılsın")
    ap.add_argument("--big", type=int, default=0, help="N>0: sadece diske N görsellik büyük set üret (big_*.npy)")
    ap.add_argument("--lines", type=int, default=0, help="N>0: sadece okuyucu (OCR) için N satır üret")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    bg = mn = None
    if not a.no_download:
        bg, mn = prepare_backgrounds(use_flowers=not a.no_flowers), prepare_mnist()
    coverage = build_coverage(charset_table())
    missing = [k for k, v in coverage.items() if not v]
    if missing:
        print("font bulunamayan setler (atlanacak):", missing)
    t = time.time()
    if a.lines:
        generate_lines(a.lines, coverage, bg, a.workers)
        return
    if a.big:
        generate_big(a.big, coverage, bg, mn, a.workers)
        x = np.load(os.path.join(DATA, "big_x.npy"), mmap_mode="r")[:64]
        save_preview(np.asarray(x), np.load(os.path.join(DATA, "big_y.npy"))[:64], os.path.join(DATA, "preview.png"))
    else:
        for split, n, seed in [("train", a.train, 1), ("val", a.val, 1_000_003)]:
            x, y, s = generate(n, coverage, bg, mn, seed, a.workers)
            np.savez(os.path.join(DATA, f"{split}.npz"), x=x, y=y, s=s)
            print(f"{split}: {x.shape}  pozitif oranı={y.mean():.2f}  ({time.time() - t:.0f}s)")
        save_preview(x, y, os.path.join(DATA, "preview.png"))

    # per-digit-set positives + matching count of negatives for the report
    sets = [k[2:] for k, v in coverage.items() if k.startswith("d_") and v]
    xs, ys, ss = [], [], []
    for i, name in enumerate(sets):
        x, y, s = generate(a.per_set_test, coverage, bg, mn, 7_000_000 + i, a.workers, forced=name)
        xs.append(x); ys.append(y); ss.append(s)
    x, y, s = generate(a.per_set_test * 3, coverage, bg, mn, 9_000_000, a.workers)
    neg = y == 0
    xs.append(x[neg]); ys.append(y[neg]); ss.append(np.array(["negatif"] * neg.sum()))
    np.savez(os.path.join(DATA, "test_sets.npz"), x=np.concatenate(xs), y=np.concatenate(ys), s=np.concatenate(ss))
    print(f"bitti ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
