"""Tespit (CNN) + okuma (CRNN) birleşik analiz.

1) Tespit modeli görseli 3 ölçekte 128x128 karolarla tarar -> numara olasılığı
   ve numaranın bulunduğu bölgeler.
2) Okuyucu, SADECE bu bölgelerden geçen 64 px yüksekliğindeki yatay şeritleri
   3 ölçekte, tek toplu çağrıda okur; oylamayla en güvenilir okuma seçilir.
   Numarasız görsellerde okuyucu hiç çalışmaz.
"""
import base64
import io
import os
import re
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import gpu_setup  # noqa: F401
import tensorflow as tf

from predict import tiles
from reader import LINE_H, LINE_W, build_reader, decode

HERE = os.path.dirname(os.path.abspath(__file__))
DET_PATH = os.path.join(HERE, "models", "digit_detector_cnn.h5")
READ_PATH = os.environ.get("DIGIT_READER", os.path.join(HERE, "models", "digit_reader.h5"))
READ_SCALES = (640, 1024, 384)   # uzun kenar; en sık işe yarayan ölçek önce (erken çıkış için)
EARLY_EXIT = os.environ.get("EARLY_EXIT", "1") == "1"
# TR cep: 5XX XXX XX XX (50x, 53x, 54x, 55x, 561), başında 0 / 90 olabilir.
MOBILE_RE = re.compile(r"(?:90)?0?(5(?:0[1-9]|[345]\d|61)\d{7})")
# TR sabit hat: alan kodu 2XX/3XX/4XX + 7 hane (sadece tam eşleşme).
LANDLINE_RE = re.compile(r"(?:90)?0?([2-4]\d{9})")
# Telefon formatına uymayan ama bu uzunluktaki okumalar "olası telefon" sayılır (sessizce geçmesin):
# eval_images.py'de kaçan cep numaralarının çoğu 1-2 rakamı eksik/yanlış okunmuş 9-12 haneli dizilerdi.
SUSPECT_LEN = (9, 12)
MAX_SEARCH_LEN = 14  # bundan uzun okumalarda numara "içinde aranmaz" (rastgele rakam dizisi riski)
# Girişler bu sabit boyutlara tamamlanır ve hepsi açılışta bir kez çalıştırılır: GPU her yeni
# boyutta yeniden hazırlık yapıyordu (ilk görülen boyut +400-800 ms).
DET_BATCHES = (16, 32, 64, 128)
READ_WIDTHS = (256, 384, 512, 640, 768, 1024)
READ_BATCHES = (8, 16, 32)
MAX_UPSCALE = 2.0   # küçük görseller en fazla bu kadar büyütülür (daha fazlası yazıyı 64 px şeride sığdırmaz)
STRIP_STEP = int(os.environ.get("STRIP_STEP", LINE_H // 3))   # şeritler arası dikey adım (px)
MIN_VOTES = 2    # okumanın kabulü için en az kaç pencerede aynı sonucun çıkması gerekir
MIN_DIGITS = 3


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _bucket(n, sizes):
    return next((s for s in sizes if s >= n), sizes[-1])


def _pad_batch(x, size):
    return x if len(x) == size else np.concatenate([x, np.zeros((size - len(x),) + x.shape[1:], x.dtype)])


def classify(digits):
    """Okunan rakam dizisi -> ("cep" | "sabit" | None, 10 haneli numara | None).
    Cep numarası, okuyucunun başa/sona eklediği fazladan rakamlara rağmen bulunabilsin diye
    kısa okumaların İÇİNDE de aranır (ör. '7605304836987' -> 5304836987)."""
    m = MOBILE_RE.fullmatch(digits) or (MOBILE_RE.search(digits) if len(digits) <= MAX_SEARCH_LEN else None)
    if m:
        return "cep", m.group(1)
    m = LANDLINE_RE.fullmatch(digits)
    if m:
        return "sabit", m.group(1)
    return None, None


def format_phone(core):
    return f"0{core[:3]} {core[3:6]} {core[6:8]} {core[8:]}"


class Analyzer:
    def __init__(self, threshold=0.9):
        # 0.9: tam boyutlu görsel testinde (eval_images.py) doğruluk %94.3, yakalama %94.3, yanlış alarm %5.7
        self.th = threshold
        if not os.path.exists(DET_PATH):
            raise SystemExit(f"Model bulunamadı: {DET_PATH}\nModeller repoda yok; önce README'deki "
                             "'Kurulum ve kullanım' adımlarıyla veriyi üretip modelleri eğitin.")
        self.det = tf.keras.models.load_model(DET_PATH, compile=False)
        self.reader = None
        if os.path.exists(READ_PATH):
            self.reader = build_reader(width=None)       # değişken genişlik, aynı ağırlıklar
            self.reader.load_weights(READ_PATH)
            # sabit imza -> genişlik değişse de tek kez derlenir
            self._read_fn = tf.function(lambda x: self.reader(x, training=False),
                                        input_signature=[tf.TensorSpec([None, LINE_H, None, 1], tf.float32)])
        self._det_fn = tf.function(lambda x: self.det(x, training=False),
                                   input_signature=[tf.TensorSpec([None, 128, 128, 1], tf.float32)])
        self._warmup()

    def _warmup(self):
        """İlk isteğin derleme yüzünden yavaş olmaması için modelleri bir kez çalıştır."""
        for fmt in ("JPEG", "PNG", "WEBP"):          # görsel çözücüleri de önceden yüklensin
            b = io.BytesIO()
            try:
                Image.new("RGB", (8, 8)).save(b, fmt)
                Image.open(io.BytesIO(b.getvalue())).convert("L")
            except Exception:
                pass
        for b in DET_BATCHES:
            self._det_fn(tf.zeros([b, 128, 128, 1]))
        if self.reader is not None:
            for w in READ_WIDTHS:
                for b in READ_BATCHES:
                    self._read_fn(tf.zeros([b, LINE_H, w, 1]))

    def _read(self, gray, boxes):
        """Şüpheli bölgelerden geçen yatay şeritleri (64 px yüksek, bölgelerin yatay
        kapsamı kadar geniş) ölçek ölçek okur -> uzun numaralar tek parça okunur.
        Bir ölçekte aynı telefon numarası birden fazla şeritte emin okunursa diğer
        ölçeklere geçilmez (erken çıkış)."""
        cands = []
        for side in READ_SCALES:
            cands += self._read_scale(gray, boxes, side)
            if EARLY_EXIT:
                seen = {}
                for c in cands:
                    kind, core = classify(c["digits"])
                    if c["conf"] >= 0.9 and kind:
                        seen[core] = seen.get(core, 0) + 1
                if any(n >= MIN_VOTES for n in seen.values()):
                    break
        return cands

    def _read_scale(self, gray, boxes, side):
        W, H = gray.size
        s = side / max(W, H)
        # Küçük görseller de büyütülerek okunur: aksi halde tek ölçek kalır ve 64 px'lik şerit
        # numaranın üst/alt satırlarındaki yazıları da içine alır -> fazladan uydurma rakamlar.
        if s > MAX_UPSCALE:
            return []
        strips, meta = [], []
        im = gray.resize((max(LINE_W, round(W * s)), max(LINE_H, round(H * s))), Image.BILINEAR)
        arr = np.asarray(im)
        for y in range(0, arr.shape[0] - LINE_H + 1, STRIP_STEP):
            band = (0, y / s, W, (y + LINE_H) / s)
            row = [b for b in boxes if _overlap(band, b)]
            if not row:
                continue
            x0 = max(0, int(min(b[0] for b in row) * s) - LINE_W // 4)
            x1 = min(arr.shape[1], int(max(b[2] for b in row) * s) + LINE_W // 4)
            if x1 - x0 < LINE_W:
                x0 = max(0, min(x0, arr.shape[1] - LINE_W))
                x1 = x0 + LINE_W
            strips.append(arr[y:y + LINE_H, x0:x1])
            meta.append((x0 / s, y / s, x1 / s, (y + LINE_H) / s))
        if not strips:
            return []
        # Bu ölçeğin şeritleri TEK seferde: genişlik bu ölçekteki en geniş şeride göre (sabit
        # boyutlardan birine) tamamlanır; kenar pikseli tekrarlanır (sahte çizgi oluşmaz).
        # Süre toplam piksel sayısıyla orantılı: küçük ölçeğin şeritlerini 1024'e tamamlamamak önemli.
        width = _bucket(max(st.shape[1] for st in strips), READ_WIDTHS)
        strips = [st[:, :width] for st in strips]
        batch = np.stack([np.pad(st, ((0, 0), (0, width - st.shape[1])), mode="edge") for st in strips])
        valid = [int(np.ceil(st.shape[1] / 4)) for st in strips]   # dolgu kısmının zaman adımlarını atla
        probs = []
        for i in range(0, len(batch), READ_BATCHES[-1]):
            chunk = batch[i:i + READ_BATCHES[-1], ..., None].astype(np.float32)
            out = self._read_fn(tf.constant(_pad_batch(chunk, _bucket(len(chunk), READ_BATCHES)))).numpy()
            probs.append(out[:len(chunk)])
        probs = np.concatenate(probs)
        cands = []
        for pr, v, box in zip(probs, valid, meta):
            text, conf, _ = decode(pr[None, :v])[0]
            if text:
                cands.append({"digits": text, "conf": conf, "box": box})
        return cands

    def analyze(self, image_bytes, detect_th=None):
        tm, t0 = {}, time.perf_counter()

        def lap(name):
            nonlocal t0
            now = time.perf_counter()
            tm[name] = round((now - t0) * 1000, 1)
            t0 = now

        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        gray = img.convert("L")
        lap("decode")
        t = list(tiles(gray))
        x = np.stack([a for a, _ in t])[..., None].astype(np.float32)
        lap("tiles")
        p = []
        for i in range(0, len(x), DET_BATCHES[-1]):
            chunk = x[i:i + DET_BATCHES[-1]]
            p.append(self._det_fn(tf.constant(_pad_batch(chunk, _bucket(len(chunk), DET_BATCHES)))).numpy()[:len(chunk), 0])
        p = np.concatenate(p)
        lap("detect")
        boxes = [b for _, b in t]
        score = float(p.max())
        region_th = min(0.5, self.th) if detect_th is None else detect_th  # okuma bölgeleri daha geniş tutulur
        hits = [(float(pi), b) for pi, b in zip(p, boxes) if pi >= region_th]

        best, readings = None, []
        # Tespit eşiği geçilmediyse uyarı zaten verilmeyecek: okuyucuyu hiç çalıştırma.
        if hits and self.reader is not None and score >= self.th:
            # görselin tamamını kaplayan (en kaba ölçek) karo okuma bölgesi olarak işe yaramaz
            area = img.width * img.height
            small = [b for _, b in hits if (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * area]
            strong = [b for pi, b in hits if pi >= self.th and (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * area]
            cands = self._read(gray, strong or small or [b for _, b in hits])   # sadece emin olunan bölgeler okunur
            # Oylama: aynı rakam dizisi birçok pencerede (farklı ölçek/konum) okunuyorsa güvenilirdir.
            # Telefon olarak tanınan okumalar öz numaraya göre birleşir: '05304836987' ile
            # başına fazladan rakam eklenmiş '7605304836987' aynı numaraya oy verir.
            votes = {}
            for c in cands:
                if c["conf"] < 0.6:
                    continue
                kind, core = classify(c["digits"])
                key = (kind, core) if kind else (None, c["digits"])
                v = votes.setdefault(key, {"digits": core or c["digits"], "kind": kind, "conf": 0.0,
                                           "votes": 0, "box": c["box"]})
                v["votes"] += 1
                if c["conf"] > v["conf"]:
                    v["conf"], v["box"] = c["conf"], c["box"]
            rank = {"cep": 2, "sabit": 1, None: 0}
            readings = sorted(votes.values(), key=lambda v: (
                rank[v["kind"]], v["votes"] >= MIN_VOTES,
                min(len(v["digits"]), 12) * v["conf"] * min(v["votes"], 5)), reverse=True)
            best = readings[0] if readings else None
            if best and (len(best["digits"]) < MIN_DIGITS or
                         (best["votes"] < MIN_VOTES and not (len(best["digits"]) >= 6 and best["conf"] >= 0.85))):
                best = None   # tek pencerede görülen kısa / emin olunmayan okuma: büyük olasılıkla uydurma

        lap("read")
        # has_number: görselde rakam var mı (tespit modeli). phone_type: okunan numara cep / sabit
        # telefon mu? Uyarı sadece telefon için verilir; ilan kodu (2501-2159), fiyat, m² uyarı değildir.
        has_number = score >= self.th
        phone_type = best["kind"] if best else None
        is_phone = phone_type == "cep"
        if phone_type:
            number = format_phone(best["digits"])
        else:
            number = best["digits"] if best else None
        if not has_number:
            status = "yok"
        elif phone_type == "cep":
            status = "cep"
        elif phone_type == "sabit":
            status = "sabit"
        elif best and SUSPECT_LEN[0] <= len(best["digits"]) <= SUSPECT_LEN[1]:
            status = "olasi_telefon"      # telefon uzunluğunda ama formatı tutmuyor: okuyucu 1-2 rakamı kaçırmış olabilir
        elif best:
            status = "telefon_degil"      # okundu, telefon değil (ilan kodu, fiyat, m² ...)
        else:
            status = "okunamadi"          # rakam var ama okunamadı: elle kontrol edilmeli

        # işaretli görsel: önce küçült, sonra çiz (tam boyutta kopyala/çiz/küçült ~2x yavaş)
        vis = img.copy()
        vis.thumbnail((960, 960), Image.BILINEAR)
        k = vis.width / img.width
        d = ImageDraw.Draw(vis)
        lw = max(2, vis.width // 250)
        for pi, b in hits:
            d.rectangle([v * k for v in b], outline=(255, 170, 0), width=max(1, lw // 2))
        if best:
            d.rectangle([v * k for v in best["box"]], outline=(230, 30, 30), width=lw * 2)
        buf = io.BytesIO()
        vis.save(buf, "JPEG", quality=80)
        lap("render")

        return {
            "has_number": has_number,
            "score": score,
            "is_phone": is_phone,
            "phone_type": phone_type,
            "status": status,
            "number": number,
            "number_raw": best["digits"] if best else None,
            "read_conf": best["conf"] if best else None,
            "votes": best["votes"] if best else 0,
            "others": [c["digits"] for c in readings if c is not best and c["votes"] >= MIN_VOTES][:4],
            "n_regions": len(hits),
            "timings_ms": tm,
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        }
