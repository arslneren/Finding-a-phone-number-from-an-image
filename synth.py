"""Synthetic 128x128 grayscale patch generator.

label 1 -> patch contains a number (any Unicode digit style, lookalike-mixed,
           handwritten MNIST digits, keycap/boxed digits ...)
label 0 -> patch contains no digits (plain background, words in fancy Unicode
           letters, phone icons, symbols, scribbles, window-grid patterns ...)
"""
import io
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from unicode_sets import DIGIT_SETS, LOOKALIKES, LETTER_SETS, SYMBOLS, TR_WORDS, EN_WORDS
from fonts import load_font, has_glyph, MISSING_PROBE, PROBE_SIZE

IMG = 128
ASCII_LETTERS = LETTER_SETS["ascii"][:52]
TR_TO_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
LINE_W, LINE_H, MAX_LABEL = 256, 64, 20   # okuyucu girişi ve en uzun etiket
MIN_TEXT = 12   # px: bundan küçük yazı 128'lik karoda okunamaz -> etiket gürültüsü olur
ALL_DIGITS = {c for t in DIGIT_SETS.values() for c in t if c}
DIGIT_VALUE = {c: str(i) for t in DIGIT_SETS.values() for i, c in enumerate(t) if c}
_MISSING = {}


def _missing_sig(face):
    if face not in _MISSING:
        m = load_font(face[0], face[1], PROBE_SIZE).getmask(MISSING_PROBE)
        _MISSING[face] = (m.size, bytes(m))
    return _MISSING[face]


def charset_table():
    """Everything whose font coverage we need (keys used by build_coverage)."""
    t = {"d_" + k: v for k, v in DIGIT_SETS.items()}
    t.update({"l_" + k: v for k, v in LETTER_SETS.items()})
    t["symbols"] = ["☎", "★", "✓", "→", "•"]  # core; _safe() handles the rest
    return t


class Synth:
    def __init__(self, coverage, backgrounds=None, mnist=None, seed=None):
        self.r = random.Random(seed)
        self.np_r = np.random.default_rng(seed)
        self.cov = {k: [tuple(f) for f in v] for k, v in coverage.items() if v}
        self.digit_sets = [k[2:] for k in self.cov if k.startswith("d_")]
        self.letter_sets = [k[2:] for k in self.cov if k.startswith("l_")]
        self.backgrounds = backgrounds      # (N, H, W) uint8, may be memmap
        self.mnist = mnist                  # (N, 28, 28) uint8
        self._glyph_cache = {}
        self.line_mode = False

    # ------------------------------------------------------------ backgrounds
    def background(self, W=IMG, H=IMG):
        r = self.r
        p = r.random()
        if self.backgrounds is not None and p < 0.6:
            bg = self.backgrounds[r.randrange(len(self.backgrounds))]
            h, w = bg.shape
            sw = r.randint(min(w, max(W // 2, 32)), w)          # en-boy oranı W:H korunur
            sh = max(1, min(h, sw * H // W))
            y, x = r.randint(0, h - sh), r.randint(0, w - sw)
            img = Image.fromarray(np.asarray(bg[y:y + sh, x:x + sw])).resize((W, H), Image.BILINEAR)
            if r.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif p < 0.8:
            a, b = r.randint(0, 255), r.randint(0, 255)
            if r.random() < 0.5:
                arr = np.tile(np.linspace(a, b, W, dtype=np.float32), (H, 1))
            else:
                arr = np.tile(np.linspace(a, b, H, dtype=np.float32)[:, None], (1, W))
            img = Image.fromarray(arr.astype(np.uint8))
        else:
            img = Image.new("L", (W, H), r.randint(0, 255))
        if r.random() < 0.35:
            self._draw_clutter(img)
        return img

    def _draw_clutter(self, img):
        """Lines, grids, circles: shapes that resemble 1/0 but are not digits."""
        r, d = self.r, ImageDraw.Draw(img)
        for _ in range(r.randint(1, 8)):
            c, w = r.randint(0, 255), r.randint(1, 4)
            k = r.random()
            W, H = img.size
            x0, y0 = r.randint(-20, W), r.randint(-20, H)
            x1, y1 = x0 + r.randint(4, 80), y0 + r.randint(4, 80)
            if k < 0.35:
                d.line([(x0, y0), (r.randint(0, W), r.randint(0, H))], fill=c, width=w)
            elif k < 0.6:
                d.rectangle([x0, y0, x1, y1], outline=c, width=w)
            elif k < 0.8:
                d.ellipse([x0, y0, x1, y1], outline=c, width=w)
            else:  # window grid
                step = r.randint(6, 20)
                for gx in range(x0, x1, step):
                    d.line([(gx, y0), (gx, y1)], fill=c, width=1)
                for gy in range(y0, y1, step):
                    d.line([(x0, gy), (x1, gy)], fill=c, width=1)

    # --------------------------------------------------------------- strings
    def _face(self, key):
        return self.r.choice(self.cov[key])

    def _ok(self, ch, face):
        key = (ch, face)
        if key not in self._glyph_cache:
            f = load_font(face[0], face[1], PROBE_SIZE)
            self._glyph_cache[key] = ch.isspace() or has_glyph(f, ch, _missing_sig(face))
        return self._glyph_cache[key]

    def _safe(self, pieces):
        """Swap faces for chars the chosen face cannot render; drop the char if
        no fallback works (never let a .notdef box leak into the data)."""
        out = []
        for p in pieces:
            ch, face = p[0], p[1]
            if not self._ok(ch, face):
                for _ in range(4):
                    face = self._face("d_ascii")
                    if self._ok(ch, face):
                        break
                else:
                    continue
            out.append((ch, face) + tuple(p[2:]))
        return out

    def _digit_char(self, dset, d):
        ch = DIGIT_SETS[dset][d]
        return ch if ch is not None else DIGIT_SETS["ascii"][d]

    def number_pieces(self, dset, allow_prefix=True):
        """List of (char, face, value) for one number; value = the digit ('0'-'9')
        the glyph stands for (lookalikes included) or '' for separators/letters.
        Returns pieces, n_real_digits."""
        r = self.r
        face = self._face("d_" + dset)
        kind = r.random()
        if kind < 0.45:          # Turkish mobile / landline
            digits = "0" * (r.random() < 0.6) + r.choice("25345") + "".join(r.choice("0123456789") for _ in range(9))
            groups = [digits[:-7], digits[-7:-4], digits[-4:-2], digits[-2:]]
        elif kind < 0.7:         # price / area / generic short number
            n = r.randint(1, 8)
            digits = "".join(r.choice("0123456789") for _ in range(n))
            groups = [digits] if r.random() < 0.5 else [digits[max(0, i - 3):i] for i in range(len(digits), 0, -3)][::-1]
        else:                    # random grouping
            n = r.randint(2, 12)
            digits = "".join(r.choice("0123456789") for _ in range(n))
            groups, i = [], 0
            while i < n:
                k = r.randint(1, 4)
                groups.append(digits[i:i + k])
                i += k

        sep = r.choice([" ", " ", "", "-", ".", "/", " ", "  ", "•", "*", "_", ","])
        mix_p = r.choice([0.0, 0.0, 0.15, 0.4])
        look_p = r.choice([0.0, 0.0, 0.1, 0.25])
        pieces, real = [], 0
        if allow_prefix and r.random() < 0.25:
            for ch in r.choice(["Tel: ", "TEL ", "+90 ", "Ara ", "WhatsApp ", "GSM: ", "(", "No:"]):
                pieces.append((ch, face, DIGIT_VALUE.get(ch, "")))
        for gi, g in enumerate(groups):
            for ch in g:
                d = int(ch)
                if real > 0 and r.random() < look_p:
                    pieces.append((r.choice(LOOKALIKES[d]), face, ch))
                    continue
                s, f = dset, face
                if r.random() < mix_p:
                    s = r.choice(self.digit_sets)
                    f = self._face("d_" + s)
                pieces.append((self._digit_char(s, d), f, ch))
                real += 1
            if gi < len(groups) - 1:
                for ch in sep:
                    pieces.append((ch, face, ""))
        if r.random() < 0.15:
            for ch in r.choice([" TL", " m²", " ₺", "+1", " $", " €"]):
                pieces.append((ch, face, DIGIT_VALUE.get(ch, "")))
        return self._safe(pieces), real

    def word_pieces(self):
        """Digit-free text: Turkish/English words, fancy Unicode letters, symbols."""
        r = self.r
        if r.random() < 0.15 and "symbols" in self.cov:
            face = self._face("symbols")
            return self._safe([(r.choice(SYMBOLS), face) for _ in range(r.randint(1, 4))])
        words = " ".join(r.choice(TR_WORDS + EN_WORDS) for _ in range(r.randint(1, 3)))
        case = r.random()
        words = words.upper() if case < 0.3 else words.title() if case < 0.5 else words
        lset = "ascii" if r.random() < 0.55 else r.choice(self.letter_sets)
        face = self._face("l_" + lset)
        table = LETTER_SETS[lset]
        out = []
        if lset != "ascii":
            words = words.translate(TR_TO_ASCII)
        for ch in words:
            if lset in ("greek", "cyrillic") and ch in ASCII_LETTERS:
                ch = r.choice(table)
            elif lset != "ascii" and ch in ASCII_LETTERS:
                ch = table[ASCII_LETTERS.index(ch) % len(table)]
            out.append((ch, face))
        if r.random() < 0.15 and "symbols" in self.cov:
            out.insert(0, (r.choice(SYMBOLS[:10]), self._face("symbols")))
            out.insert(1, (" ", face))
        return self._safe(out)

    # ------------------------------------------------------------- rendering
    def render_pieces(self, pieces, size, stroke=0, keycap=False):
        """Render (char, face) pieces on a baseline -> (fill_mask, stroke_mask|None)."""
        spacing = self.r.choice([0, 0, 0, 1, 2, size // 6])
        pieces = [p[:2] for p in pieces]
        fonts = [load_font(f[0], f[1], size) for _, f in pieces]
        widths = [max(1, f.getlength(ch)) + spacing + (size // 3 if keycap else 0)
                  for (ch, _), f in zip(pieces, fonts)]
        W = int(sum(widths) + size * 2 + stroke * 2)
        H = int(size * 2.4 + stroke * 2)
        fill = Image.new("L", (W, H), 0)
        strk = Image.new("L", (W, H), 0) if stroke else None
        df, ds = ImageDraw.Draw(fill), ImageDraw.Draw(strk) if stroke else None
        x, base = size, int(size * 1.6)
        for (ch, _), f, w in zip(pieces, fonts, widths):
            if keycap and not ch.isspace():
                bw = int(w - spacing)
                df.rounded_rectangle([x - 2, base - size * 1.05, x + bw - 2, base + size * 0.3],
                                     radius=max(2, size // 5), outline=255, width=max(1, size // 14))
                df.text((x + size // 6, base), ch, font=f, fill=255, anchor="ls")
            else:
                df.text((x, base), ch, font=f, fill=255, anchor="ls")
            if stroke:
                ds.text((x, base), ch, font=f, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
            x += w
        return fill, strk

    def mnist_mask(self, size):
        r = self.r
        n = r.randint(1, max(1, min(11, int((IMG - 8) / size))))  # küçültmeden sığsın
        digits = []
        for _ in range(n):
            d = Image.fromarray(np.asarray(self.mnist[r.randrange(len(self.mnist))]))
            bb = d.getbbox() or (0, 0, 28, 28)
            d = d.crop((max(0, bb[0] - 1), 0, min(28, bb[2] + 1), 28))
            digits.append(d)
        gap = r.randint(0, 4)
        W = sum(d.width for d in digits) + gap * n + 4
        m = Image.new("L", (W, 28), 0)
        x = 2
        for d in digits:
            m.paste(d, (x, 0))
            x += d.width + gap
        scale = size / 20.0
        return m.resize((max(1, int(W * scale)), max(1, int(28 * scale))), Image.BILINEAR)

    def scribble_mask(self, size):
        """Pen-like curves with MNIST-ish stroke width (hard negative)."""
        r = self.r
        W, H = int(size * r.uniform(1.5, 6)), int(size * 1.3)
        m = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(m)
        pts = [(r.uniform(0, W), r.uniform(0, H)) for _ in range(r.randint(3, 10))]
        d.line(pts, fill=255, width=max(1, size // 7), joint="curve")
        return m.filter(ImageFilter.GaussianBlur(0.6))

    def _transform(self, masks):
        r = self.r
        if self.line_mode:  # okuyucu satırları: hafif eğim, dikey yazı yok
            angle = max(-6, min(6, r.gauss(0, 2.5)))
        else:
            angle = 90 if r.random() < 0.03 else max(-25, min(25, r.gauss(0, 7)))
        out = []
        shear = r.gauss(0, 0.12)
        # dolgu ve kontur maskesi AYNI kutuyla kırpılsın ki üst üste otursunlar
        bbs = [m.getbbox() for m in masks if m is not None and m.getbbox()]
        bb = (min(b[0] for b in bbs), min(b[1] for b in bbs),
              max(b[2] for b in bbs), max(b[3] for b in bbs)) if bbs else None
        for m in masks:
            if m is None:
                out.append(None)
                continue
            if bb:
                m = m.crop(bb)
            if abs(shear) > 0.02:
                w, h = m.size
                extra = int(abs(shear) * h)
                m = m.transform((w + extra, h), Image.AFFINE,
                                (1, shear, -extra if shear > 0 else 0, 0, 1, 0), Image.BILINEAR)
            out.append(m.rotate(angle, resample=Image.BILINEAR, expand=True))
        return out

    def _fit(self, masks, max_w, max_h):
        w, h = masks[0].size
        s = min(1.0, max_w / max(w, 1), max_h / max(h, 1))
        if s < 1.0:
            masks = [m.resize((max(1, int(m.width * s)), max(1, int(m.height * s))), Image.BILINEAR)
                     if m is not None else None for m in masks]
        return masks

    def composite(self, img, fill, strk, fully_inside, place=None):
        r = self.r
        w, h = fill.size
        CW, CH = img.size
        if fully_inside:
            x = r.randint(0, max(0, CW - w))
            y = r.randint(0, max(0, CH - h))
        else:
            x = r.randint(-w // 2, CW - w // 2)
            y = r.randint(-h // 2, CH - h // 2)
        if place is not None:
            x, y = place
        bg = np.asarray(img, dtype=np.float32)
        # Kontrast, bölge ortalamasına değil YAZININ TAM ALTINDAKİ piksellere göre.
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(CH, y + h), min(CW, x + w)
        m = 128.0
        if y1 > y0 and x1 > x0:
            under = np.asarray(fill, dtype=np.uint8)[y0 - y:y1 - y, x0 - x:x1 - x] > 128
            region = bg[y0:y1, x0:x1]
            if under.any():
                m = float(region[under].mean())
            elif region.size:
                m = float(region.mean())

        def contrasting():
            c = r.choice([0.0, 255.0, r.uniform(0, 255)])
            need = r.choice([60, 90, 130])
            if abs(c - m) < need:
                c = min(255.0, m + need) if m < 128 else max(0.0, m - need)
            return c

        canvas = Image.fromarray(bg.astype(np.uint8))
        if r.random() < 0.2:  # sticker / label box behind text
            pad = r.randint(2, 8)
            box_c = r.randint(0, 255)
            d = ImageDraw.Draw(canvas)
            d.rounded_rectangle([x - pad, y - pad, x + w + pad, y + h + pad], radius=r.randint(0, 8), fill=box_c)
            m = box_c
        if r.random() < 0.15:  # drop shadow
            sh = fill.filter(ImageFilter.GaussianBlur(r.uniform(0.5, 2)))
            off = r.randint(1, 4)
            canvas.paste(0 if m > 100 else 255, (x + off, y + off),
                         sh.point(lambda v: int(v * 0.6)))
        if strk is not None:
            canvas.paste(int(contrasting()), (x, y), strk)
        alpha = r.uniform(0.85, 1.0)
        canvas.paste(int(contrasting()), (x, y), fill.point(lambda v: int(v * alpha)))
        return canvas, (x, y, w, h)

    def degrade(self, img):
        r = self.r
        if r.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.3)))
        if r.random() < 0.25:
            s = r.uniform(0.6, 0.85)
            W0, H0 = img.size
            img = img.resize((int(W0 * s), int(H0 * s)), Image.BILINEAR).resize((W0, H0), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if r.random() < 0.3:
            arr = arr + self.np_r.normal(0, r.uniform(2, 14), arr.shape)
        if r.random() < 0.3:
            arr = (arr - 128) * r.uniform(0.6, 1.3) + 128 + r.uniform(-30, 30)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if r.random() < 0.5:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=r.randint(20, 95))
            img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
        return img

    # ------------------------------------------------------------------ main
    def _text_block(self, pieces, size_range=None, size=None):
        r = self.r
        size = size or r.randint(*size_range)
        stroke = r.choice([0, 0, 0, 0, 1, 2, max(1, size // 10)])
        keycap = r.random() < 0.05
        fill, strk = self.render_pieces(pieces, size, stroke, keycap)
        return self._transform([fill, strk])

    def _number_block(self, pieces, max_w=IMG - 4, max_h=IMG - 4, size_range=(MIN_TEXT, 64)):
        """Numara 128 px'e sığmıyorsa okunamaz boyuta KÜÇÜLTMEK yerine numaranın
        bir parçasını göster (içinde mutlaka rakam olsun). Küçültülmüş hali
        MIN_TEXT px'in altına inmez -> 'numara var' etiketi hep görünür olur.
        Döner: (maskeler, gerçekten çizilen parçalar)."""
        r = self.r
        size = r.randint(*size_range)
        for _ in range(8):
            masks = self._text_block(pieces, size=size)
            w, h = masks[0].size
            s = min(1.0, max_w / w, max_h / h)
            if size * s >= MIN_TEXT or len(pieces) <= 2:
                break
            keep = max(2, int(len(pieces) * size * s / MIN_TEXT * 0.9))
            for _ in range(10):
                st = r.randint(0, len(pieces) - keep)
                sub = pieces[st:st + keep]
                if any(p[0] in ALL_DIGITS for p in sub):
                    pieces = sub
                    break
            else:
                size = max(MIN_TEXT, int(size * 0.8))
        return self._fit(masks, max_w, max_h), pieces

    def sample(self, label, force_set=None):
        r = self.r
        img = self.background()
        meta = {"label": label, "kind": "empty"}
        partial = False

        # distractor words (appear in BOTH classes so text alone is not a cue)
        n_distr = r.choice([0, 1, 1, 2, 3]) if label == 0 else r.choice([0, 0, 1, 2])
        if label == 0 and r.random() < 0.12:
            n_distr = 0   # pure background negative
        for _ in range(n_distr):
            if label == 0 and r.random() < 0.12:
                masks = self._transform([self.scribble_mask(r.randint(12, 40)), None])
            else:
                masks = self._text_block(self.word_pieces(), (MIN_TEXT, 56))
            img, _ = self.composite(img, masks[0], masks[1], fully_inside=r.random() < 0.5)
            meta["kind"] = "words"

        if label == 1:
            if self.mnist is not None and force_set is None and r.random() < 0.08:
                masks = self._transform([self.mnist_mask(r.randint(12, 60)), None])
                meta["kind"] = "mnist"
                meta["set"] = "handwritten"
            else:
                dset = force_set or ("ascii" if r.random() < 0.2 else r.choice(self.digit_sets))
                partial = force_set is None and r.random() < 0.12
                pieces, _ = self.number_pieces(dset, allow_prefix=not partial)
                meta["kind"] = "number"
                meta["set"] = dset
                if partial:  # kenardan taşan uzun numara (sadece rakam+ayraç, önek yok)
                    masks = self._text_block(pieces, (20, 64))
                    img, _ = self.composite(img, masks[0], masks[1], fully_inside=False,
                                            place=(r.randint(-masks[0].width // 3, IMG // 3),
                                                   r.randint(0, max(0, IMG - masks[0].height))))
                else:
                    masks, _ = self._number_block(pieces)
            if not partial:
                masks = self._fit(masks, IMG - 4, IMG - 4)
                img, _ = self.composite(img, masks[0], masks[1], fully_inside=True)
        return np.asarray(self.degrade(img), dtype=np.uint8), meta

    # ------------------------------------------------------- okuyucu (OCR) verisi
    def sample_line(self):
        """64x256 satır görseli + okunacak rakam dizisi (ASCII, ör. '05340012445').
        Rakam hangi Unicode setinde / hangi taklit harfle yazılmış olursa olsun
        etiket normal rakamdır. %15 örnek sadece kelime içerir (etiket '')."""
        r = self.r
        self.line_mode = True
        try:
            img = self.background(LINE_W, LINE_H)
            label = ""
            if r.random() < 0.15:
                masks = self._fit(self._text_block(self.word_pieces(), (18, 40)), LINE_W - 4, LINE_H - 4)
            else:
                dset = "ascii" if r.random() < 0.2 else r.choice(self.digit_sets)
                pieces, _ = self.number_pieces(dset)
                masks, pieces = self._number_block(pieces, LINE_W - 4, LINE_H - 4, (18, 44))
                label = "".join(p[2] for p in pieces if len(p) > 2)
            # İlanlarda numaranın üstünde/altında çoğu zaman başka yazı satırı olur ("EV SAHİBİNİN
            # NUMARASI" / "BU NUMARAYI ARAYIN"). Şeridin kenarından kesilmiş yarım satırlar ekle ki
            # okuyucu bunları rakam sanmasın (etiket değişmez: sadece ortadaki numara okunur).
            if r.random() < 0.5:
                for side in ("top", "bottom"):
                    if r.random() < 0.7:
                        wm = self._fit(self._text_block(self.word_pieces(), (18, 40)), LINE_W * 2, LINE_H)
                        h = wm[0].height
                        vis = r.uniform(0.15, 0.45) * h          # görünen kısım
                        y = int(vis - h) if side == "top" else int(LINE_H - vis)
                        img, _ = self.composite(img, wm[0], wm[1], fully_inside=False,
                                                place=(r.randint(-wm[0].width // 4, LINE_W // 3), y))
            img, _ = self.composite(img, masks[0], masks[1], fully_inside=True)
            return np.asarray(self.degrade(img), dtype=np.uint8), label[:MAX_LABEL]
        finally:
            self.line_mode = False
