"""Create full-size listing-style demo photos to sanity-check predict.py."""
import glob
import os
import random

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "samples")
F = "C:/Windows/Fonts/"

CASES = [  # (file, text, font, size, has_number)
    ("01_ascii_phone",       "0532 418 27 69",              "arialbd.ttf", 60, True),
    ("02_math_bold",         "𝟎𝟓𝟑𝟐 𝟒𝟏𝟖 𝟐𝟕 𝟔𝟗",              "seguisym.ttf", 56, True),
    ("03_fullwidth",         "０５４４ ９８７ ６５ ４３",     "msgothic.ttc", 48, True),
    ("04_circled",           "⓪⑤③② ④①⑧ ②⑦ ⑥⑨",           "seguisym.ttf", 52, True),
    ("05_neg_circled",       "⓿❺❸❷ ❹❶❽ ❷❼ ❻❾",           "seguisym.ttf", 52, True),
    ("06_double_struck",     "𝟘𝟝𝟛𝟚-𝟜𝟙𝟠-𝟚𝟟-𝟞𝟡",              "seguisym.ttf", 50, True),
    ("07_arabic_indic",      "٠٥٣٢ ٤١٨ ٢٧ ٦٩",              "DUBAI-BOLD.TTF", 60, True),
    ("08_lookalike_mix",     "O5З2 4l8 2७ 69",              "segoeui.ttf", 58, True),
    ("09_small_corner",      "Tel: 0555 123 45 67",         "calibri.ttf", 22, True),
    ("10_superscript",       "⁰⁵³² ⁴¹⁸ ²⁷ ⁶⁹",              "segoeui.ttf", 90, True),
    ("11_words_only",        "SATILIK DAİRE - SAHİBİNDEN",  "arialbd.ttf", 54, False),
    ("12_fancy_letters",     "𝐒𝐀𝐓𝐈𝐋𝐈𝐊 𝐕𝐈𝐋𝐋𝐀",                 "seguisym.ttf", 60, False),
    ("13_phone_icon_words",  "☎ Bilgi için arayın",         "seguisym.ttf", 56, False),
    ("14_circled_letters",   "ⓈⒶⓉⒾⓁⒾⓀ ⒹⒶⒾⓇⒺ",              "seguisym.ttf", 56, False),
    ("15_clean_photo",       "",                            "arial.ttf", 10, False),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    photos = sorted(glob.glob(os.path.join(HERE, "data", "flower_photos", "*", "*.jpg")))
    r = random.Random(42)
    for name, text, font, size, _ in CASES:
        im = Image.open(r.choice(photos)).convert("RGB").resize((1024, 720))
        if text:
            f = ImageFont.truetype(F + font, size)
            d = ImageDraw.Draw(im)
            w = d.textlength(text, font=f)
            x = r.randint(20, max(21, int(1024 - w - 20)))
            y = r.choice([40, 330, 620]) if size < 40 else r.randint(40, 600)
            d.text((x, y), text, font=f, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        im.save(os.path.join(OUT, name + ".jpg"), quality=88)
    print("demo görselleri ->", OUT)


if __name__ == "__main__":
    main()
