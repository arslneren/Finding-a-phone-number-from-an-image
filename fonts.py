"""System font discovery and per-glyph coverage checks."""
import os
import glob
import json
import functools

from PIL import ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "data", "font_coverage.json")

FONT_DIRS = [
    os.path.join(HERE, "fonts"),                       # user-supplied extra fonts
    os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"),
    os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
    "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
    "/System/Library/Fonts", "/Library/Fonts",
]

# Symbol/icon fonts map ASCII code points to pictures -> would poison labels.
BLOCKLIST = {
    "wingding", "wingdng2", "wingdng3", "webdings", "marlett", "symbol", "holomdl2",
    "segmdl2", "bssym7", "mtextra", "refspcl", "outlook", "seguiemj", "msuighub",
    "msuighur", "segoeicons", "segfluenticons", "parchm", "goudysto",
}

MISSING_PROBE = "\U000F0000"   # private-use plane, no font has it -> .notdef glyph
PROBE_SIZE = 32


def find_font_files():
    files = []
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for ext in ("ttf", "otf", "ttc", "TTF", "OTF", "TTC"):
            files += glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)
    seen, out = set(), []
    for f in files:
        key = os.path.basename(f).lower()
        if key in seen or os.path.splitext(key)[0] in BLOCKLIST:
            continue
        seen.add(key)
        out.append(f)
    return sorted(out)


def font_faces(path):
    """(path, index) for each face in a .ttc collection, or just index 0."""
    faces = []
    for i in range(8 if path.lower().endswith(".ttc") else 1):
        try:
            ImageFont.truetype(path, PROBE_SIZE, index=i)
            faces.append((path, i))
        except OSError:
            break
    return faces


def _glyph_signature(font, ch):
    mask = font.getmask(ch)
    return mask.size, bytes(mask)


def has_glyph(font, ch, missing_sig):
    try:
        sig = _glyph_signature(font, ch)
    except Exception:
        return False
    if sig[0][0] == 0 or sig[0][1] == 0 or not any(sig[1]):
        return False
    return sig != missing_sig


def build_coverage(charsets, force=False):
    """charsets: {name: [chars]} -> {name: [[path, index], ...]} of faces that
    render *every* non-None char of the set. Cached to data/font_coverage.json."""
    if not force and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        if set(cached) >= set(charsets):
            return cached

    coverage = {name: [] for name in charsets}
    for path in find_font_files():
        for face in font_faces(path):
            try:
                font = ImageFont.truetype(face[0], PROBE_SIZE, index=face[1])
                missing = _glyph_signature(font, MISSING_PROBE)
            except Exception:
                continue
            for name, chars in charsets.items():
                if all(has_glyph(font, c, missing) for c in chars if c is not None):
                    coverage[name].append(list(face))
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=1)
    return coverage


@functools.lru_cache(maxsize=4096)
def load_font(path, index, size):
    return ImageFont.truetype(path, size, index=index)
