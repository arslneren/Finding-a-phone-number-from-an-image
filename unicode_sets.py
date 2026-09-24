"""Unicode digit variants people use to sneak numbers past filters, plus
digit-free character sets used for negative (no number) samples."""

def _seq(start, n=10):
    return [chr(start + i) for i in range(n)]


# Each set is a list of 10 glyphs for 0..9. None = that digit does not exist
# in the set (e.g. no "circled zero" in the parenthesized block); the generator
# substitutes a digit from another set in that case.
DIGIT_SETS = {
    "ascii":               _seq(0x30),
    "fullwidth":           _seq(0xFF10),
    "math_bold":           _seq(0x1D7CE),
    "math_double_struck":  _seq(0x1D7D8),
    "math_sans":           _seq(0x1D7E2),
    "math_sans_bold":      _seq(0x1D7EC),
    "math_monospace":      _seq(0x1D7F6),
    "superscript":         ["⁰", "¹", "²", "³"] + _seq(0x2074, 6),
    "subscript":           _seq(0x2080),
    "circled":             ["⓪"] + _seq(0x2460, 9),
    "circled_neg":         ["⓿"] + _seq(0x2776, 9),
    "circled_sans":        [None] + _seq(0x2780, 9),
    "circled_sans_neg":    [None] + _seq(0x278A, 9),
    "double_circled":      [None] + _seq(0x24F5, 9),
    "parenthesized":       [None] + _seq(0x2474, 9),
    "full_stop":           ["\U0001F100"] + _seq(0x2488, 9),
    "comma":               _seq(0x1F101),
    "arabic_indic":        _seq(0x0660),
    "ext_arabic_indic":    _seq(0x06F0),
    "devanagari":          _seq(0x0966),
    "bengali":             _seq(0x09E6),
    "thai":                _seq(0x0E50),
    "segmented":           _seq(0x1FBF0),
}

# Letters/symbols that visually pass for digits. Used *inside* numbers
# (mixed with real digits) so the model learns e.g. "O532 l23 45 67" is a number.
LOOKALIKES = {
    0: ["O", "o", "Ο", "О", "Ø", "○", "⭘"],
    1: ["l", "I", "|", "ı", "І", "Ⅰ", "!"],
    2: ["Z", "Ƨ"],
    3: ["З", "Ʒ", "Ӡ"],
    4: ["Ꮞ", "A"],
    5: ["S", "Ƽ"],
    6: ["b", "б", "G"],
    7: ["⅄", "T"],
    8: ["B", "Ȣ", "∞"],
    9: ["g", "q", "१"],
}

# --- negatives -------------------------------------------------------------

def _alpha(start_upper, start_lower=None):
    s = _seq(start_upper, 26)
    if start_lower is not None:
        s += _seq(start_lower, 26)
    return s


LETTER_SETS = {
    "ascii":          [chr(c) for c in range(0x41, 0x5B)] + [chr(c) for c in range(0x61, 0x7B)]
                      + list("ÇĞİÖŞÜçğıöşü"),
    "fullwidth":      _alpha(0xFF21, 0xFF41),
    "math_bold":      _alpha(0x1D400, 0x1D41A),
    # U+1D455 (italic h) is unassigned; Unicode uses U+210E instead.
    "math_italic":    [c if c != "\U0001D455" else "ℎ" for c in _alpha(0x1D434, 0x1D44E)],
    "math_sans_bold": _alpha(0x1D5D4, 0x1D5EE),
    "math_monospace": _alpha(0x1D670, 0x1D68A),
    "circled":        _alpha(0x24B6, 0x24D0),
    "parenthesized":  _seq(0x249C, 26),
    "squared":        _seq(0x1F130, 26),
    "neg_squared":    _seq(0x1F170, 26),
    "greek":          _seq(0x391, 17) + _seq(0x3B1, 17),
    "cyrillic":       _seq(0x410, 32),
}

# Digit-free symbols that commonly appear next to phone numbers (hard negatives:
# a phone icon alone is NOT a number).
SYMBOLS = list("☎☏✆✉★☆♥✓✔✗→←↑↓•·—–-+*#@&%/\\()[]:;!?.,\"'") + [
    "☎", "✆", "\U0001F4DE", "\U0001F4F1", "✉", "★", "■", "●",
]

TR_WORDS = """satılık kiralık daire villa arsa residence site havuz otopark asansör
merkezi doğalgaz kombi eşyalı boş sahibinden acil fırsat deniz manzaralı bahçeli
dubleks tripleks çatı katı yeni bina krediye uygun takaslı metro yakın okul hastane
salon mutfak banyo balkon oda ebeveyn giyinme kiler teras site içi güvenlik
ara kat bahçe katı yüksek giriş bakımlı lüks geniş ferah aydınlık cadde üzeri
emlak ofisi danışman arayın ulaşın whatsapp iletişim telefon numara bilgi için
Kadıköy Beşiktaş Çankaya Karşıyaka Bornova Ataşehir Üsküdar Maltepe Bodrum Antalya
İstanbul Ankara İzmir Bursa Muğla ev konut işyeri dükkan ofis depo fabrika tarla""".split()

EN_WORDS = """for sale rent apartment house home call now contact info luxury view garden
pool parking new open offer best price owner agent real estate room studio loft
BOSS OSLO SOLO LOBBY BIG ZONE GOOD BEST BLISS SILL IO lol ill BOO ZOO""".split()
