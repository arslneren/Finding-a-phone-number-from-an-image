# Kendi arka plan fotoğraflarınız

Buraya koyduğunuz fotoğraflar sentetik veri üretiminde **arka plan** olarak kullanılır. Rakamlar ve kelimeler bu fotoğrafların üzerine çizilir.

- Gerçek ilan fotoğrafları (oda, bina, cephe) koymak, modeli gerçek görsellere çok daha iyi uyarlar.
- Fotoğraflarda **hiç rakam olmamalı** (telefon, ilan kodu, fiyat, filigran dahil). Aksi halde "numara yok" diye etiketlenen örneklerde rakam bulunur ve model yanlış öğrenir.
- Biçim: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`. Alt klasör kullanabilirsiniz.

Fotoğrafları ekledikten sonra veriyi yeniden üretin:

```bash
python prepare_data.py --big 320000               # flower_photos + bu klasör
python prepare_data.py --big 320000 --no_flowers  # sadece bu klasör
```

Bu klasörün içeriği GitHub'a yüklenmez (`.gitignore`).
