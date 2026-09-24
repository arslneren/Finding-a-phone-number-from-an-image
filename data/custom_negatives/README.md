# Kendi numarasız fotoğraflarınız (zor negatifler)

`mine_negatives.py` bu klasördeki fotoğrafları modelin gördüğü gibi karolara böler. Modelin yanlışlıkla "numara var" dediği karoları toplar ve bunları ince ayarda **"numara yok"** örneği olarak kullanır. Gerçek ilan fotoğraflarındaki yanlış alarmları azaltmanın en etkili yolu budur.

- Fotoğraflarda **hiç rakam olmamalı**. Buradan çıkan her karo "numara yok" olarak eğitime girer.
- Biçim: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`. Alt klasör kullanabilirsiniz.

```bash
python mine_negatives.py
python train.py --arch cnn --epochs 3 --lr 3e-4 --init models/digit_detector_cnn.h5 --hard_neg data/hard_neg.npy
```

Bu klasörün içeriği GitHub'a yüklenmez (`.gitignore`).
