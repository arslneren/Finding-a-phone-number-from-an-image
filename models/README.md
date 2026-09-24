# Eğitilmiş modeller

Model dosyaları repoda yok. Aşağıdaki komutlar bu klasöre yazar:

| Dosya | Üreten komut |
|---|---|
| `digit_detector_cnn.h5` | `python train.py --arch cnn --epochs 15`, ardından zor negatiflerle ince ayar |
| `digit_reader.h5` | `python train_reader.py --epochs 10` |
| `digit_detector_mlp.h5` | `python train.py --arch mlp` (yalnızca karşılaştırma için) |
| `metrics_*.json`, `history_*.csv` | Eğitim ve test sonuçları |

Adımların tamamı ana `README.md` dosyasında.
