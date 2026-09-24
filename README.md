# Görselde telefon numarası bulma

İlan görsellerinde **telefon numarası olup olmadığını** tespit eden ve bulduğu numarayı **okuyan** iki Keras YSA modeli ile bir web arayüzü.
Normal rakamların yanında filtre atlatmak için kullanılan Unicode varyantlarını da tanır:

| Set | Örnek | Set | Örnek |
|---|---|---|---|
| ASCII | 0532 | Fullwidth | ０５３２ |
| Math Bold / Sans / Mono / Double-struck | 𝟎𝟓𝟑𝟐 𝟢𝟧𝟥𝟤 𝟶𝟻𝟹𝟸 𝟘𝟝𝟛𝟚 | Üst / alt simge | ⁰⁵³² ₀₅₃₂ |
| Daire içi (normal, negatif, sans, çift) | ⓪⑤③② ⓿❺❸❷ ➄➂➁ ⓹⓷⓶ | Parantezli / noktalı / virgüllü | ⑸⑶⑵ ⒌⒊⒉ 🄆🄄🄃 |
| Arapça-Hint / Farsça | ٠٥٣٢ ۰۵۳۲ | Devanagari / Bengalce / Tay | ०५३२ ০৫৩২ ๐๕๓๒ |
| Harf taklidi karışık | O5З2 4l8 | El yazısı (MNIST) | ✍ |

Modeller ve veri repoda **yok**. Hepsi aşağıdaki adımlarla, sıfırdan üretilir.

## Modeller

| Model | Dosya | Görev | Mimari |
|---|---|---|---|
| Tespit | `models/digit_detector_cnn.h5` | 128×128 karoda rakam var mı? | Evrişimli YSA (Keras `Sequential`, global max-pool) |
| Okuyucu | `models/digit_reader.h5` | 64 px yüksekliğindeki satırdaki rakamları oku | CRNN: CNN + 2× çift yönlü LSTM + CTC |
| Karşılaştırma | `models/digit_detector_mlp.h5` | Tespitin aynısı | Klasik tam bağlantılı YSA (MLP) |

Okuyucu, rakam hangi Unicode setinde ya da hangi taklit harfle yazılmış olursa olsun normal rakam döndürür (`⓪⑤③④` → `0534`).

Tüm eğitimlerde veri **%80 eğitim / %20 test** olarak ayrılır (`train_test_split`, stratified). Test kısmı eğitimde hiç görülmez. Model seçimi, eğitim kısmından ayrılan %10'luk doğrulama dilimiyle yapılır.

### Sonuçlar (sentetik veri, GTX 1060)

| Test | Sonuç |
|---|---|
| Tespit CNN, %20 test (68.498 karo) | %92.7 doğruluk |
| Klasik MLP, %20 test | %52.4 (bu iş için uygun değil) |
| Uçtan uca, eğitimde görülmemiş 600 tam boyutlu görsel (eşik 0.9) | %94.3 doğruluk, %94.3 yakalama, %5.7 yanlış alarm |
| Cep numaralı görsellerde uyarı verilmesi | %95.7 |
| Numarasız görsellerde yanlış "cep" uyarısı | %0 |
| Okuyucu, %20 test (30.000 satır) | Satırların %74.5'i tam doğru, karakter hatası %8.9 |
| Analiz süresi (GPU) | Ortalama ~100 ms, medyan ~70 ms |

Arka planlar çiçek fotoğrafları olduğu için bu sayılar gerçek ilan görsellerindeki başarıyı birebir yansıtmaz. Gerçek fotoğraflarla ince ayar için aşağıdaki "Kendi verinizi kullanma" bölümüne bakın.

## Veri

Bu kadar Unicode varyantını içeren hazır, etiketli bir veri seti olmadığı için veri **sentetik** olarak üretilir:

- **İnternetten otomatik indirilenler:** TensorFlow `flower_photos` (3.670 gerçek fotoğraf, arka plan olarak) ve MNIST (70.000 el yazısı rakam).
- **Pozitifler:** Telefon, fiyat, m² ve rastgele gruplu numaralar. Bunlar 22 Unicode setinde, set karışımıyla ve harf taklitleriyle, sistemdeki fontlarla çizilir. Her fontun glifi gerçekten çizip çizemediği kontrol edilir.
- **Negatifler:** Aynı efektlerle çizilmiş Türkçe/İngilizce ilan kelimeleri, süslü Unicode harfler, ☎ gibi semboller, karalamalar ve "SOLO", "ZOO" gibi rakama benzeyen kelimeler.
- **Zor negatifler (`mine_negatives.py`):** Tam boyutlu, numarasız fotoğraflarda modelin yanlış alarm verdiği karolar toplanıp eğitime eklenir.
- **Okuyucu verisi:** Numaranın üstünde ve altında kesik yazı satırları olan örnekler de üretilir ("EV SAHİBİNİN NUMARASI / 0530 … / BU NUMARAYI ARAYIN" gibi ilan düzenleri).

### Kendi verinizi kullanma

| Klasör | Ne konur | Ne işe yarar |
|---|---|---|
| `data/custom_backgrounds/` | **Rakam içermeyen** gerçek fotoğraflar (ör. ilan fotoğrafları) | Sentetik numaralar bu fotoğrafların üzerine çizilir. Model gerçek görsellere uyum sağlar. |
| `data/custom_negatives/` | **Rakam içermeyen** gerçek fotoğraflar | Modelin bunlarda verdiği yanlış alarmlar toplanıp "numara yok" olarak eğitilir. |
| `fonts/` | Ek `.ttf` / `.otf` fontlar | Veri üretiminde kullanılan font çeşitliliği artar. |

Klasörlere fotoğraf koyduktan sonra "Kurulum ve kullanım" adımlarını baştan çalıştırın. `prepare_data.py` yeni fotoğrafları otomatik algılar. Yalnızca kendi fotoğraflarınızı kullanmak için `--no_flowers` ekleyin. Klasörlerin içeriği GitHub'a yüklenmez.

## Kurulum ve kullanım

```bash
pip install -r requirements.txt

python prepare_data.py --big 320000          # tespit verisi (diske, ~5 GB; flower_photos + MNIST indirilir)
python prepare_data.py --lines 150000        # okuyucu verisi
python train.py --arch cnn --epochs 15       # tespit modeli, %80/%20
python mine_negatives.py                     # zor negatif madenciliği
python train.py --arch cnn --epochs 3 --lr 3e-4 --init models/digit_detector_cnn.h5 --hard_neg data/hard_neg.npy
python train_reader.py --epochs 10           # okuyucu, %80/%20
python eval_images.py --n 300                # tam boyutlu görsellerde uçtan uca test

python web.py                                # http://localhost:8000
```

Disk ihtiyacı yaklaşık 10 GB. GPU'da tüm eğitim birkaç saat sürer, CPU'da çok daha uzun.

### GPU (Windows)

TensorFlow 2.10'dan sonraki sürümler Windows'ta GPU'yu doğrudan desteklemiyor. GPU için ayrı bir sanal ortam kullanılır:

```bash
python -m venv .venv-gpu
.venv-gpu\Scripts\python -m pip install "tensorflow==2.10.1" "numpy<1.24" pillow scikit-learn nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-cudnn-cu11==8.9.4.19 nvidia-cufft-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11
```

Scriptler `.venv-gpu\Scripts\python` ile çalıştırılır. `gpu_setup.py`, CUDA DLL'lerini otomatik bulur. Modeller `.h5` biçiminde kaydedildiği için CPU'daki TF 2.15 ile de açılır.

## Web arayüzü

`python web.py` → http://localhost:8000. Görsel sürükleyin, seçin ya da Ctrl+V ile yapıştırın.

| Durum | Arayüz |
|---|---|
| Cep telefonu (5XX XXX XX XX; 0 / 90 / +90 önekli) | 🔴 "Bu görselde telefon numarası bulunmaktadır." |
| Sabit hat, 9-12 haneli ama formatı tutmayan numara, okunamayan rakam | 🟠 "…olabilir, kontrol edin" |
| Telefon olmayan numara (ilan kodu `2501-2159`, fiyat, m²) | ℹ️ Uyarı yok, yalnızca bilgi |
| Rakam yok | ✓ |

Ekranda ayrıca numara bulunma olasılığı, okunan numara ve işaretli görsel gösterilir: şüpheli bölgeler turuncu, okunan numara kırmızı kutu içinde.

Dışarıya paylaşmak için (ör. ngrok) mutlaka şifre koyun: `ngrok http 8000 --basic-auth "kullanici:sifre"`.

## Dosyalar

- `unicode_sets.py`, `fonts.py`, `synth.py`, `prepare_data.py`: veri üretimi
- `model.py`, `train.py`: tespit YSA'ları (CNN ve MLP)
- `reader.py`, `train_reader.py`: okuyucu (CRNN + CTC)
- `mine_negatives.py`: zor negatif madenciliği
- `analyzer.py`, `predict.py`: tespit, okuma ve telefon sınıflandırma hattı
- `web.py`, `web/index.html`: web arayüzü
- `gpu_setup.py`: Windows GPU ayarları
- `test_phone.py`: belirli bir numaranın (ör. 05340012445) tüm Unicode setlerinde test edilmesi
- `eval_images.py`, `errors.py`, `make_demo_images.py`: test, hata analizi ve demo görselleri

## Sınırlar

- Yazıyla yazılmış numaralar ("sıfır beş yüz otuz iki") kapsam dışıdır.
- `segmented` (🯰🯱) seti için Windows'ta font yoktur. `fonts/` klasörüne *Noto Sans Symbols 2* koyup `data/font_coverage.json` dosyasını silin.
- Okuyucu uzun ve süslü numaralarda bazen bir-iki rakamı kaçırır. Bu yüzden 9-12 haneli, formatı tutmayan okumalar da "kontrol edin" uyarısı alır.
