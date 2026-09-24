"""Windows'ta pip ile kurulan NVIDIA CUDA/cuDNN DLL'lerini TensorFlow'dan ÖNCE
arama yoluna ekler. TensorFlow import edilmeden önce import edilmeli.
(Diğer ortamlarda hiçbir şey yapmaz.)"""
import glob
import os
import site

# GPU belleğini baştan tamamen ayırma, ihtiyaç kadar al (web sunucusu açıkken başka süreç de GPU kullanabilsin)
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
# cuDNN her YENİ giriş boyutunda algoritma denemesi yapar (istek başına +400-800 ms); bu küçük
# modellerde kazancı yok, kapat. (analyzer.py ayrıca girişleri sabit boyutlara tamamlar.)
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")

if os.name == "nt":
    for sp in site.getsitepackages():
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            os.add_dll_directory(d)
