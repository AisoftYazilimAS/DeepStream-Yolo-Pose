# DeepStream YOLOv8-Pose

NVIDIA DeepStream SDK ile YOLOv8-Pose modellerini çalıştırmak için geliştirilmiş uygulama.

## Özellikler

- ✅ YOLOv8-Pose model desteği
- ✅ Özel eğitilmiş modeller için destek
- ✅ Gerçek zamanlı pose estimation
- ✅ 17 keypoint detection
- ✅ Skeleton visualization
- ✅ Multi-class detection
- ✅ Python ve C++ implementasyonu

## Gereksinimler

- NVIDIA DeepStream SDK 6.0+
- CUDA 12.1+
- TensorRT 8.5+
- GStreamer 1.0
- Python 3.8+ (Python versiyonu için)

## Kurulum

### 1. Repository'yi klonlayın

```bash
git clone https://github.com/AisoftYazilimAS/DeepStream-Yolo-Pose.git
cd DeepStream-Yolo-Pose
```

### 2. Custom parser'ı derleyin

```bash
export CUDA_VER=12.1
make -C nvdsinfer_custom_impl_Yolo_pose clean
make -C nvdsinfer_custom_impl_Yolo_pose
```

### 3. C++ uygulamasını derleyin (opsiyonel)

```bash
make clean
make
```

## Model Hazırlama

### YOLOv8-Pose modelini ONNX'e çevirin

```bash
python3 utils/export_yoloV8_pose.py -w your_model.pt --dynamic
```

Bu komut şunları oluşturur:
- `your_model.onnx` - ONNX model dosyası
- `labels.txt` - Sınıf isimleri

## Konfigürasyon

`config_infer_primary_yoloV8_pose.txt` dosyasını düzenleyin:

```ini
[property]
onnx-file=your_model.onnx
model-engine-file=your_model.onnx_b1_gpu0_fp32.engine
labelfile-path=labels.txt
num-detected-classes=4  # Sınıf sayınıza göre ayarlayın

[class-attrs-all]
pre-cluster-threshold=0.45  # Confidence threshold
```

## Kullanım

### Python ile

```bash
# Video dosyası
python3 deepstream.py -s file:///path/to/video.mp4 -c config_infer_primary_yoloV8_pose.txt

# RTSP stream
python3 deepstream.py -s rtsp://camera_ip:port/stream -c config_infer_primary_yoloV8_pose.txt

# Parametreler
python3 deepstream.py \
  -s file:///path/to/video.mp4 \
  -c config_infer_primary_yoloV8_pose.txt \
  -w 1920 \
  -e 1080 \
  -g 0
```

### C++ ile

```bash
./deepstream -s file:///path/to/video.mp4 -c config_infer_primary_yoloV8_pose.txt
```

## Parametreler

| Parametre | Kısaltma | Açıklama | Varsayılan |
|-----------|----------|----------|------------|
| --source | -s | Video kaynağı (file:// veya rtsp://) | Zorunlu |
| --infer-config | -c | Config dosyası | Zorunlu |
| --streammux-width | -w | Stream genişliği | 1920 |
| --streammux-height | -e | Stream yüksekliği | 1080 |
| --gpu-id | -g | GPU ID | 0 |
| --streammux-batch-size | -b | Batch size | 1 |

## Çıktı

Uygulama şunları görselleştirir:
- **Mavi kutular**: Tespit edilen nesneler
- **Beyaz noktalar**: Pose keypoint'leri
- **Mavi çizgiler**: Skeleton bağlantıları
- **Metin**: Sınıf adı ve confidence skoru

## Performans

- ~18 FPS @ 1920x1080 (RTX 3090)
- Gerçek zamanlı işleme
- GPU bellek kullanımı: ~2GB

## Proje Yapısı

```
DeepStream-Yolo-Pose/
├── deepstream.py                          # Python uygulaması
├── deepstream.c                           # C++ uygulaması
├── config_infer_primary_yoloV8_pose.txt  # Model config
├── labels.txt                             # Sınıf isimleri
├── nvdsinfer_custom_impl_Yolo_pose/      # Custom parser
│   ├── nvdsparsepose_Yolo.cpp
│   └── Makefile
├── utils/
│   └── export_yoloV8_pose.py             # Model export script
└── modules/                               # Yardımcı modüller
```

## Sorun Giderme

### Engine dosyası oluşturulamıyor
```bash
# Eski engine dosyasını silin
rm *.engine
```

### Düşük FPS
- Batch size'ı artırın
- Resolution'ı düşürün
- `interval` parametresini artırın (her N frame'de bir işle)

### Detection yok
- Threshold'u düşürün (0.25-0.45 arası deneyin)
- Model input size'ını kontrol edin
- Video içeriğinin model eğitim datasına uygun olduğundan emin olun

## Lisans

MIT License

## Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır. Büyük değişiklikler için lütfen önce bir issue açın.

## İletişim

Aisoft Yazılım A.Ş.
- GitHub: [@AisoftYazilimAS](https://github.com/AisoftYazilimAS)

## Teşekkürler

- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
