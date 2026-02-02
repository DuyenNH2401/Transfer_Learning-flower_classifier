# 🌸 Flower Classification Web App

Ứng dụng web nhận dạng 102 loại hoa sử dụng ResNet50 Transfer Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

## ✨ Tính năng

- 📤 **Upload ảnh** - Tải lên ảnh hoa từ thiết bị
- 📷 **Camera** - Chụp ảnh trực tiếp từ camera
- 🎯 **Top-5 predictions** - Hiển thị 5 dự đoán cao nhất với độ tin cậy
- 🌺 **102 loại hoa** - Nhận dạng đa dạng các loại hoa

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/flower-classifier.git
cd flower-classifier
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Thêm model files

Đảm bảo các file sau nằm trong thư mục app:
- `best_model.pt` - Model đã train
- `cat_to_name.json` - Mapping tên hoa

### 4. Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`

## 📦 Deploy lên Streamlit Cloud

1. Push code lên GitHub repository
2. Truy cập [share.streamlit.io](https://share.streamlit.io)
3. Kết nối GitHub repository
4. Chọn branch và file `app.py`
5. Deploy!

> ⚠️ **Lưu ý:** File model `best_model.pt` (~96MB) cần sử dụng Git LFS hoặc host riêng.

## 🏗️ Cấu trúc project

```
flower_app/
├── app.py              # Ứng dụng Streamlit chính
├── requirements.txt    # Dependencies
├── README.md           # Documentation
├── best_model.pt       # Model đã train (ResNet50)
└── cat_to_name.json    # Mapping class → tên hoa
```

## 🧠 Model

- **Architecture**: ResNet50 (pretrained) + Custom FC Layer
- **Classes**: 102 loại hoa
- **Input size**: 224x224 RGB
- **Training data**: PyTorch Flower Dataset

## 📸 Screenshots

*Coming soon...*

## 📄 License

MIT License © 2024

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
