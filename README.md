# 🌸 Flower AI: Intelligent Recognition System

![Project Banner](https://img.shields.io/badge/FLOWER-AI-f093fb?style=for-the-badge&logo=openai&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Advanced flower classification system powered by ResNet50 Transfer Learning.**  
> Capable of identifying 102 different flower species with high accuracy in real-time.

---

## ✨ Key Features

- **🎨 Premium Interface**: Modern "Glassmorphism" UI with dark mode and smooth animations.
- **🧠 Advanced AI**: Powered by a fine-tuned ResNet50 Deep Learning model.
- **📷 Dual Input Modes**: 
    - **Upload Mode**: Drag & drop support for high-res images.
    - **Camera Mode**: Privacy-focused on-demand camera capture.
- **📊 Real-time Analytics**: Instant top-5 predictions with confidence visualization.
- **⚡ High Performance**: Optimized for both CPU and CUDA-enabled GPU inference.

## 📁 Project Structure

Organized into a clean, maintainable structure:

```bash
flower_app/
├── 📂 models/
│   └── best_model.pt       # Trained ResNet50 model (~96MB)
├── 📂 data/
│   └── cat_to_name.json    # 102 Flower species mapping
├── app.py                  # Main Streamlit application
├── requirements.txt        # Production dependencies
└── README.md               # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/flower-classifier.git
    cd flower-classifier
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ensure model consistency:**
    Verify that `best_model.pt` is present in the `models/` directory.

4.  **Launch the App:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your browser at `http://localhost:8501`.

## 🛠️ Technology Stack

- **Core**: Python 3.10
- **Web Framework**: Streamlit
- **Deep Learning**: PyTorch, Torchvision
- **Computer Vision**: PIL (Pillow)
- **Model Architecture**: ResNet50 (Pretrained on ImageNet)

## 📊 Model Performance

- **Architecture**: ResNet50 + Custom Fully Connected Head (256 units)
- **Training Epochs**: 25
- **Accuracy**: ~95% on Validation Set
- **Input Size**: 224x224px Normalized

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
