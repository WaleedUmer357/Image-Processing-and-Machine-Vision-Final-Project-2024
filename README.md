 Image-Processing-and-Machine-Vision-Final-Project-2024
Machine vision 2024 final project on German traffic sign recognition with YOLO.
 Real-Time Traffic Sign Recognition with YOLOv5

  Project Overview
This project implements a real-time traffic sign recognition system using the YOLOv5 object detection algorithm trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The system detects and classifies 43 different types of traffic signs with high accuracy, suitable for applications in autonomous vehicles and intelligent transportation systems.

  Key Performance Metrics
| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|----------|---------------|
| YOLOv5s (SGD) | 90.7% | 89.5% | 92.0% | 86.9% |
| YOLOv5s (Adam) | 77.2% | 67.4% | 71.2% | 67.0% |
| Preliminary Model | ~95% | ~90% | ~95% | ~80% |

 Project Objectives
1. Fine-tune YOLOv5 models for traffic sign detection on the GTSRB dataset
2. Implement comprehensive data augmentation strategies
3. Compare SGD and Adam optimizer performance
4. Achieve ≥95% accuracy for traffic sign classification
5. Develop a real-time detection system suitable for ITS applications

  Dataset Information
- Dataset: German Traffic Sign Recognition Benchmark (GTSRB)
- Original Images: 43,000+ realistic traffic sign images
- Classes: 43 different traffic sign categories
- Final Augmented Dataset: 67,020 images
- Data Split: Train/Validation/Test with balanced class distribution

  Data Augmentation Techniques
 Spatial Transformations
- Random 90° rotations (p=0.5)
- Horizontal and vertical flips (p=0.5)
- Shift, scale, and rotation (shift limit: 0.0625, scale limit: 0.1, rotation limit: 45°)

 Photometric Transformations
- Brightness and contrast adjustments (p=0.3)
- Gaussian blur (p=0.3)
- Hue, saturation, and value alterations (p=0.3)
- CLAHE enhancement (p=0.2)
- Gamma adjustments (p=0.2)
- Channel shuffling (p=0.2)
- Motion blur (p=0.2)

  Model Architecture
- Base Model: YOLOv5s (small variant)
- Backbone: CSPDarknet
- Neck: PANet
- Head: YOLO detection layers
- Activation Functions: SiLU (hidden layers), Sigmoid (output layer)
- Loss Functions: BCE for classification/objectness, CIoU for localization

  Training Configuration
 Hardware
- GPU: Tesla V100 (16GB) on Google Colab
- CPU: For preliminary model training
- Training Time: 1-2 days for 100 epochs

 Hyperparameters
- Image Size: 640×640
- Batch Size: 16
- Epochs: 100
- Optimizers: SGD vs Adam comparison
- Learning Rate: 0.001 (Adam), adaptive scheduling
- Loss Functions: Box, objectness, and classification losses

 Evaluation Metrics
- Precision: Proportion of correctly identified traffic signs
- Recall: Ability to identify all relevant traffic signs
- mAP@0.5: Mean Average Precision at IoU threshold 0.5
- mAP@0.5:0.95: Average mAP across IoU thresholds from 0.5 to 0.95
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Class-wise performance analysis

  Key Findings
1. SGD vs Adam: SGD showed better generalization with 90.7% precision vs Adam's 77.2%
2. Data Imbalance: Class distribution imbalance affected minority class performance
3. Augmentation Impact: Extensive augmentation improved model robustness
4. Convergence: SGD required more epochs but achieved better final performance
5. Real-time Capability: Both models demonstrated real-time detection capabilities

  Installation & Usage

 Prerequisites
```bash
Python 3.8+
PyTorch 1.7+
CUDA-capable GPU (recommended)
```

 Setup
```bash
 Clone repository
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition

 Install dependencies
pip install -r requirements.txt

 Clone YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -qr requirements.txt
```

 Training
```bash
 Train with SGD optimizer
python train.py --img 640 --batch 16 --epochs 100 --data gtsrb.yaml --weights yolov5s.pt --optimizer SGD

 Train with Adam optimizer
python train.py --img 640 --batch 16 --epochs 100 --data gtsrb.yaml --weights yolov5s.pt --optimizer Adam
```

 Inference
```python
 Load trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

 Perform detection
results = model('traffic_image.jpg')
results.show()
```

  Project Structure
```
traffic-sign-recognition/
├── data/
│   ├── GTSRB/                  Original dataset
│   ├── augmented/              Augmented images
│   └── splits/                 Train/val/test splits
├── models/
│   ├── yolov5s_sgd.pt         SGD-trained weights
│   └── yolov5s_adam.pt        Adam-trained weights
├── notebooks/
│   ├── data_augmentation.ipynb
│   ├── training_sgd.ipynb
│   └── evaluation.ipynb
├── scripts/
│   ├── augment_data.py         Albumentations pipeline
│   ├── convert_format.py       YOLO format conversion
│   └── evaluate_model.py       Performance evaluation
├── configs/
│   ├── gtsrb.yaml             Dataset configuration
│   └── hyp_custom.yaml        Custom hyperparameters
└── results/
    ├── metrics/               Training metrics
    ├── plots/                 Performance plots
    └── inference/             Sample detections
```

 Experimental Results

 SGD Model Performance
- Achieved 92.0% mAP@0.5 and 86.9% mAP@0.5:0.95
- Stable precision (~0.99) and recall (~0.9) throughout training
- Better generalization compared to Adam optimizer

 Adam Model Performance
- Faster convergence but lower final accuracy
- Achieved 71.2% mAP@0.5 and 67.0% mAP@0.5:0.95
- More sensitive to learning rate fluctuations

  Challenges & Solutions
1. Class Imbalance: Some classes underrepresented in dataset
   - Solution: Targeted augmentation for minority classes
   
2. Augmentation Complexity: Bounding box transformations challenging
   - Solution: Albumentations library with bounding box support
   
3. Hardware Limitations: Training on CPU for preliminary model
   - Solution: Google Colab Pro for GPU acceleration
   
4. Optimizer Selection: Trade-off between convergence speed and final accuracy
   - Solution: Comparative study of SGD vs Adam

 References
1. Stallkamp et al. - "German Traffic Sign Recognition Benchmark" (2012)
2. Redmon et al. - "You Only Look Once: Unified, Real-Time Object Detection" (2016)
3. Ultralytics YOLOv5 - GitHub repository and documentation
4. Albumentations - Fast image augmentation library

  Authors
Waleed Umer (s5336317)  
Griffith University  
Course: 7510ENG - Image Processing and Machine Vision  
Supervisors: Mr. Joel Dedini, Dr. Andrew Busch

 License
This project is for academic purposes as part of Griffith University coursework. The GTSRB dataset is publicly available for research use.

 Acknowledgments
- Griffith University School of Engineering
- Ultralytics for YOLOv5 implementation
- Google Colab for computational resources
- GTSRB dataset creators and maintainers

---

This project demonstrates the effectiveness of YOLOv5 for real-time traffic sign recognition, achieving high accuracy suitable for autonomous vehicle applications.
