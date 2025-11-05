# Image Classification Projects

This repository contains two comprehensive Jupyter notebooks demonstrating different approaches to image classification using deep learning with TensorFlow/Keras.

## 📚 Projects Overview

### 1. Image Classification with Keras (`image_classification_karas.ipynb`)
A beginner-friendly introduction to Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset.

**Features:**
- CNN architecture for multi-class image classification
- Training on CIFAR-10 dataset (10 classes, 60,000 images)
- Model evaluation and visualization
- Custom image prediction capabilities
- Test on your own images

**Classes:** airplane, car, bird, cat, deer, dog, frog, horse, ship, truck

### 2. Image Classification with Localization (`image_classification_with_localization.ipynb`)
An advanced multi-task learning project that performs both object classification and localization (bounding box prediction) using PASCAL VOC-style data.

**Features:**
- Multi-task learning architecture (classification + localization)
- Transfer learning with VGG16 backbone
- Bounding box regression
- IoU (Intersection over Union) metrics
- Custom loss functions for localization
- Support for PASCAL VOC dataset format
- Synthetic data generation for demonstration
- Real-time visualization of predictions with bounding boxes

**Classes:** person, cat, dog, car, bird

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- GPU (optional, but recommended for faster training)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd image-classification-main
```

2. **Install dependencies:**

For the basic classification notebook:
```bash
pip install -r requirements_classification.txt
```

For the localization notebook:
```bash
pip install -r requirements_localization.txt
```

Or install all dependencies at once:
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

## 📁 Project Structure

```
image-classification-main/
│
├── image_classification_karas.ipynb          # Basic CNN classification
├── image_classification_with_localization.ipynb  # Classification + localization
├── test_images/                               # Directory for custom test images
│   ├── airplane1.png
│   ├── airplane2.png
│   ├── bird1.png
│   ├── bird2.png
│   ├── sh1.png
│   └── sh2.png
├── requirements.txt                           # All dependencies
├── requirements_classification.txt            # Basic notebook dependencies
├── requirements_localization.txt              # Localization notebook dependencies
└── README.md                                  # This file
```

## 🎯 Usage

### Basic Image Classification

1. Open `image_classification_karas.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Load and preprocess CIFAR-10 dataset
   - Build and train a CNN model
   - Evaluate model performance
   - Test on custom images from `test_images/` directory

### Classification with Localization

1. Open `image_classification_with_localization.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Generate synthetic PASCAL VOC-style data (or download real PASCAL VOC dataset)
   - Build multi-task CNN with classification and localization heads
   - Train with custom IoU loss for bounding box prediction
   - Visualize predictions with bounding boxes
   - Test on custom images

## 📊 Datasets

### CIFAR-10
- **Size:** 60,000 images (50,000 training, 10,000 test)
- **Image Size:** 32×32 pixels, RGB
- **Classes:** 10 categories
- **Source:** Automatically downloaded by TensorFlow/Keras

### PASCAL VOC (Optional)
- **Size:** Varies (20+ object categories)
- **Image Size:** Variable (resized to 224×224)
- **Classes:** 20 categories (subset of 5 used in this project)
- **Source:** Can be downloaded from [PASCAL VOC website](http://host.robots.ox.ac.uk/pascal/VOC/)
- **Note:** The localization notebook includes synthetic data generation if PASCAL VOC is not available

## 🔧 Adding Custom Images

To test the models with your own images:

1. Add your images to the `test_images/` directory
2. Supported formats: PNG, JPG, JPEG, BMP, TIFF
3. Run the prediction cells in the notebooks
4. The models will automatically resize and preprocess your images

**Note:** 
- For basic classification: Images are resized to 32×32
- For localization: Images are resized to 224×224

## 📈 Model Performance

### Basic Classification Model
- **Architecture:** Custom CNN (3 Conv layers + Dense layers)
- **Expected Accuracy:** ~70-75% on CIFAR-10 test set
- **Training Time:** ~10-15 minutes (20 epochs, CPU)

### Localization Model
- **Architecture:** VGG16 backbone + Multi-task heads
- **Expected Performance:**
  - Classification Accuracy: ~85-90% (synthetic data)
  - Mean IoU: ~0.60-0.75 (synthetic data)
  - With real PASCAL VOC: 90-95% accuracy
- **Training Time:** ~20-30 minutes (20 epochs, CPU)

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **OpenCV** - Image processing
- **scikit-learn** - Machine learning utilities
- **Pillow** - Image manipulation

## 📖 Key Concepts Covered

### Basic Classification Notebook:
- Convolutional Neural Networks (CNNs)
- Image preprocessing and normalization
- Model training and evaluation
- Transfer of learning to custom images

### Localization Notebook:
- Multi-task learning
- Transfer learning with pre-trained models (VGG16)
- Bounding box regression
- IoU (Intersection over Union) metrics
- Custom loss functions
- Data augmentation preserving bounding boxes
- PASCAL VOC dataset format

## 🎓 Learning Outcomes

After completing these notebooks, you will understand:
1. How to build CNN architectures for image classification
2. Data preprocessing and augmentation techniques
3. Transfer learning and fine-tuning pre-trained models
4. Multi-task learning for classification and localization
5. Evaluation metrics (accuracy, IoU, confusion matrix)
6. How to deploy models on custom images
7. Best practices for computer vision projects

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for:
- Bug fixes
- New features
- Additional datasets
- Model improvements
- Documentation enhancements

## 📝 License

This project is open source and available for educational purposes.

## 🙋 Support

If you encounter any issues or have questions:
1. Check that all dependencies are installed correctly
2. Ensure you're using Python 3.7+
3. Verify GPU drivers if using GPU acceleration
4. Check image paths and formats for custom images

## 🔗 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

## 📧 Contact

For questions or feedback, please open an issue in this repository.

---

**Happy Learning! 🚀**
