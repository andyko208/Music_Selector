# Music Selector

A TensorFlow-based application that determines personal music preferences by classifying audio files into three categories: **Chills**, **Hypes**, and **Trips**. The system uses Convolutional Neural Networks (CNN) to analyze MFCC (Mel-frequency cepstral coefficients) features extracted from audio files.

## 🎵 Features

- **Audio Classification**: Automatically categorizes music into three distinct mood-based categories
- **Deep Learning**: Uses CNN architecture for robust feature learning
- **Easy to Use**: Simple API for preprocessing, training, and prediction
- **Visualization**: Built-in performance plotting and data visualization
- **Flexible**: Supports custom datasets and model parameters

## 📁 Project Structure

```
Music_Selector/
├── Music_Selector.ipynb          # Main Jupyter notebook with all functions
├── music_selector.py             # Python module with organized functions
├── API_DOCUMENTATION.md          # Comprehensive API documentation
├── QUICK_START.md               # Quick start guide
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Music_Selector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your audio data** (see [Quick Start Guide](QUICK_START.md) for details)

4. **Run the application**:
   ```python
   import music_selector as ms
   model, history, test_acc = ms.complete_workflow()
   ```

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get up and running in 10 minutes
- **[API Documentation](API_DOCUMENTATION.md)** - Comprehensive function reference
- **[Python Module](music_selector.py)** - Organized, reusable code with docstrings

## 🎯 Music Categories

The system classifies music into three categories:

| Category | Label | Description |
|----------|-------|-------------|
| **Chills** | 0 | Relaxing, ambient, and calming music |
| **Hypes** | 1 | Energetic, upbeat, and exciting music |
| **Trips** | 2 | Psychedelic, experimental, and mind-bending music |

## 🔧 How It Works

1. **Audio Processing**: Loads WAV files and segments them into 20-second chunks
2. **Feature Extraction**: Computes MFCC features using librosa
3. **Model Training**: Trains a CNN on the extracted features
4. **Classification**: Predicts music category for new audio files

### Model Architecture

```
Input: (862, 20, 1) MFCC features
├── Conv2D(32, 3x3) + ReLU
├── MaxPooling2D(3x3, stride=2) + BatchNormalization
├── Conv2D(32, 3x3) + ReLU
├── MaxPooling2D(3x3, stride=2) + BatchNormalization
├── Conv2D(32, 2x2) + ReLU
├── MaxPooling2D(3x3, stride=2) + BatchNormalization + Dropout(0.3)
├── Flatten()
├── Dense(64) + ReLU
└── Dense(10) + Softmax
```

## 📊 Performance

Typical performance metrics:
- **Training Accuracy**: ~95-100%
- **Validation Accuracy**: ~60-70%
- **Test Accuracy**: ~60-70%

*Note: Performance may vary based on data quality and quantity*

## 🛠️ Usage Examples

### Basic Usage

```python
import music_selector as ms

# Run complete workflow
model, history, test_acc = ms.complete_workflow()
```

### Custom Training

```python
import music_selector as ms

# Preprocess data
ms.preprocess_data('Training')
ms.preprocess_data('Test')

# Load and prepare data
train_x, train_y = ms.load_data('Training')
test_x, test_y = ms.load_data('Test')
Xtrain, Xval, Xtest, ytrain, yval, ytest = ms.prepare_datasets(train_x, train_y, 0.2)

# Train with custom parameters
model, history = ms.train_model(
    Xtrain, ytrain, Xval, yval,
    epochs=100,
    batch_size=32,
    learning_rate=0.0001
)

# Evaluate
test_acc, sample_acc = ms.evaluate_model(model, Xtest, ytest)
```

### Individual Predictions

```python
# Make prediction on specific sample
correct = ms.make_prediction(model, Xtest, ytest, 0)
```

## 📋 Requirements

### Python Dependencies

- **TensorFlow** >= 2.0.0
- **librosa** >= 0.8.0
- **numpy** >= 1.19.0
- **scikit-learn** >= 0.24.0
- **matplotlib** >= 3.3.0

### Audio Data Requirements

- **Format**: WAV files
- **Naming**: `{type}_{number}.wav` (e.g., `chill_1.wav`, `hype_2.wav`)
- **Duration**: Variable length (segmented into 20-second chunks)
- **Quality**: High-quality audio recommended

## 🗂️ Data Structure

```
Music_Selector/
├── Training/
│   ├── Chills/
│   │   ├── chill_1.wav
│   │   ├── chill_2.wav
│   │   └── ...
│   ├── Hypes/
│   │   ├── hype_1.wav
│   │   ├── hype_2.wav
│   │   └── ...
│   └── Trips/
│       ├── trip_1.wav
│       ├── trip_2.wav
│       └── ...
└── Test/
    ├── Chills/
    ├── Hypes/
    └── Trips/
```

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Install dependencies with `pip install -r requirements.txt`
2. **File not found**: Check directory structure and file naming
3. **Memory errors**: Reduce batch size or audio segment length
4. **Poor performance**: Verify audio quality and labeling

For detailed troubleshooting, see the [Quick Start Guide](QUICK_START.md).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please refer to the original repository for licensing information.

## 🙏 Acknowledgments

- **librosa**: Audio processing library
- **TensorFlow**: Deep learning framework
- **scikit-learn**: Machine learning utilities

## 📞 Support

For questions and support:
- Check the [API Documentation](API_DOCUMENTATION.md)
- Review the [Quick Start Guide](QUICK_START.md)
- Examine the [Python module](music_selector.py) for detailed function documentation

---

**Happy Music Classification! 🎵🎶**
