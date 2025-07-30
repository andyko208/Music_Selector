# Music Selector - API Documentation

## Overview

The Music Selector is a TensorFlow-based application that determines personal music preferences by classifying audio files into three categories: Chills, Hypes, and Trips. The system uses Convolutional Neural Networks (CNN) to analyze MFCC (Mel-frequency cepstral coefficients) features extracted from audio files.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Data Processing Functions](#data-processing-functions)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation Functions](#training-and-evaluation-functions)
5. [Prediction Functions](#prediction-functions)
6. [Usage Examples](#usage-examples)
7. [Dependencies](#dependencies)

## Project Structure

```
Music_Selector/
├── Music_Selector.ipynb          # Main Jupyter notebook with all functions
├── README.md                     # Project overview
├── API_DOCUMENTATION.md          # This documentation file
└── .gitignore                    # Git ignore file
```

## Data Processing Functions

### `preprocess_data(Training_or_Test)`

**Purpose**: Processes raw WAV audio files and extracts MFCC features for training or testing.

**Parameters**:
- `Training_or_Test` (str): Either 'Training' or 'Test' to specify the dataset type

**Returns**: None (saves data to JSON file)

**Description**: 
This function processes audio files from the specified directory structure:
- `Training/` or `Test/` directory containing subdirectories for each music type
- Subdirectories: `Chills/`, `Hypes/`, `Trips/`
- Each subdirectory contains `.wav` files

**Process**:
1. Iterates through each music type directory
2. Loads each WAV file using librosa
3. Segments audio into 20-second chunks
4. Extracts MFCC features from each chunk
5. Assigns labels: 0 (Chills), 1 (Hypes), 2 (Trips)
6. Saves all data to a JSON file: `{Training_or_Test}_data.json`

**Example Usage**:
```python
# Process training data
preprocess_data('Training')

# Process test data
preprocess_data('Test')
```

**Output Format**:
```json
{
    "labels": [0, 0, 1, 1, 2, 2, ...],
    "mfccs": [[[mfcc_features]], [[mfcc_features]], ...]
}
```

### `check_data(Training_or_Test)`

**Purpose**: Loads and visualizes processed data to verify the preprocessing results.

**Parameters**:
- `Training_or_Test` (str): Either 'Training' or 'Test' to specify the dataset

**Returns**: None (displays visualizations)

**Description**:
- Loads the JSON file created by `preprocess_data()`
- Prints the number of MFCC vectors and their shape
- Displays spectrograms for sample data points from each music type

**Example Usage**:
```python
check_data('Training')
check_data('Test')
```

**Output**:
```
292 mfcc vectors with shape (862, 20).
[Displays spectrogram plots]
```

### `load_data(Training_or_Test)`

**Purpose**: Loads processed data from JSON files for model training/testing.

**Parameters**:
- `Training_or_Test` (str): Either 'Training' or 'Test' to specify the dataset

**Returns**:
- `x` (numpy.ndarray): MFCC features with shape (n_samples, 862, 20)
- `y` (numpy.ndarray): Labels with shape (n_samples,)

**Example Usage**:
```python
train_x, train_y = load_data('Training')
test_x, test_y = load_data('Test')
print(f"Training data shape: {train_x.shape}")
print(f"Training labels shape: {train_y.shape}")
```

## Model Architecture

### `design_model(input_shape)`

**Purpose**: Creates a Convolutional Neural Network (CNN) model for music classification.

**Parameters**:
- `input_shape` (tuple): Shape of input data (height, width, channels)

**Returns**:
- `model` (tf.keras.Model): Compiled CNN model

**Architecture**:
```
1. Conv2D(32, 3x3) + ReLU
2. MaxPooling2D(3x3, stride=2) + BatchNormalization
3. Conv2D(32, 3x3) + ReLU
4. MaxPooling2D(3x3, stride=2) + BatchNormalization
5. Conv2D(32, 2x2) + ReLU
6. MaxPooling2D(3x3, stride=2) + BatchNormalization + Dropout(0.3)
7. Flatten()
8. Dense(64) + ReLU
9. Dense(10) + Softmax
```

**Example Usage**:
```python
input_shape = (862, 20, 1)  # MFCC features with channel dimension
model = design_model(input_shape)
model.summary()
```

## Training and Evaluation Functions

### `prepare_datasets(inputs, targets, split_size)`

**Purpose**: Splits data into training, validation, and test sets and prepares them for CNN input.

**Parameters**:
- `inputs` (numpy.ndarray): Input features
- `targets` (numpy.ndarray): Target labels
- `split_size` (float): Fraction of data to use for validation/test (e.g., 0.2)

**Returns**:
- `inputs_train, inputs_val, inputs_test`: Training, validation, and test features
- `targets_train, targets_val, targets_test`: Training, validation, and test labels

**Description**:
- Performs two train-test splits to create validation and test sets
- Adds channel dimension to inputs for CNN compatibility
- Returns 6 arrays: training, validation, and test data for both inputs and targets

**Example Usage**:
```python
Xtrain, Xval, Xtest, ytrain, yval, ytest = prepare_datasets(train_x, train_y, 0.2)
print(f"Training set: {Xtrain.shape}")
print(f"Validation set: {Xval.shape}")
print(f"Test set: {Xtest.shape}")
```

### `plot_performance(hist)`

**Purpose**: Visualizes training and validation performance metrics.

**Parameters**:
- `hist` (tf.keras.callbacks.History): Training history object from model.fit()

**Returns**: None (displays plots)

**Description**:
Creates two plots:
1. Training and validation accuracy over epochs
2. Training and validation loss over epochs

**Example Usage**:
```python
history = model.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=50)
plot_performance(history)
```

## Prediction Functions

### `make_prediction(model, X, y, idx)`

**Purpose**: Makes predictions on individual samples and compares with ground truth.

**Parameters**:
- `model` (tf.keras.Model): Trained model
- `X` (numpy.ndarray): Input features
- `y` (numpy.ndarray): Ground truth labels
- `idx` (int): Index of the sample to predict

**Returns**:
- `int`: 1 if prediction is correct, 0 otherwise

**Description**:
- Makes prediction on the specified sample
- Maps numeric predictions to music type names
- Prints prediction result and ground truth
- Returns binary accuracy indicator

**Music Type Mapping**:
- 0: "Chills"
- 1: "Hypes" 
- 2: "Trips"

**Example Usage**:
```python
# Test individual predictions
correct = make_prediction(model, Xtest, ytest, 0)

# Test multiple predictions
nums_correct = 0
for i in range(10):
    nums_correct += make_prediction(model, Xtest, ytest, i)
print(f"Accuracy: {nums_correct}/10")
```

## Usage Examples

### Complete Workflow Example

```python
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Preprocess data
print("Processing training data...")
preprocess_data('Training')
print("Processing test data...")
preprocess_data('Test')

# 2. Load data
print("Loading data...")
train_x, train_y = load_data('Training')
test_x, test_y = load_data('Test')

# 3. Prepare datasets
print("Preparing datasets...")
Xtrain, Xval, Xtest, ytrain, yval, ytest = prepare_datasets(train_x, train_y, 0.2)

# 4. Build model
print("Building model...")
input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)
model = design_model(input_shape)

# 5. Compile model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

# 6. Train model
print("Training model...")
history = model.fit(
    Xtrain, ytrain,
    validation_data=(Xval, yval),
    epochs=50,
    batch_size=16
)

# 7. Plot performance
plot_performance(history)

# 8. Evaluate on test set
print("Evaluating model...")
test_loss, test_acc = model.evaluate(Xtest, ytest)
print(f"Test accuracy: {test_acc:.4f}")

# 9. Make predictions
print("Making predictions...")
nums_correct = 0
for i in range(10):
    nums_correct += make_prediction(model, Xtest, ytest, i)
print(f"Sample prediction accuracy: {nums_correct}/10")
```

### Data Structure Requirements

**Directory Structure**:
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

**Audio File Requirements**:
- Format: WAV files
- Naming convention: `{type}_{number}.wav` (e.g., `chill_1.wav`, `hype_2.wav`)
- Duration: Variable length (will be segmented into 20-second chunks)
- Sample rate: Automatically handled by librosa

## Dependencies

### Required Python Packages

```python
# Core dependencies
import os
import pickle as pkl
import json
import warnings

# Audio processing
import librosa
import librosa.display

# Data manipulation and visualization
import numpy as np
import matplotlib.pyplot as plt

# Machine learning
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Google Colab (if using Colab environment)
from google.colab import drive
```

### Installation

```bash
pip install tensorflow librosa matplotlib numpy scikit-learn
```

### Environment Setup

For Google Colab:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set home directory
HOME_DIR = '/content/drive/MyDrive/Music_Selector/'
```

For local environment:
```python
# Set home directory to current working directory
HOME_DIR = './'
```

## Performance Metrics

The model typically achieves:
- Training accuracy: ~95-100%
- Validation accuracy: ~60-70%
- Test accuracy: ~60-70%

**Note**: The model shows signs of overfitting, which is common with small datasets. Consider:
- Data augmentation
- Regularization techniques
- Collecting more training data
- Using pre-trained models

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure audio files are in the correct directory structure
2. **Memory errors**: Reduce batch size or use smaller audio segments
3. **Poor performance**: Check audio file quality and ensure proper labeling
4. **Import errors**: Verify all dependencies are installed

### Debugging Tips

- Use `check_data()` to verify preprocessing results
- Monitor training curves with `plot_performance()`
- Test individual predictions with `make_prediction()`
- Check data shapes at each step of the pipeline

## License

This project is open source. Please refer to the original repository for licensing information.