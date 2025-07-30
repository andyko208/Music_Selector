# Music Selector - Quick Start Guide

## Overview

This guide will help you get started with the Music Selector application in under 10 minutes. The application classifies music into three categories: Chills, Hypes, and Trips using a Convolutional Neural Network.

## Prerequisites

- Python 3.7 or higher
- Audio files in WAV format
- Basic understanding of Python and machine learning

## Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd Music_Selector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your audio data**:
   Create the following directory structure:
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

## Quick Start

### Option 1: Using the Python Module (Recommended)

1. **Import the module**:
   ```python
   import music_selector as ms
   ```

2. **Set your data directory** (if different from current directory):
   ```python
   ms.set_home_directory('/path/to/your/data')
   ```

3. **Run the complete workflow**:
   ```python
   model, history, test_acc = ms.complete_workflow()
   ```

### Option 2: Using Individual Functions

1. **Preprocess your data**:
   ```python
   import music_selector as ms
   
   # Process training and test data
   ms.preprocess_data('Training')
   ms.preprocess_data('Test')
   ```

2. **Load and prepare data**:
   ```python
   # Load data
   train_x, train_y = ms.load_data('Training')
   test_x, test_y = ms.load_data('Test')
   
   # Prepare datasets
   Xtrain, Xval, Xtest, ytrain, yval, ytest = ms.prepare_datasets(train_x, train_y, 0.2)
   ```

3. **Train the model**:
   ```python
   # Train model
   model, history = ms.train_model(Xtrain, ytrain, Xval, yval)
   
   # Plot performance
   ms.plot_performance(history)
   ```

4. **Evaluate the model**:
   ```python
   # Evaluate on test set
   test_acc, sample_acc = ms.evaluate_model(model, Xtest, ytest)
   print(f"Test accuracy: {test_acc:.4f}")
   ```

### Option 3: Using the Jupyter Notebook

1. **Open the notebook**:
   ```bash
   jupyter notebook Music_Selector.ipynb
   ```

2. **Follow the cells in order**:
   - Run the import and setup cells
   - Execute the data preprocessing cells
   - Run the model training cells
   - Evaluate the results

## Expected Results

After running the complete workflow, you should see:

- **Training progress**: Epoch-by-epoch training with accuracy and loss metrics
- **Performance plots**: Training and validation accuracy/loss curves
- **Test results**: Overall test accuracy and sample predictions
- **Typical performance**: 60-70% test accuracy (may vary based on data quality)

## Customization

### Adjusting Model Parameters

```python
# Custom training parameters
model, history = ms.train_model(
    Xtrain, ytrain, Xval, yval,
    epochs=100,           # More training epochs
    batch_size=32,        # Larger batch size
    learning_rate=0.0001  # Lower learning rate
)
```

### Using Different Data

```python
# Set custom directory
ms.set_home_directory('/path/to/your/custom/data')

# Process your data
ms.preprocess_data('Training')
ms.preprocess_data('Test')
```

### Saving and Loading Models

```python
# Save the trained model
model.save('music_selector_model.h5')

# Load the model later
from tensorflow import keras
loaded_model = keras.models.load_model('music_selector_model.h5')
```

## Troubleshooting

### Common Issues

1. **"No module named 'librosa'"**:
   ```bash
   pip install librosa
   ```

2. **"File not found" errors**:
   - Check your directory structure matches the requirements
   - Ensure audio files are in WAV format
   - Verify file naming convention: `{type}_{number}.wav`

3. **Memory errors**:
   - Reduce batch size: `batch_size=8`
   - Use smaller audio segments (modify the 20-second chunk size in the code)

4. **Poor performance**:
   - Ensure audio files are high quality
   - Check that labels are correctly assigned
   - Try increasing training data

### Getting Help

- Check the full [API Documentation](API_DOCUMENTATION.md)
- Review the [README.md](README.md) for project overview
- Examine the [music_selector.py](music_selector.py) module for detailed function documentation

## Next Steps

1. **Improve the model**:
   - Collect more training data
   - Experiment with different architectures
   - Try data augmentation techniques

2. **Deploy the model**:
   - Save the trained model
   - Create a prediction API
   - Build a web interface

3. **Extend functionality**:
   - Add more music categories
   - Implement real-time prediction
   - Create a music recommendation system

## Example Output

```
=== Music Selector Complete Workflow ===

1. Processing training data...
Processing: chill_1.wav --- 8 music data vectors are created.
Processing: hype_1.wav --- 9 music data vectors are created.
Processing: trip_1.wav --- 12 music data vectors are created.
...

2. Loading data...
Training data shape: (292, 862, 20)
Test data shape: (142, 862, 20)

3. Preparing datasets...
Training set: (186, 862, 20, 1)
Validation set: (47, 862, 20, 1)
Test set: (47, 862, 20, 1)

4. Training model...
Epoch 1/50
12/12 [==============================] - 10s 62ms/step - loss: 2.0942 - acc: 0.5430

5. Plotting performance...
[Displays training curves]

6. Evaluating model...
Test accuracy: 0.6809
Sample prediction accuracy: 9/10

=== Workflow Complete ===
```

Congratulations! You've successfully set up and run the Music Selector application.