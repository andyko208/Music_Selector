# Music Selector - Function Guide

## What This App Does

The Music Selector helps you automatically sort your music into three mood categories: Chills, Hypes, and Trips. It learns from your music files and can predict which category new songs belong to.

## Table of Contents

1. [Project Files](#project-files)
2. [Music Processing Functions](#music-processing-functions)
3. [Learning System](#learning-system)
4. [Training and Testing Functions](#training-and-testing-functions)
5. [Prediction Functions](#prediction-functions)
6. [Usage Examples](#usage-examples)
7. [What You Need](#what-you-need)

## Project Files

```
Music_Selector/
├── Music_Selector.ipynb          # Main notebook with all functions
├── music_selector.py             # Clean version of the code
├── API_DOCUMENTATION.md          # This guide
├── QUICK_START.md               # Simple getting started guide
├── requirements.txt             # List of needed software
├── README.md                    # Project overview
└── .gitignore                   # Git settings
```

## Music Processing Functions

### `preprocess_data(Training_or_Test)`

**What it does**: Takes your music files and prepares them for the learning system.

**What you give it**:
- `Training_or_Test` (text): Either 'Training' or 'Test' to tell it which music to process

**What it does**:
1. Looks through your music folders
2. Reads each WAV file
3. Cuts each song into 20-second pieces
4. Analyzes the sound patterns in each piece
5. Saves all the information in a file

**How to use it**:
```python
# Process music for learning
preprocess_data('Training')

# Process music for testing
preprocess_data('Test')
```

**What it creates**:
- A file called `Training_data.json` or `Test_data.json` with all the music information

### `check_data(Training_or_Test)`

**What it does**: Shows you what your processed music looks like.

**What you give it**:
- `Training_or_Test` (text): Either 'Training' or 'Test'

**What it shows**:
- How many music pieces you have
- Pictures of the sound patterns from different music types

**How to use it**:
```python
check_data('Training')
check_data('Test')
```

**What you'll see**:
```
292 music pieces with shape (862, 20).
[Shows pictures of sound patterns]
```

### `load_data(Training_or_Test)`

**What it does**: Loads your processed music data for the learning system.

**What you give it**:
- `Training_or_Test` (text): Either 'Training' or 'Test'

**What you get back**:
- `x`: The sound pattern data
- `y`: The labels (0=Chills, 1=Hypes, 2=Trips)

**How to use it**:
```python
train_x, train_y = load_data('Training')
test_x, test_y = load_data('Test')
print(f"Training data shape: {train_x.shape}")
print(f"Training labels shape: {train_y.shape}")
```

## Learning System

### `design_model(input_shape)`

**What it does**: Creates the learning system (called a "neural network") that will study your music.

**What you give it**:
- `input_shape` (numbers): The size of your music data

**What you get back**:
- A learning system ready to study your music

**How the learning system works**:
1. Looks at sound patterns in your music
2. Learns what makes "chill" music different from "hype" music
3. Gets better at sorting as it sees more examples

**How to use it**:
```python
input_shape = (862, 20, 1)  # Size of your music data
model = design_model(input_shape)
model.summary()
```

## Training and Testing Functions

### `prepare_datasets(inputs, targets, split_size)`

**What it does**: Organizes your music data for learning and testing.

**What you give it**:
- `inputs`: Your music data
- `targets`: The labels (Chills/Hypes/Trips)
- `split_size` (number): How much to use for testing (like 0.2 for 20%)

**What you get back**:
- Six sets of data: training, validation, and test data for both music and labels

**What it does**:
- Splits your music into learning and testing parts
- Prepares the data in the right format for the learning system

**How to use it**:
```python
Xtrain, Xval, Xtest, ytrain, yval, ytest = prepare_datasets(train_x, train_y, 0.2)
print(f"Training set: {Xtrain.shape}")
print(f"Validation set: {Xval.shape}")
print(f"Test set: {Xtest.shape}")
```

### `plot_performance(hist)`

**What it does**: Shows you charts of how well the learning system is working.

**What you give it**:
- `hist`: The learning history from training

**What it shows**:
1. How well the system is learning over time
2. How well it's doing on new music it hasn't seen before

**How to use it**:
```python
history = model.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=50)
plot_performance(history)
```

## Prediction Functions

### `make_prediction(model, X, y, idx)`

**What it does**: Tests the learning system on one specific song.

**What you give it**:
- `model`: Your trained learning system
- `X`: Music data
- `y`: Correct labels
- `idx`: Which song to test

**What you get back**:
- 1 if the prediction was correct, 0 if it was wrong

**What it does**:
- Makes a prediction about which category a song belongs to
- Compares it to the correct answer
- Shows you the result

**Music Categories**:
- 0: "Chills" (relaxing music)
- 1: "Hypes" (energetic music) 
- 2: "Trips" (experimental music)

**How to use it**:
```python
# Test one song
correct = make_prediction(model, Xtest, ytest, 0)

# Test multiple songs
nums_correct = 0
for i in range(10):
    nums_correct += make_prediction(model, Xtest, ytest, i)
print(f"Got {nums_correct} out of 10 correct")
```

## Usage Examples

### Complete Example

```python
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Process your music
print("Processing training music...")
preprocess_data('Training')
print("Processing test music...")
preprocess_data('Test')

# 2. Load your music
print("Loading music...")
train_x, train_y = load_data('Training')
test_x, test_y = load_data('Test')

# 3. Organize for learning
print("Organizing data...")
Xtrain, Xval, Xtest, ytrain, yval, ytest = prepare_datasets(train_x, train_y, 0.2)

# 4. Create learning system
print("Creating learning system...")
input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)
model = design_model(input_shape)

# 5. Set up learning
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

# 6. Train the system
print("Training...")
history = model.fit(
    Xtrain, ytrain,
    validation_data=(Xval, yval),
    epochs=50,
    batch_size=16
)

# 7. Show results
plot_performance(history)

# 8. Test on new music
print("Testing...")
test_loss, test_acc = model.evaluate(Xtest, ytest)
print(f"Test accuracy: {test_acc:.4f}")

# 9. Test individual songs
print("Testing individual songs...")
nums_correct = 0
for i in range(10):
    nums_correct += make_prediction(model, Xtest, ytest, i)
print(f"Got {nums_correct} out of 10 songs correct")
```

### How to Organize Your Music

**Folder Structure**:
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

**Music File Requirements**:
- **Format**: WAV files only
- **Names**: Use format like `chill_1.wav`, `hype_2.wav`
- **Length**: Any length (will be cut into 20-second pieces)
- **Quality**: Better quality = better results

## What You Need

### Required Software

```python
# Core programs
import os
import pickle as pkl
import json
import warnings

# Music processing
import librosa
import librosa.display

# Data and charts
import numpy as np
import matplotlib.pyplot as plt

# Learning system
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Google Colab (if using Colab)
from google.colab import drive
```

### How to Install

```bash
pip install tensorflow librosa matplotlib numpy scikit-learn
```

### Setting Up Your Environment

**For Google Colab**:
```python
# Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set your music folder
HOME_DIR = '/content/drive/MyDrive/Music_Selector/'
```

**For your computer**:
```python
# Set music folder to current directory
HOME_DIR = './'
```

## How Well It Works

The learning system typically achieves:
- **Learning accuracy**: 95-100% (how well it learns from your music)
- **Test accuracy**: 60-70% (how well it sorts new music it hasn't seen before)

**Note**: Results depend on:
- How much music you provide
- How good the music quality is
- How clearly different the music categories are

## Common Problems

### Problems You Might Have

1. **"File not found"**: Check your folder structure and file names
2. **"Not enough memory"**: Use smaller batch sizes or shorter music pieces
3. **"Bad results"**: Check music quality and make sure labels are correct
4. **"Can't import"**: Make sure all software is installed

### Tips for Better Results

- Use `check_data()` to see your processed music
- Watch learning progress with `plot_performance()`
- Test individual songs with `make_prediction()`
- Check data shapes at each step

## Getting Help

- Check the [Quick Start Guide](QUICK_START.md) for simple instructions
- Look at the [README](README.md) for project overview
- Read the [Clean Code](music_selector.py) for detailed examples

## Legal Information

This project is open source. Check the original repository for license details.