# Music Selector - Quick Start Guide

## What This Does

This guide helps you get the Music Selector app running in under 10 minutes. The app sorts your music into three types: Chills, Hypes, and Trips.

## What You Need

- Python 3.7 or newer
- Music files in WAV format
- Basic computer skills

## Step 1: Get the App

1. **Download the project**:
   ```bash
   git clone <repository-url>
   cd Music_Selector
   ```

2. **Install the needed programs**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Organize your music** (see details below)

## Step 2: Organize Your Music

Create folders for your music like this:

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

**Important**: 
- Use WAV files only
- Name files like `chill_1.wav`, `hype_2.wav`, etc.
- Put some music in Training folders (for learning)
- Put some music in Test folders (for testing)

## Step 3: Run the App

### Option 1: Easy Way (Recommended)

1. **Open Python** and type:
   ```python
   import music_selector as ms
   ```

2. **Set your music folder** (if different from current folder):
   ```python
   ms.set_home_directory('/path/to/your/music')
   ```

3. **Run everything**:
   ```python
   model, history, test_acc = ms.complete_workflow()
   ```

### Option 2: Step by Step

1. **Prepare your music**:
   ```python
   import music_selector as ms
   
   # Process your music
   ms.preprocess_data('Training')
   ms.preprocess_data('Test')
   ```

2. **Load your music**:
   ```python
   # Load music data
   train_x, train_y = ms.load_data('Training')
   test_x, test_y = ms.load_data('Test')
   
   # Organize for learning
   Xtrain, Xval, Xtest, ytrain, yval, ytest = ms.prepare_datasets(train_x, train_y, 0.2)
   ```

3. **Train the app**:
   ```python
   # Teach the app about your music
   model, history = ms.train_model(Xtrain, ytrain, Xval, yval)
   
   # See how it's learning
   ms.plot_performance(history)
   ```

4. **Test the results**:
   ```python
   # See how well it worked
   test_acc, sample_acc = ms.evaluate_model(model, Xtest, ytest)
   print(f"Test accuracy: {test_acc:.4f}")
   ```

### Option 3: Use the Notebook

1. **Open the notebook**:
   ```bash
   jupyter notebook Music_Selector.ipynb
   ```

2. **Follow the steps** in order:
   - Run the setup code
   - Process your music
   - Train the app
   - See the results

## What You'll See

After running the app, you should see:

- **Learning progress**: Shows how well the app is learning from your music
- **Charts**: Pictures showing learning progress over time
- **Test results**: How well it sorts new music it hasn't seen before
- **Typical results**: 60-70% accuracy on new music

## Change Settings

### Different Learning Settings

```python
# Use different learning settings
model, history = ms.train_model(
    Xtrain, ytrain, Xval, yval,
    epochs=100,           # Learn longer
    batch_size=32,        # Look at more songs at once
    learning_rate=0.0001  # Learn slower
)
```

### Use Different Music

```python
# Point to different music folder
ms.set_home_directory('/path/to/your/music')

# Process your music
ms.preprocess_data('Training')
ms.preprocess_data('Test')
```

### Save Your Work

```python
# Save the trained app
model.save('my_music_sorter.h5')

# Load it later
from tensorflow import keras
my_app = keras.models.load_model('my_music_sorter.h5')
```

## Common Problems

### "Can't find librosa"
```bash
pip install librosa
```

### "File not found"
- Check your folder structure matches the example above
- Make sure music files are WAV format
- Check file names like `chill_1.wav`, `hype_2.wav`

### "Not enough memory"
- Use smaller batch size: `batch_size=8`
- Use shorter music pieces (change the 20-second setting in the code)

### "Bad results"
- Make sure your music files are good quality
- Check that you labeled them correctly
- Try adding more music to learn from

### Need More Help?

- Check the [Detailed Instructions](API_DOCUMENTATION.md)
- Look at the [README](README.md) for project overview
- Read the [Clean Code](music_selector.py) for examples

## What's Next?

1. **Make it better**:
   - Add more music to learn from
   - Try different learning settings
   - Add more music categories

2. **Use it**:
   - Save your trained app
   - Sort new music automatically
   - Build a music recommendation system

3. **Share it**:
   - Show others how to use it
   - Help improve the code
   - Share your music sorting results

## Example of What You'll See

```
=== Music Selector Complete Workflow ===

1. Processing training data...
Processing: chill_1.wav --- 8 music pieces created.
Processing: hype_1.wav --- 9 music pieces created.
Processing: trip_1.wav --- 12 music pieces created.
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
[Shows learning charts]

6. Evaluating model...
Test accuracy: 0.6809
Sample prediction accuracy: 9/10

=== Workflow Complete ===
```

**Great job! You've successfully set up and run the Music Selector app.**