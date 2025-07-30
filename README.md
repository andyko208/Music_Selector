# Music Selector

A simple app that automatically sorts your music into three mood categories: **Chills**, **Hypes**, and **Trips**. It learns from your music files and can predict which category new songs belong to.

## 🎵 What It Does

- **Sorts Music**: Automatically puts songs into mood-based categories
- **Learns Patterns**: Uses AI to understand what makes each category unique
- **Easy to Use**: Just point it to your music files and let it work
- **Shows Results**: Creates charts to see how well it's learning
- **Flexible**: Works with your own music collection

## 📁 Files in This Project

```
Music_Selector/
├── Music_Selector.ipynb          # Main notebook with all the code
├── music_selector.py             # Clean version of the code
├── API_DOCUMENTATION.md          # Detailed instructions
├── QUICK_START.md               # Simple getting started guide
├── requirements.txt             # List of needed software
├── README.md                    # This file
└── .gitignore                   # Git settings
```

## 🚀 Get Started Quickly

### Step 1: Install Required Software

1. **Download the project**:
   ```bash
   git clone <repository-url>
   cd Music_Selector
   ```

2. **Install the needed programs**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Organize your music** (see [Quick Start Guide](QUICK_START.md) for details)

4. **Run the app**:
   ```python
   import music_selector as ms
   model, history, test_acc = ms.complete_workflow()
   ```

## 📚 Help and Instructions

- **[Quick Start Guide](QUICK_START.md)** - Get running in 10 minutes
- **[Detailed Instructions](API_DOCUMENTATION.md)** - Complete function guide
- **[Clean Code](music_selector.py)** - Organized, easy-to-read code

## 🎯 Music Categories

The app sorts music into three types:

| Category | Number | What It Means |
|----------|--------|---------------|
| **Chills** | 0 | Relaxing, calm, peaceful music |
| **Hypes** | 1 | Energetic, exciting, upbeat music |
| **Trips** | 2 | Weird, experimental, mind-bending music |

## 🔧 How It Works

1. **Reads Music**: Takes your WAV music files and breaks them into 20-second pieces
2. **Analyzes Sound**: Looks at the sound patterns in each piece
3. **Learns**: Studies your music to understand what makes each category different
4. **Predicts**: Can guess which category new songs belong to

### The Learning System

The app uses a type of AI called a "neural network" that:
- Looks at sound patterns in your music
- Learns what makes "chill" music different from "hype" music
- Gets better at sorting as it sees more examples

## 📊 How Well It Works

Typical results:
- **Learning Accuracy**: 95-100% (how well it learns from your music)
- **Test Accuracy**: 60-70% (how well it sorts new music it hasn't seen before)

*Note: Results depend on how much music you give it and how good the quality is*

## 🛠️ How to Use It

### Simple Way

```python
import music_selector as ms

# Run everything at once
model, history, test_acc = ms.complete_workflow()
```

### Custom Way

```python
import music_selector as ms

# Prepare your music
ms.preprocess_data('Training')
ms.preprocess_data('Test')

# Load your music data
train_x, train_y = ms.load_data('Training')
test_x, test_y = ms.load_data('Test')
Xtrain, Xval, Xtest, ytrain, yval, ytest = ms.prepare_datasets(train_x, train_y, 0.2)

# Train with your own settings
model, history = ms.train_model(
    Xtrain, ytrain, Xval, yval,
    epochs=100,           # How long to learn
    batch_size=32,        # How many songs to look at at once
    learning_rate=0.0001  # How fast to learn
)

# See how well it worked
test_acc, sample_acc = ms.evaluate_model(model, Xtest, ytest)
```

### Test Individual Songs

```python
# Test one specific song
correct = ms.make_prediction(model, Xtest, ytest, 0)
```

## 📋 What You Need

### Software Requirements

- **Python** 3.7 or newer
- **TensorFlow** (for the AI learning)
- **librosa** (for reading music files)
- **numpy** (for number calculations)
- **matplotlib** (for making charts)

### Music File Requirements

- **Format**: WAV files only
- **Names**: Use format like `chill_1.wav`, `hype_2.wav`
- **Length**: Any length (app cuts them into 20-second pieces)
- **Quality**: Better quality = better results

## 🗂️ How to Organize Your Music

Put your music files in folders like this:

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

## 🔍 Common Problems and Solutions

### Problems You Might Have

1. **"Can't find librosa"**: Run `pip install librosa`
2. **"File not found"**: Check that your folders and file names are correct
3. **"Not enough memory"**: Use smaller batch sizes or shorter music pieces
4. **"Bad results"**: Make sure your music files are good quality and properly labeled

For more help, see the [Quick Start Guide](QUICK_START.md).

## 🤝 Want to Help Improve This?

1. Copy the project
2. Make your changes
3. Test that everything still works
4. Send us your improvements

## 📄 Legal Stuff

This project is open source. Check the original repository for license details.

## 🙏 Thanks

Thanks to the people who made:
- **librosa**: For reading music files
- **TensorFlow**: For the AI learning
- **scikit-learn**: For data organization

## 📞 Need Help?

If you're stuck:
- Check the [Detailed Instructions](API_DOCUMENTATION.md)
- Look at the [Quick Start Guide](QUICK_START.md)
- Read the [Clean Code](music_selector.py) for examples

---

**Enjoy sorting your music! 🎵🎶**
