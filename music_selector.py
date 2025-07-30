"""
Music Selector - TensorFlow Application for Music Preference Classification

This module provides functions for processing audio files, training a CNN model,
and making predictions to classify music into three categories: Chills, Hypes, and Trips.

Author: Music Selector Team
License: Open Source
"""

import os
import pickle as pkl
import librosa
import json
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Global configuration
HOME_DIR = './'  # Can be overridden for different environments


def set_home_directory(directory_path):
    """
    Set the home directory for the Music Selector application.
    
    Parameters:
        directory_path (str): Path to the home directory containing Training/ and Test/ folders
    """
    global HOME_DIR
    HOME_DIR = directory_path


def preprocess_data(training_or_test):
    """
    Process raw WAV audio files and extract MFCC features for training or testing.
    
    This function processes audio files from the specified directory structure:
    - Training/ or Test/ directory containing subdirectories for each music type
    - Subdirectories: Chills/, Hypes/, Trips/
    - Each subdirectory contains .wav files
    
    Process:
    1. Iterates through each music type directory
    2. Loads each WAV file using librosa
    3. Segments audio into 20-second chunks
    4. Extracts MFCC features from each chunk
    5. Assigns labels: 0 (Chills), 1 (Hypes), 2 (Trips)
    6. Saves all data to a JSON file: {training_or_test}_data.json
    
    Parameters:
        training_or_test (str): Either 'Training' or 'Test' to specify the dataset type
        
    Returns:
        None (saves data to JSON file)
        
    Example:
        >>> preprocess_data('Training')
        >>> preprocess_data('Test')
    """
    training_or_test_dir = os.path.join(HOME_DIR, training_or_test)
    
    # What json file will contain, MFCC of each .wav data and labels of each
    mydict = {
        "labels": [],
        "mfccs": []
    }

    # music_type = Chills, Hypes, Trips
    for music_type in os.listdir(training_or_test_dir):
        # music_type_dir = '/content/drive/MyDrive/Music_Selector/Training/Chills'
        music_type_dir = os.path.join(training_or_test_dir, music_type)
        
        # Number of music vectors created from music_type_dir
        total_num_data_files = 0
        
        # checking if it is a directory
        if os.path.isdir(music_type_dir):
            # music = chill_1.wav
            for music in os.listdir(music_type_dir):
                # music_file = '/content/drive/MyDrive/Music_Selector/Training/Chills/chill_1.wav'
                music_file = os.path.join(music_type_dir, music)

                if music_file.find('Chills') > -1:
                    label = 0
                elif music_file.find('Hypes') > -1:
                    label = 1
                elif music_file.find('Trips') > -1:
                    label = 2
                
                # Number of music vectors created from music_file
                num_data_files = 0

                # checking if it is a file
                if os.path.isfile(music_file):
                    print('Processing: ' + music, end='')
                    music_samples, sr = librosa.load(music_file)
                    
                    # Have each music data vector have same length (20seconds of music data)
                    partial_samples = 20 * sr
                    
                    # increment by 20 seconds of music each time
                    for curr_sample in range(0, len(music_samples)-partial_samples, partial_samples):
                        mfcc = librosa.feature.mfcc(music_samples[curr_sample:curr_sample+partial_samples], sr=sr)
                        num_data_files += 1
                        total_num_data_files += 1

                        mydict["labels"].append(label)
                        mydict["mfccs"].append(mfcc.T.tolist())
                        
                    print(' --- {} music data vectors are created.'.format(num_data_files))
            print('--- Total of {} music data vectors are created for the music type: {}(0: Chills, 1: Hypes, 2: Trips) ---'.format(total_num_data_files, label))
        
    # Write the dictionary in a json file.
    json_path = os.path.join(training_or_test_dir, training_or_test + '_data.json')
    with open(json_path, 'w') as f:
        json.dump(mydict, f)
    f.close()


def check_data(training_or_test):
    """
    Load and visualize processed data to verify the preprocessing results.
    
    Parameters:
        training_or_test (str): Either 'Training' or 'Test' to specify the dataset
        
    Returns:
        None (displays visualizations)
        
    Example:
        >>> check_data('Training')
        >>> check_data('Test')
    """
    json_path = training_or_test + '_' + 'data.json'
    json_path = os.path.join(HOME_DIR, training_or_test, json_path)
    f = open(json_path)
    # returns JSON object as a dictionary
    data = json.load(f)
    print('{} mfcc vectors with shape {}.'.format(len(data['mfccs']), np.array(data['mfccs'][0]).shape))
    for i in range(0, len(data['labels']), int(len(data['labels'])/3)):
        plt.figure()
        librosa.display.specshow(np.array(data['mfccs'][i]).T, sr=22050, x_axis='time')
    f.close()


def load_data(training_or_test):
    """
    Load processed data from JSON files for model training/testing.
    
    Parameters:
        training_or_test (str): Either 'Training' or 'Test' to specify the dataset
        
    Returns:
        tuple: (x, y) where:
            - x (numpy.ndarray): MFCC features with shape (n_samples, 862, 20)
            - y (numpy.ndarray): Labels with shape (n_samples,)
            
    Example:
        >>> train_x, train_y = load_data('Training')
        >>> test_x, test_y = load_data('Test')
    """
    json_path = training_or_test + '_' + 'data.json'
    json_path = os.path.join(HOME_DIR, training_or_test, json_path)
    f = open(json_path)
    # returns JSON object as a dictionary
    data = json.load(f)
    f.close()

    x = np.array(data["mfccs"])
    y = np.array(data["labels"])

    return x, y


def prepare_datasets(inputs, targets, split_size):
    """
    Split data into training, validation, and test sets and prepare them for CNN input.
    
    Parameters:
        inputs (numpy.ndarray): Input features
        targets (numpy.ndarray): Target labels
        split_size (float): Fraction of data to use for validation/test (e.g., 0.2)
        
    Returns:
        tuple: (inputs_train, inputs_val, inputs_test, targets_train, targets_val, targets_test)
        
    Description:
        - Performs two train-test splits to create validation and test sets
        - Adds channel dimension to inputs for CNN compatibility
        - Returns 6 arrays: training, validation, and test data for both inputs and targets
        
    Example:
        >>> Xtrain, Xval, Xtest, ytrain, yval, ytest = prepare_datasets(train_x, train_y, 0.2)
    """
    # Creating a validation set and a test set.
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=split_size)
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs_train, targets_train, test_size=split_size)
    
    # Our CNN model expects 3D input shape.
    inputs_train = inputs_train[..., np.newaxis]
    inputs_val = inputs_val[..., np.newaxis]
    inputs_test = inputs_test[..., np.newaxis]
    
    return inputs_train, inputs_val, inputs_test, targets_train, targets_val, targets_test


def design_model(input_shape):
    """
    Create a Convolutional Neural Network (CNN) model for music classification.
    
    Architecture:
    1. Conv2D(32, 3x3) + ReLU
    2. MaxPooling2D(3x3, stride=2) + BatchNormalization
    3. Conv2D(32, 3x3) + ReLU
    4. MaxPooling2D(3x3, stride=2) + BatchNormalization
    5. Conv2D(32, 2x2) + ReLU
    6. MaxPooling2D(3x3, stride=2) + BatchNormalization + Dropout(0.3)
    7. Flatten()
    8. Dense(64) + ReLU
    9. Dense(10) + Softmax
    
    Parameters:
        input_shape (tuple): Shape of input data (height, width, channels)
        
    Returns:
        tf.keras.Model: Compiled CNN model
        
    Example:
        >>> input_shape = (862, 20, 1)
        >>> model = design_model(input_shape)
        >>> model.summary()
    """
    # Let's design the model architecture.
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
        tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model


def plot_performance(hist):
    """
    Visualize training and validation performance metrics.
    
    Creates two plots:
    1. Training and validation accuracy over epochs
    2. Training and validation loss over epochs
    
    Parameters:
        hist (tf.keras.callbacks.History): Training history object from model.fit()
        
    Returns:
        None (displays plots)
        
    Example:
        >>> history = model.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=50)
        >>> plot_performance(history)
    """
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def make_prediction(model, X, y, idx):
    """
    Make predictions on individual samples and compare with ground truth.
    
    Parameters:
        model (tf.keras.Model): Trained model
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Ground truth labels
        idx (int): Index of the sample to predict
        
    Returns:
        int: 1 if prediction is correct, 0 otherwise
        
    Music Type Mapping:
        - 0: "Chills"
        - 1: "Hypes" 
        - 2: "Trips"
        
    Example:
        >>> correct = make_prediction(model, Xtest, ytest, 0)
        >>> nums_correct = 0
        >>> for i in range(10):
        >>>     nums_correct += make_prediction(model, Xtest, ytest, i)
        >>> print(f"Accuracy: {nums_correct}/10")
    """
    genre_dict = {
        0: "Chills",
        1: "Hypes",
        2: "Trips",
    }
        
    predictions = model.predict(X)
    genre = np.argmax(predictions[idx])
    
    print("\n---Now testing the model for one audio file---\nThe model predicts: {}, and ground truth is: {}.\n".format(genre_dict[genre], genre_dict[y[idx]]))
    if genre_dict[genre] == genre_dict[y[idx]]:
        return 1
    return 0


def train_model(Xtrain, ytrain, Xval, yval, epochs=50, batch_size=16, learning_rate=0.001):
    """
    Train the music classification model.
    
    Parameters:
        Xtrain (numpy.ndarray): Training features
        ytrain (numpy.ndarray): Training labels
        Xval (numpy.ndarray): Validation features
        yval (numpy.ndarray): Validation labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        tuple: (model, history) where:
            - model (tf.keras.Model): Trained model
            - history (tf.keras.callbacks.History): Training history
            
    Example:
        >>> model, history = train_model(Xtrain, ytrain, Xval, yval)
    """
    # Build model
    input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)
    model = design_model(input_shape)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )
    
    # Train model
    history = model.fit(
        Xtrain, ytrain,
        validation_data=(Xval, yval),
        epochs=epochs,
        batch_size=batch_size
    )
    
    return model, history


def evaluate_model(model, Xtest, ytest, num_samples=10):
    """
    Evaluate the trained model on test data.
    
    Parameters:
        model (tf.keras.Model): Trained model
        Xtest (numpy.ndarray): Test features
        ytest (numpy.ndarray): Test labels
        num_samples (int): Number of sample predictions to test
        
    Returns:
        tuple: (test_accuracy, sample_accuracy) where:
            - test_accuracy (float): Overall test accuracy
            - sample_accuracy (int): Number of correct predictions in sample
            
    Example:
        >>> test_acc, sample_acc = evaluate_model(model, Xtest, ytest)
        >>> print(f"Test accuracy: {test_acc:.4f}")
        >>> print(f"Sample accuracy: {sample_acc}/10")
    """
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(Xtest, ytest)
    
    # Test sample predictions
    nums_correct = 0
    for i in range(num_samples):
        nums_correct += make_prediction(model, Xtest, ytest, i)
    
    return test_acc, nums_correct


def complete_workflow():
    """
    Run the complete music classification workflow.
    
    This function demonstrates the entire pipeline from data preprocessing
    to model training and evaluation.
    
    Returns:
        tuple: (model, history, test_accuracy) where:
            - model (tf.keras.Model): Trained model
            - history (tf.keras.callbacks.History): Training history
            - test_accuracy (float): Test accuracy
            
    Example:
        >>> model, history, test_acc = complete_workflow()
    """
    print("=== Music Selector Complete Workflow ===")
    
    # 1. Preprocess data
    print("\n1. Processing training data...")
    preprocess_data('Training')
    print("Processing test data...")
    preprocess_data('Test')
    
    # 2. Load data
    print("\n2. Loading data...")
    train_x, train_y = load_data('Training')
    test_x, test_y = load_data('Test')
    print(f"Training data shape: {train_x.shape}")
    print(f"Test data shape: {test_x.shape}")
    
    # 3. Prepare datasets
    print("\n3. Preparing datasets...")
    Xtrain, Xval, Xtest, ytrain, yval, ytest = prepare_datasets(train_x, train_y, 0.2)
    print(f"Training set: {Xtrain.shape}")
    print(f"Validation set: {Xval.shape}")
    print(f"Test set: {Xtest.shape}")
    
    # 4. Train model
    print("\n4. Training model...")
    model, history = train_model(Xtrain, ytrain, Xval, yval)
    
    # 5. Plot performance
    print("\n5. Plotting performance...")
    plot_performance(history)
    
    # 6. Evaluate model
    print("\n6. Evaluating model...")
    test_acc, sample_acc = evaluate_model(model, Xtest, ytest)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Sample prediction accuracy: {sample_acc}/10")
    
    print("\n=== Workflow Complete ===")
    
    return model, history, test_acc


if __name__ == "__main__":
    # Example usage when running the script directly
    print("Music Selector - TensorFlow Application")
    print("Run complete_workflow() to start the full pipeline")
    print("Or import individual functions for custom usage")