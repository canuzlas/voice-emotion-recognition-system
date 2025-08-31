"""
Professional Voice Emotion Recognition Model Training Script
===========================================================

This script trains a Random Forest classifier for voice emotion recognition
using audio feature extraction and normalization techniques.

Features:
- 45 audio features including MFCC, spectral, and temporal characteristics
- StandardScaler normalization for consistent predictions
- 8 emotion classes: happy, sad, angry, neutral, fear, disgust, surprise, calm
- Cross-validation and comprehensive evaluation

Author: Voice Emotion Recognition Team
Version: 1.0
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import librosa
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
EMOTION_MAPPING = {
    "happy": 0, "sad": 1, "angry": 2, "neutral": 3,
    "fear": 4, "disgust": 5, "surprise": 6, "calm": 7
}

MODEL_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1,
    'max_depth': 15,
    'min_samples_split': 5
}

def extract_audio_features(file_path, sr=22050, duration=3.0):
    """
    Extract comprehensive audio features for emotion recognition.
    
    Parameters:
    -----------
    file_path : str
        Path to audio file
    sr : int
        Sample rate for audio loading
    duration : float
        Duration to load in seconds
        
    Returns:
    --------
    np.array
        45-dimensional feature vector
    """
    try:
        # Load audio with librosa
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
        
        features = []
        
        # 1. MFCC Features (26 features: 13 mean + 13 std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1).tolist())
        features.extend(np.std(mfcc, axis=1).tolist())
        
        # 2. Spectral Features (4 features)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([float(np.mean(spectral_centroids)), float(np.std(spectral_centroids))])
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff))])
        
        # 3. Zero Crossing Rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([float(np.mean(zcr)), float(np.std(zcr))])
        
        # 4. Tempo (1 feature)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo) if np.isscalar(tempo) else float(tempo[0]))
        
        # 5. Chroma Features (12 features)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1).tolist())
        
        # Ensure exactly 45 features
        features = features[:45] + [0.0] * max(0, 45 - len(features))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"⚠️  Error processing {file_path}: {e}")
        return np.zeros(45, dtype=np.float32)

def load_dataset(metadata_path, sample_ratio=1.0):
    """
    Load and preprocess the emotion dataset.
    
    Parameters:
    -----------
    metadata_path : str
        Path to metadata CSV file
    sample_ratio : float
        Ratio of data to use (for testing with smaller datasets)
        
    Returns:
    --------
    tuple
        (features, labels, metadata)
    """
    print("📂 Loading Dataset")
    print("-" * 40)
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    if sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
    
    print(f"📊 Total samples: {len(df)}")
    print(f"📋 Emotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count}")
    
    return df

def extract_features_from_dataset(df, show_progress=True):
    """Extract features from all files in dataset."""
    print("\n🔄 Feature Extraction")
    print("-" * 40)
    
    X, y = [], []
    failed_files = []
    
    for idx, row in df.iterrows():
        if show_progress and (idx + 1) % 50 == 0:
            print(f"   Processed {idx + 1}/{len(df)} files...")
        
        features = extract_audio_features(row['filepath'])
        
        if np.all(features == 0):
            failed_files.append(row['filepath'])
            continue
            
        X.append(features)
        y.append(EMOTION_MAPPING[row['emotion']])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Features extracted: {X.shape}")
    print(f"📈 Feature statistics:")
    print(f"   Mean: {X.mean():.4f}")
    print(f"   Std:  {X.std():.4f}")
    print(f"   Min:  {X.min():.4f}")
    print(f"   Max:  {X.max():.4f}")
    
    if failed_files:
        print(f"⚠️  Failed to process {len(failed_files)} files")
    
    return X, y

def train_emotion_model(X, y):
    """Train and evaluate the emotion recognition model."""
    print("\n🤖 Model Training")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔄 Training set: {X_train.shape[0]} samples")
    print(f"🔄 Test set: {X_test.shape[0]} samples")
    
    # Feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Normalized features:")
    print(f"   Train mean: {X_train_scaled.mean():.4f}")
    print(f"   Train std:  {X_train_scaled.std():.4f}")
    
    # Train model
    print("🔄 Training Random Forest...")
    model = RandomForestClassifier(**MODEL_CONFIG)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"✅ Training completed!")
    print(f"🎯 Training accuracy: {train_acc:.4f}")
    print(f"🎯 Test accuracy: {test_acc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"🔄 Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return model, scaler, test_acc, y_test, test_pred

def save_model(model, scaler, accuracy, output_dir="models"):
    """Save the trained model and scaler."""
    print(f"\n💾 Saving Model")
    print("-" * 40)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare model data
    model_data = {
        'model': model,
        'scaler': scaler,
        'emotion_mapping': EMOTION_MAPPING,
        'feature_count': 45,
        'accuracy': accuracy,
        'training_date': datetime.now().isoformat(),
        'model_config': MODEL_CONFIG,
        'version': '1.0'
    }
    
    # Save model
    model_path = output_path / "emotion_recognition_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"📊 Model info:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Features: {model_data['feature_count']}")
    print(f"   Emotions: {len(EMOTION_MAPPING)}")
    
    return model_path

def print_evaluation_report(y_test, test_pred):
    """Print detailed evaluation report."""
    print(f"\n📊 Evaluation Report")
    print("=" * 50)
    
    emotion_names = list(EMOTION_MAPPING.keys())
    
    print("Classification Report:")
    print(classification_report(y_test, test_pred, target_names=emotion_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print("Predicted ->", " ".join([f"{name[:4]:>4}" for name in emotion_names]))
    for i, name in enumerate(emotion_names):
        print(f"{name[:8]:<8} {' '.join([f'{cm[i][j]:>4}' for j in range(len(emotion_names))])}")

def main():
    """Main training pipeline."""
    print("🎙️ Voice Emotion Recognition Model Training")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    metadata_path = "data/real_datasets/RAVDESS/metadata.csv"
    sample_ratio = 0.25  # Use 25% of data for faster training
    
    try:
        # Load dataset
        df = load_dataset(metadata_path, sample_ratio=sample_ratio)
        
        # Extract features
        X, y = extract_features_from_dataset(df)
        
        if len(X) == 0:
            raise ValueError("No valid features extracted!")
        
        # Train model
        model, scaler, accuracy, y_test, test_pred = train_emotion_model(X, y)
        
        # Save model
        model_path = save_model(model, scaler, accuracy)
        
        # Print evaluation
        print_evaluation_report(y_test, test_pred)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"🎯 Final accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
