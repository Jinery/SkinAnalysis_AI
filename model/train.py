import numpy as np
from keras.src.layers.preprocessing.image_preprocessing.random_flip import HORIZONTAL
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D

from sklearn.utils.class_weight import compute_class_weight

import utils as ut
import tensorflow as tf
import paint as pnt

IMG_SIZE: tuple[int, int] = (224, 224) # Suze for every photo.
BATCH_SIZE: int = 64 # Photos in a pack.
EPOCH: int = 30 # Global learning epochs.

def create_data_pipeline():
    try:
        train_dataset_path: str = ut.get_dataset_path() # Try get dataset path.
    except FileNotFoundError:
        print("Dataset not found. Please download the dataset and start dataset_sorter.py.")
        exit(-1)

    # Load train dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode="categorical",
        labels='inferred',
    )

    #Load validation dataset.
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode="categorical",
        labels='inferred',
    )

    # Get all class names and class counts from dataset
    class_names = train_dataset.class_names
    class_counts = len(class_names)

    print("Class names:", class_names)
    print("Number of classes:", class_counts)

    # Prefetch train and validation datasets.
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, class_names, class_counts

# Function to build model architecture.
def build_model(class_counts: int):
    """
    Builds a transfer learning model based on MobileNetV2.

    Args:
        class_counts (int): Number of output classes for classification

    Returns:
        tf.keras.Model: Compiled model ready for training
    """

    # Data augmentation layer for training - helps prevent overfitting.
    # by artificially expanding the dataset with random transformations.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(HORIZONTAL), # Correct constant - random horizontal flip.
        tf.keras.layers.RandomRotation(0.15), # Random rotation up to 15%.
        tf.keras.layers.RandomZoom(0.15), # Random zoom up to 15%.
        tf.keras.layers.GaussianNoise(0.07), # Add Gaussian noise (7% standard deviation).
        tf.keras.layers.RandomBrightness(0.15), # Random brightness adjustment +-15%
        tf.keras.layers.RandomContrast(0.1),  # Random contrast adjustment +-10%.
    ])

    # Input layer - define the shape of input images.
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs) # Apply data augmentation to inputs.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # Preprocess augmented data.

    # Load MobileNetV2 as base model (pre-trained on ImageNet).
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False, # Exclude original classification layers.
        weights="imagenet", # Use pre-trained ImageNet weights.
        pooling=None # No pooling - we'll add our own.
    )

    # Freeze base model layers - transfer learning approach.
    # Pre-trained weights won't be updated during initial training.
    base_model.trainable = False

    x = base_model(x, training=False) # training=False important for BatchNorm in inference mode.
    x = GlobalAveragePooling2D()(x) # Converts feature maps to feature vectors.
    x = BatchNormalization()(x) # Stabilizes and accelerates training.
    x = tf.keras.layers.Dense(512, activation="relu")(x)   # First classification head layer.
    x = tf.keras.layers.Dropout(0.4)(x) # Dropout for regularization (40% neurons dropped).
    x = tf.keras.layers.Dense(256, activation="relu")(x) # Second classification head layer.
    x = tf.keras.layers.Dropout(0.3)(x) # Less dropout in deeper layer (30%).
    outputs = tf.keras.layers.Dense(class_counts, activation='softmax', name="prediction_output")(x) # Output layer: softmax activation for multi-class classification.

    # Create the final model.
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile model with optimizer, loss function and metrics.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy', # Standard for multi-class
        metrics=[
            'accuracy', # Overall accuracy.
            tf.keras.metrics.Precision(), # Precision: TP/(TP+FP).
            tf.keras.metrics.Recall() # Recall: TP/(TP+FN) - important for imbalanced classes.
        ]
    )

    return model

def train_model():
    """Main training pipeline for the model."""
    print("[1/8] Creating data pipeline...")
    train_dataset, validation_dataset, class_names, class_counts = create_data_pipeline()

    print("[2/8] Calculating class weights...")
    y_train = np.concatenate([y for x, y in train_dataset], axis=0)
    y_train_indicies = np.argmax(y_train, axis=1)

    classes = np.unique(y_train_indicies)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_indicies,
    )
    class_weights_dict = dict(zip(classes, weights))
    print(f"Class weights: {class_weights_dict}")

    print("[3/8] Building model architecture...")
    model = build_model(class_counts)

    print("[4/8] Setting callbacks...")
    model_path, weights_path = ut.get_model_and_weights_name()
    # Callbacks for better training control.
    callbacks = [
        EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True, mode='max'), # Stop training if validation recall doesn't improve for 5 epochs.
        ModelCheckpoint(weights_path, monitor='val_recall', save_best_only=True, mode='max'), # Save best model weights during training.
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1) # Reduce learning rate when loss plateaus.
    ]

    print(f"\n[5/8] Training model for {EPOCH} epochs...")
    print("-" * 40)
    # Train the model with initial configuration.
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,  # Dataset for validation metrics
        epochs=EPOCH,  # Number of training epochs
        callbacks=callbacks,  # Training callbacks for monitoring and control
        class_weight=class_weights_dict,  # Class weights to handle imbalance
        verbose=1  # Show progress bar and metrics
    )

    print("\n[6/8] Fine-tuning...")
    model.trainable = True # Unfreeze the base model for fine-tuning - allow pre-trained layers to adapt.
    # Recompile with lower learning rate for fine-tuning (prevovershoot).
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),  # Very low LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),  # Rename for clarity
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    # Special callbacks for fine-tuning phase.
    callbacks_fine = [
        EarlyStopping(monitor='val_recall', patience=3, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8)
    ]

    # Fine-tune the model for a few epochs with all layers trainable.
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=7,  # Short fine-tuning phase
        callbacks=callbacks_fine,
        class_weight=class_weights_dict,
        verbose=1
    )

    print("\n[7/8] Saving model and generating visualizations...")
    model.save(model_path) # Save entire model architecture + weights.
    print(f"Model saved to: {model_path}")

    print("\n[8/8] Painting training and validation data...")
    pnt.plot_training_history(history) # Visualize training progress.

if __name__ == "__main__":
    train_model()