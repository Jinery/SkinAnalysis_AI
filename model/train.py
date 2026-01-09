from keras.src.layers.preprocessing.image_preprocessing.random_flip import HORIZONTAL
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D

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
    print("[1/6] Creating data pipeline...")
    train_dataset, validation_dataset, class_names, class_counts = create_data_pipeline()

    print("[2/6] Building model architecture...")
    model = build_model(class_counts)

    print("[3/6] Setting callbacks...")
    model_path, weights_path = ut.get_model_and_weights_name()
    # Callbacks for better training control.
    callbacks = [
        EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True, mode='max'), # Stop training if validation recall doesn't improve for 5 epochs.
        ModelCheckpoint(weights_path, monitor='val_recall', save_best_only=True, mode='max'), # Save best model weights during training.
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1) # Reduce learning rate when loss plateaus.
    ]

    print(f"\n[4/6] Training model for {EPOCH} epochs...")
    print("-" * 40)
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCH,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[5/6] Saving model and generating visualizations...")
    model.save(model_path) # Save entire model architecture + weights.
    print(f"Model saved to: {model_path}")

    print("\n[6/6] Painting training and validation data...")
    pnt.plot_training_history(history) # Visualize training progress.

if __name__ == "__main__":
    train_model()