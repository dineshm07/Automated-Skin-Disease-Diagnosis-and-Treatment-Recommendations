import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define paths

TRAIN_DIR = '../dataset/split_data/train'
VAL_DIR = '../dataset/split_data/validation'
IMG_SIZE = (160, 160)
BATCH_SIZE = 8
NUM_CLASSES = 7
EPOCHS = 70

# Setup GPU configuration
tf.keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[INFO] GPU memory growth enabled.")
    except Exception as e:
        print(f"[WARNING] Couldn't set memory growth: {e}")
else:
    print("[INFO] No GPU found. Using CPU.")

# Use mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Partial Oversampling with strong augmentation
sample_limits = {
    'nv': 3000,
    'mel': 1100,
    'bkl': 1000,
    'bcc': 700,
    'akiec': 500,
    'vasc': 300,
    'df': 300
}

def build_partial_oversampled_generator():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    def generate_balanced_data(directory):
        all_images = []
        all_labels = []
        class_indices = {name: idx for idx, name in enumerate(sorted(os.listdir(directory)))}

        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            images = os.listdir(class_path)
            random.shuffle(images)
            limit = sample_limits[class_name]
            sampled_images = images[:limit] if len(images) > limit else images + random.choices(images, k=limit - len(images))

            for img_name in sampled_images:
                all_images.append(os.path.join(class_path, img_name))
                all_labels.append(class_name)

        data = list(zip(all_images, all_labels))
        random.shuffle(data)
        all_images, all_labels = zip(*data)

        def generator():
            for img_path, label in zip(all_images, all_labels):
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = datagen.random_transform(img)
                img = datagen.standardize(img)
                yield img, tf.keras.utils.to_categorical(class_indices[label], num_classes=NUM_CLASSES)

        dataset = tf.data.Dataset.from_generator(generator,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=([*IMG_SIZE, 3], [NUM_CLASSES]))
        print(f"[INFO] Oversampled training images: {len(all_images)} from 7 classes.")
        return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return generate_balanced_data(TRAIN_DIR)

# Validation generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Build model with BatchNormalization and Dropout
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

# Focal loss implementation
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

# Cosine decay learning rate
lr_schedule = CosineDecayRestarts(
    initial_learning_rate=1e-4,
    first_decay_steps=10,
    t_mul=2.0,
    m_mul=0.9,
    alpha=1e-6
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=focal_loss(),
    metrics=['accuracy']
)

# Callbacks
checkpoint_cb = ModelCheckpoint(
    "best_skin_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train model
train_generator = build_partial_oversampled_generator()
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint_cb]
)

# Unfreeze and fine-tune
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=focal_loss(),
    metrics=['accuracy']
)

fine_tune_epochs = 10
history_fine = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=val_generator,
    callbacks=[checkpoint_cb]
)

model.save('mobilenetv2_model_fin2.h5')
print("Model saved.")

# Save plots
plt.figure()
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='val_acc')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_progress_fin2.png')

# Confusion matrix
val_generator.reset()
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_fin2.png')

# Classification report
target_names = list(val_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=target_names)
print(report)
