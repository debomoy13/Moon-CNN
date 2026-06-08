import os
from glob import glob
from PIL import Image
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

classes = glob('Train/Full moon/*')
print(classes)

print(os.getcwd())

file_path='Train/Full moon'
for img_name in os.listdir(file_path):
    img=Image.open(os.path.join(file_path,img_name))
    sizes=(set(img.size))

from tensorflow.keras.preprocessing import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
    'Train',
    image_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    label_mode='int',
    shuffle=True
)

val_dataset = image_dataset_from_directory(
    'Test',
    image_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    label_mode='int',
    shuffle=False
)
import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Resizing(140, 140),

    tf.keras.layers.CenterCrop(128, 128),

    tf.keras.layers.RandomRotation(1.0),

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomBrightness(factor=0.2),

    tf.keras.layers.Rescaling(1./127.5, offset=-1)
])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 1)),
    data_augmentation,
    tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')

    ])  



model.summary()  #for model summary 

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset
)