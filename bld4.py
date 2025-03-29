import tensorflow as tf
from tensorflow import keras 
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Dropout, Flatten

# Data Loading
train = keras.utils.image_dataset_from_directory(
    directory = r"C:\Users\harsh\OneDrive\Desktop\Banana LSD\bananalsd\AugmentedSet",
    label_mode = 'int',
    labels = 'inferred',
    image_size = (64, 64),
    batch_size = 32
)
validation = keras.utils.image_dataset_from_directory(
    directory = r"C:\Users\harsh\OneDrive\Desktop\Banana LSD\bananalsd\OriginalSet",
    label_mode = 'int',
    labels = 'inferred',
    image_size = (64, 64),
    batch_size = 32
)

# Model Architecture
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dropout(0.4))
model.add(Dense(26, activation='softmax'))

# Model Compilation
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Model Training
history = model.fit(train, validation_data=validation, epochs=60)

# Display Final Accuracy
final_train_accuracy = history.history['accuracy'][-1] * 100
final_val_accuracy = history.history['val_accuracy'][-1] * 100
print(f"Final Training Accuracy: {final_train_accuracy:.2f}%")
print(f"Final Validation Accuracy: {final_val_accuracy:.2f}%")

# Save Model
model.save("Trained_Model.keras")
