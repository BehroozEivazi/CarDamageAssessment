from tensorflow.keras.applications import EfficientNetB5
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

num_classes = 2
path = 'D:/car damage detection/data1a/'
batch_size = 20

conv_base = EfficientNetB5(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

datagen = ImageDataGenerator(rescale=1. / 255,
                             rotation_range=0.2,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest'
                             )

val_gen = ImageDataGenerator(rescale=1. / 255)
train_gen = datagen.flow_from_directory(
    path + 'training',
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary'
)

validation_gen = val_gen.flow_from_directory(
    path + 'validation',
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary'
)


def non_trainable(model):
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
    return model


conv_base = non_trainable(conv_base)

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

print(model.summary())

callbacks = [
    TensorBoard(
        log_dir='./log',
        histogram_freq=1,
        embeddings_freq=1
    )
]

history = model.fit_generator(
    train_gen,
    steps_per_epoch=92,
    epochs=10,
    verbose=1,
    validation_data=validation_gen,
    shuffle=True,
    validation_steps=20,
    callbacks=callbacks
)

model.save('damageAssesmentEf.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
