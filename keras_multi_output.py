from keras import layers
from keras import Input
from keras.models import Model
from keras.utils import plot_model


def multi_model():
    # Image features block
    input_image = Input(shape=(224, 224, 3), name='Input_image')
    y = layers.Conv2D(64, (3, 3), activation='relu')(input_image)
    y = layers.MaxPooling2D(2, 2)(y)
    y = layers.Conv2D(64, (3, 3), activation='relu')(y)
    y = layers.MaxPooling2D(2, 2)(y)
    y = layers.Conv2D(64, (3, 3), activation='relu')(y)
    y = layers.MaxPooling2D(2, 2)(y)
    y = layers.Conv2D(64, (3, 3), activation='relu')(y)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.AveragePooling2D(2, 2)(y)
    # Merge them all
    final = layers.Flatten()(y)
    age = layers.Dense(1, name='Age')(final)
    gender = layers.Dense(2, activation='softmax', name='Gender')(final)
    model = Model(inputs=input_image, outputs=[age, gender])
    return model


model = multi_model()
model.summary()
model.compile(optimizer='sgd', loss=['mse', 'binary_crossentropy'])
plot_model(model, show_shapes=True, to_file='{}.png'.format("Keras Multi Output"))
