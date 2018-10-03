from keras import layers
from keras import Input
from keras.models import Model
from keras.utils import plot_model


def multi_model():
    # Text features block
    input_text = Input(shape=(None, ), name='Input_text')
    embedded_text = layers.Embedding(128, 10000)(input_text)
    encoded_text = layers.LSTM(32)(embedded_text)

    # Digit features block
    input_features = Input(shape=(50, ), name='Input_features')
    x = layers.Dense(512, activation='relu')(input_features)
    digit = layers.Dense(256, activation='relu')(x)

    # Image features block
    input_image = Input(shape=(224, 224, 3), name='Input_image')
    y = layers.Conv2D(64, (3, 3), activation='relu')(input_image)
    y = layers.MaxPooling2D(2, 2)(y)
    y = layers.Conv2D(64, (3, 3), activation='relu')(y)
    image = layers.Flatten()(y)

    # Merge them all
    concat = layers.concatenate([encoded_text, digit, image], axis=-1)
    final = layers.Dense(512, activation='relu')(concat)
    predictions_a = layers.Dense(10, name='Classifacation', activation='softmax')(final)
    model = Model(inputs=[input_text, input_features, input_image], outputs=predictions_a)
    return model


model = multi_model()
model.summary()
plot_model(model, show_shapes=True, to_file='{}.png'.format("Keras Multi Input"))
