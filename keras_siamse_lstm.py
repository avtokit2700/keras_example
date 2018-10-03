from keras import layers
from keras import Input
from keras.models import Model
from keras.utils import plot_model


def siamse_lstm():
    lstm_layer = layers.LSTM(32)

    left_input = Input(shape=(None, 128))
    left_output = lstm_layer(left_input)

    right_input = Input(shape=(None, 128))
    right_output = lstm_layer(right_input)

    merged = layers.concatenate([left_output, right_output], axis=-1)
    predictions = layers.Dense(1, activation='sigmoid')(merged)

    model = Model([left_input, right_input], predictions)
    return model


model = siamse_lstm()
model.summary()

plot_model(model, show_shapes=True, to_file='{}.png'.format("Keras Siamse LSTM"))
