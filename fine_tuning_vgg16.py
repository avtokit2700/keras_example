from keras import layers
from keras import Input
from keras.models import Model
from keras.utils import plot_model
from keras.applications import VGG16


# Define the model
def vgg_model():
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Make all layers trainable
    vgg16.trainable = True
    set_trainable = False

    for layer in vgg16.layers:
    	if layer.name == 'block5_conv1':
    		set_trainable = True
    	if set_trainable:
    		layer.trainable
    	else:
    		layer.trainable = False
    # Additional layer after last Pooling pretrained model
    x = layers.Flatten()(vgg16.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    # First Output
    some_classification = layers.Dense(10, name='Classification', activation='softmax')(x) 
    model = Model(vgg16.input, some_classification)
    return model

model = vgg_model()
model.summary()

plot_model(model, show_shapes=True, to_file='{}.png'.format("Keras VGG16"))
