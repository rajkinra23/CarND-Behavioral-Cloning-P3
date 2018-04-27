'''
Dedicated library to build our model.

We want to create a neural network that can take an image input, and correctly
predict the steering wheel angle. For a starting point, I'm using NVidias
autopilot neural network, which seems like a good starting point at the very
least, and optimistically a fully working solution.
'''

from keras.models import Sequential
from keras.backend import clear_session
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

import data_lib as dl

# Constants.
MODEL_NAME = 'model.h5'
EPOCHS = 15
NUM_SAMPLES = 30000

# The following code comes from :
# https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py
# To adapt it to use the 3 combined images, our input will be 3 x 160 x 320.
def create_model():
    # Initialize image dimensions.
    ch, row, col = 3, 160, 320

    # Build the model architechture.
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(row, col, ch),
              output_shape=(row, col, ch)))
    model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer="adam", loss="mse")

    # Return the model.
    return model


# Now, with our build model, we can train it using our test data, and save
# it as well.
def train_model():
    # Clear the cache data.
    clear_session()

    # Get the data.
    X_train, y_train = dl.get_data()

    # Create the neural network.
    network = create_model()

    # Train the network on our data, using a 0.2 validation split.
    network.fit(X_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES], validation_split=0.2, shuffle=True, epochs=EPOCHS)

    # Save the model.
    network.save(MODEL_NAME)

# Call train model in the main method.
if __name__ == '__main__':
    train_model()
