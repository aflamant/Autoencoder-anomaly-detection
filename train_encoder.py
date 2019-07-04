from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import json

import dataloader


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-p", "--path", required=True,
                help="path to datasets folder")
ap.add_argument("-n", "--dimension", required=True,
                help="dimension of the hidden layer")
args = vars(ap.parse_args())


# Loading the data
(data, ports_list) = dataloader.load_data("BENIGN", rows=1000000).values()

# Splitting test and train data
x_train, x_test = train_test_split(data, test_size=0.3)

print("Data loaded. Training set : %s, Testing set : %s." % (x_train.shape, x_test.shape))

# Dimension of the hidden layer : half of the input
input_dim = data.shape[1]
encoding_dim = int(input_dim/2)


# Defining the network layers
input_data = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_data)
decoded = Dense(input_dim, activation="relu")(encoded)

# Mapping input and output
encoder = Model(input_data, encoded)
autoencoder = Model(input_data, decoded)
encoded_input_placeholder = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input_placeholder, decoder_layer(encoded_input_placeholder))


# Compiling the autoencoder
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# Training the autoencoder on the benign data
h = autoencoder.fit(x_train, x_train,
                    epochs=40,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(x_test, x_test)
                    )

print("Model fitted, saving to file.")
autoencoder.save(args["model"])

with open('ports_list.json', 'w') as output:
    json.dump(ports_list, output)


# summarize history for loss
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
