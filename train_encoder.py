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
ap.add_argument('-d', '--dimensions', nargs='+', required=True,
                help='dimensions of the hidden layers in percentage of the input')
args = vars(ap.parse_args())
args['dimensions'] = list(map(float, args['dimensions']))


# Loading the data
(data, ports_list) = dataloader.load_data("BENIGN", path=args['path'], rows=1500000).values()

# Splitting test and train data
x_train, x_test = train_test_split(data, test_size=0.3)

print("Data loaded. Training set : %s, Testing set : %s." % (x_train.shape, x_test.shape))

# Defining the dimensions of the layers
input_dim = data.shape[1]
encoding_dim = args['dimensions'][-1]
encoding_dim = int(encoding_dim * input_dim)
layers_dimensions = args['dimensions'][:-1]
layers_dimensions = [int(i * input_dim) for i in layers_dimensions]


# Defining the network layers
input_data = Input(shape=(input_dim,))
previous_layer = input_data

for dim in layers_dimensions:
    new_layer = Dense(dim, activation='relu')(previous_layer)
    previous_layer = new_layer

encoded = Dense(encoding_dim, activation='relu')(previous_layer)
previous_layer = encoded

for dim in layers_dimensions[::-1]:     # list of dimensions in reversed order
    new_layer = Dense(dim, activation='relu')(previous_layer)
    previous_layer = new_layer

decoded = Dense(input_dim, activation='relu')(previous_layer)

# Mapping input and output
encoder = Model(input_data, encoded)
autoencoder = Model(input_data, decoded)


# Compiling the autoencoder
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# Training the autoencoder on the benign data
h = autoencoder.fit(x_train, x_train,
                    epochs=20,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(x_test, x_test)
                    )

print("Model fitted, saving to file.")
autoencoder.save(args["model"])

with open('ports_list3.json', 'w') as output:
    json.dump(ports_list, output)


# summarize history for loss
plt.plot(h.history['loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
fig = plt.gcf()
fig.savefig('plots/history.png')
plt.show()
