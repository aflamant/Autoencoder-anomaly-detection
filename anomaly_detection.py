import argparse
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc

import dataloader

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to model")
ap.add_argument("-p", "--path", required=True,
                help="path to datasets folder")
ap.add_argument("-a", "--anomaly", required=True,
                help="kind of anomaly to test")
ap.add_argument("-l", "--ports", required=True,
                help="list of ports considered by the network")

args = vars(ap.parse_args())

# Loading saved date from the training
autoencoder = load_model(args["model"])
with open(args['ports'], 'r') as input_ports:
    ports_list = json.load(input_ports)


# Loading benign and anomalous data
benign_data = dataloader.load_data("BENIGN", skip=1000000, rows=8000, most_common_ports=ports_list)['data']
anomalous_data = dataloader.load_data(args["anomaly"], rows=2000, most_common_ports=ports_list)['data']

# Tagging the data to distinguish benign and anomalous traffic
tagged_benign_data = np.hstack((benign_data, np.zeros(benign_data.shape[0]).reshape(-1, 1)))            # 0 == BENIGN
tagged_anomalous_data = np.hstack((anomalous_data, np.ones(anomalous_data.shape[0]).reshape(-1, 1)))    # 1 == ANOMALOUS

# Concatenating the two sets and shuffling them
data = np.vstack((tagged_benign_data, tagged_anomalous_data))
data = np.random.permutation(data)

# Predicting values using the autoencoder and calculating the reconstruction error
prediction = autoencoder.predict(x=data[:, :-1])
reconstruction_error = np.linalg.norm(data[:, :-1] - prediction, axis=-1)
anomaly = data[:, -1]

# Plotting the reconstruction error
plt.scatter(np.arange(0,benign_data.shape[0]+anomalous_data.shape[0]), reconstruction_error, c=data[:, -1])
fig = plt.gcf()
fig.savefig('plots/' + args['anomaly'] + '/reconstruction_error.png')
plt.show()


# Plotting ROC curves

fpr, tpr, thresholds = roc_curve(anomaly, reconstruction_error)
auc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (' + args['anomaly'] + ')')
plt.legend(loc='best')
fig = plt.gcf()
fig.savefig('plots/' + args['anomaly'] + '/ROC.png')
plt.show()

# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve' + args['anomaly'] + ') (zoomed in at top left)')
plt.legend(loc='best')
fig = plt.gcf()
fig.savefig('plots/' + args['anomaly'] + '/ROC(zoom).png')
plt.show()
