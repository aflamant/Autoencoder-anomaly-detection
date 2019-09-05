import argparse
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import json
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, recall_score, precision_score
import csv
from random import randrange

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
benign_data = dataloader.load_data("BENIGN", path=args['path'],  skip=randrange(2273097 - 8000), rows=8000, most_common_ports=ports_list)['data']
anomalous_data = dataloader.load_data(args["anomaly"], path=args['path'], rows=2000, most_common_ports=ports_list)['data']

# Tagging the data to distinguish benign and anomalous traffic
tagged_benign_data = np.hstack((benign_data, np.zeros(benign_data.shape[0]).reshape(-1, 1)))            # 0 == BENIGN
tagged_anomalous_data = np.hstack((anomalous_data, np.ones(anomalous_data.shape[0]).reshape(-1, 1)))    # 1 == ANOMALOUS

# Concatenating the two sets and shuffling them
data = np.vstack((tagged_anomalous_data, tagged_benign_data))

# Predicting values using the autoencoder and calculating the reconstruction error
prediction = autoencoder.predict(x=data[:, :-1])
reconstruction_error = np.linalg.norm(data[:, :-1] - prediction, axis=-1)
anomaly = data[:, -1]

# Selecting the best threshold to maximize F1 score
scores = {}
for thresh in np.arange(0, 10, 0.01):
    predicted_anomalies = reconstruction_error > thresh
    score = f1_score(anomaly, predicted_anomalies)
    scores[thresh]=score

selected_thresh = max(scores, key=scores.get)
selected_score = scores[selected_thresh]
predicted_anomalies = reconstruction_error > selected_thresh

# Plotting ROC curve
fpr, tpr, thresholds = roc_curve(anomaly, reconstruction_error)
auc_ = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='ROC (area = {:.3f})'.format(auc_))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (' + args['anomaly'] + ')')
plt.legend(loc='best')
fig = plt.gcf()
fig.savefig('plots/' + args['anomaly'] + '/ROC3.png')
plt.show()

# Plotting the reconstruction error
yellow_patch = mpatches.Patch(color='yellow', label='Anomalous data')
purple_patch = mpatches.Patch(color='purple', label='Benign data')
red_line = mlines.Line2D([], [], color='red', label='Threshold (F1 score = %.2f)' % selected_score)

plt.scatter(np.arange(0,benign_data.shape[0]+anomalous_data.shape[0]), reconstruction_error, c=data[:, -1])
plt.axhline(selected_thresh, linewidth=2, color='red')
plt.legend(handles = [yellow_patch, purple_patch, red_line])
plt.title('Reconstruction error comparison (%s)' % args['anomaly'])
plt.xlabel('Sample index')
plt.ylabel('Reconstruction error')
fig = plt.gcf()
fig.savefig('plots/' + args['anomaly'] + '/reconstruction_error3.png')
plt.show()

# Saving performance metrics to a file
auc_score = auc_
precision = precision_score(anomaly, predicted_anomalies)
recall = recall_score(anomaly, predicted_anomalies)
accuracy = accuracy_score(anomaly, predicted_anomalies)
f1 = selected_score

with open('metrics.csv', 'a') as f:
    thewriter = csv.writer(f)

    thewriter.writerow([args['anomaly'], auc_score, selected_thresh, accuracy, precision, recall, f1])
