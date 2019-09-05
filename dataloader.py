import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_data(traffic_type, path,
              rows=None, skip=0, most_common_ports=None):
    filename = path + traffic_type + ".csv"
    print("Loading data from %s... (this might take a while)" % filename)
    data = pd.read_csv(filename, sep=',', skiprows=range(1, skip), nrows=rows, skipinitialspace=True)

    # Removing the labels
    data = data.drop(['Binary_Label', 'Label'], axis=1)

    # Hot one encoding of the ports
    values = most_common_ports if most_common_ports is not None else get_most_common_values(data['Destination Port'])
    data['Destination Port'] = data['Destination Port'].where(data['Destination Port'].isin(values), 'other')
    data['Destination Port'] = pd.Categorical(data['Destination Port'], categories=values + ['other'])
    port_dummies = pd.get_dummies(data['Destination Port'], prefix='port')
    data = data.drop('Destination Port', axis=1)

    # Normalizing the data
    scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(data)

    data = np.hstack((port_dummies, data))

    return {'data': data, 'most_common_ports': values}


def get_most_common_values(column, thresh=0.85):
    value_counts = column.value_counts()
    acc = 0
    values = []
    for index in value_counts.index:
        if 1.0 * acc / column.shape[0] >= thresh or len(values) >= 100:
            break
        acc += value_counts[index]
        values.append(index)
    return values
