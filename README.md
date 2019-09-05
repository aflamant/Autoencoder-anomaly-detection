# Autoencoder-anomaly-detection
Anomaly detection based on autoencoder reconstruction error

# Usage

## Generating the autoencoder

Launch ```train_encoder.py``` with the corresponding parameters.

To generate a one hidden layer autoencoder:
    python3 train_autoencoder.py -m output_model.h5 -p /path/to/datasets/ -d 0.5

3 hidden layers:

    python3 train_autoencoder.py -m output_model.h5 -p /path/to/datasets/ -d 0.7 0.4

## Anomaly detection

Launch the ```anomaly_detection.py``` with the corresponding arguments.
Example:
    python3 anomaly_detection.py -m input_model.h5 -p /path/to/datasets/ -l ports_list.json -a ANOMALY_NAME

## Dataset requirements

The datasets must be placed in a folder with each anomaly type in its own CSV file and benign traffic in a file called ```BENIGN.csv```
