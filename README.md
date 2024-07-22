This project utilizes machine learning to identify birds based on a recording of their call.

The program cleans the data by decomposing the .wav files using the daubchies wavelet. The features produces by the daubchies wavelet are then used to train and test the machine learning model. The chosen model was Gaussian Naive Bayes.

The model achieves a peak accuracy of 73%. We believe a limited number of datapoints is the reason for the low accuracy.
