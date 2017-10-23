# siamese-transfer-learning
Deep learning frameworks such as Convolutional Neural Networks (CNNs) take a lot of samples
to train, even if the lower layers are transferred from a pretrained CNN. In this work, we propose
that the ability to compare a pair of images for similarity should also be considered for such transfer
learning. We are developing a Siamese network based CNN architecture which compares pairs of
images for similarity; the output of the network can be transformed into a Mercer kernel to allow
utilization of wide margin classification properties of a SVM that is useful when the amount of
training data is scarce.
