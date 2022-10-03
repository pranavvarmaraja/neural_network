# Fully-Connected Feed Forward Neural Network

## Main Objective

This repository contains an implementation of a fully connected feed forward neural network with 2 hidden activation layers. Each layer (input and output included) can be configured to contain an arbitrary number of activations. Training is achieved via gradient descent, implemented via conventional backpropagation. This project reduces a NN to its core and is optimized for low memory usage (making use of basic arrays rather than other, more bloated data structures. Furthermore, given its nature as a fully connected network, this memory optimization was of paramount importance due to the sheer number of partial derivates which need to be calculated depending on the size of the network. By default, the network is configured to recognize the number of fingers held up on a single human hand, the train/test images/files are included in this repository. However, the network can be trained and ran on input vectors of any size for any purpose. If used for a different purpose, keep in mind the limitations caused by the lack of recurrence and convolution within the network (meaning that it is not ideal for tasks requiring a huge network with many activations including, but not limited to, NLP and CV).

## Folder Structure

The project is structured as followed:

- `src`: directory containing training/running source code
- `lib`: contains .jar file of json library used to parse network configuration files (in json)
- `images`: contains the .bmp images of both raw and cleaned (grayscaled and cropped) images
- `training_files`: converted versions of the cleaned image files used to train the network
- `testing_files`: converted versions of the cleaned image files used to test the network

## Working With the Network

