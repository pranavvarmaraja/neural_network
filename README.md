# Fully-Connected Feed Forward Neural Network

## Main Objective

This repository contains an implementation of a fully connected feed forward neural network with 2 hidden activation layers. Each layer (input and output included) can be configured to contain an arbitrary number of activations. Training is achieved via gradient descent, implemented via conventional backpropagation. This project reduces a NN to its core and is optimized for low memory usage (making use of basic arrays rather than other, more bloated data structures. Furthermore, given its nature as a fully connected network, this memory optimization was of paramount importance due to the sheer number of partial derivatives which need to be calculated depending on the size of the network. By default, the network is configured to recognize the number of fingers held up on a single human hand, the train/test images/files are included in this repository. However, the network can be trained and ran on input vectors of any size for any purpose. If used for a different purpose, keep in mind the limitations caused by the lack of recurrence and convolution within the network (meaning that it is not ideal for tasks requiring a huge network with many activations including, but not limited to, NLP and CV).

## Folder Structure

The project is structured as followed:

- `src`: directory containing training/running source code
- `lib`: contains .jar file of json library used to parse network configuration files (in json)
- `images`: contains the .bmp images of both raw and cleaned (grayscaled and cropped) images
- `training_files`: converted versions of the cleaned image files used to train the network
- `testing_files`: converted versions of the cleaned image files used to test the network

## Working With the Network

### Running/Training the Network

In order to use the network (run/train) your local installation of java must be up to date and your java install must be added to your `$PATH`. To work with the network, cd into the main project directory and then execute the following command

 `bash execute.sh <configFileName>`

This will run or train the network depending on the configuration file provided. An example configuration file is provided in the main project directory as `configFile2.json`. The file reads as follows, and the descriptions for what each parameter does in the context of the network is outlined below.

```json
{ 
  "runOnly":true,
  "preLoadedWeights":true,
  "weightFileName":"outputWeights.json",
  "numInputs":2,
  "numHiddenNodes1":20,
  "numHiddenNodes2":5,    
  "numOutputs":3,
  "numPossibleInputs":4,
  "truthTableFileName":"truthTable.json",
  "inputFileName":"possibleInputs.json"
}
```

### Config File Parameters

The parameter names, descriptions, and expected values are listed below:

| Parameter Name            | Expected Value    | Description                                                  |
| ------------------------- | ----------------- | ------------------------------------------------------------ |
| `"runOnly"`               | `true/false`      | Determines whether network is to be executed in run only mode or training mode |
| `"preLoadedWeights"`      | `true/false`      | Determines whether the network will load weights from a prexisting weights file |
| `"weightFileName"`        | `<fileName.json>` | Determines the name of the weights json configuration file from which weights are read from |
| `"numInputs"`             | `int`             | Determines the size of the input vector                      |
| `"numHiddenNodes1"`       | `int`             | Determines the number of activations in the first hidden layer |
| `"numHiddenNodes2"`       | `int`             | Determines the number of activations in the second hidden layer |
| `"numOutputs"`            | `int`             | Determines the size of the output vector                     |
| `"numPossibleInputs"`     | `int`             | Determines the number of possible input sets (inputs for the network to classify/train on) |
| `"truthTableFileName"`    | `<fileName.json>` | Determines the file to which the outputs should be compared  |
| `"inputFileName"`         | `<fileName.json>` | Determines the name of the file containing the filepaths to input image files (*.in) |
| `"maxIterations"`         | `int`             | If training, the maximum number of iterations for the network to complete before training is concluded. An iteration is defined by a single gradient step |
| `"outputWeightsFileName"` | `<fileName.json>` | Determines the file to which the network weights are saved (will only work for a given network configuration ABCD). |
| `"minRandomWeight"`       | `double`          | The minimum random initial weight for each weight within the network. |
| `"maxRandomWeight"`       | `double`          | The maximum random initial weight for each weight within the network. |
| `"errorThreshold"`        | `double`          | If training, the error which, when reached, will conclude the training process |
| `"lambda"`                | `double`          | The learning rate of the network (multiplicative factor by which the weights are changed by the gradient) |

###  Final Notes

Thus, to fully utilize the network, create a json configuration file and pass it to the execution script. This will run or train the network using the given options and connecting files (containing weights, truth tables, inputs, etc).
