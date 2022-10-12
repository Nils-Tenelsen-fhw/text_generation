Code based on: https://www.tensorflow.org/text/tutorials/text_generation

Character-level RNN for text generation in Python using TensorFlow

A rough list of used libraries and versions can be found in the corresponding thesis.

The test.py file executes the training of the network. Within the file a number of variables can be used to alter the input data (fullPath) or the output location for the trained model (output_path).

Load.py can be used to load an already trained model and generate a number of sequences with it. The location of the model can be specified in the first line of the program.
Load_from_checkpoint.py can be used to load an already trained model at it's state during any epoch of the training process, the program then generates a few sequences. The path to the saved model can be specified using the variable "load_path".

texts.csv contains example data that can be used to train the model.
