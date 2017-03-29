# Convolutional-Neural-Net-Hvass-Labs
A Convolutional Neural Network built with Tensorflow to classify the MNIST handwritten digits

### Objective
+ To classify MNIST handwritten digits.
+ To get up to 99% accuracy with the Convolutional Neural Network Model
+ To visualize the weights and predictions (both correct and incorrect)

### Dependencies
+ `tensorflow` for building the Neural Network.
+ `numpy` for scientific computing and using numpy's `array`.
+ `matplotlib` to visulaise the accurate and inaccurate predictions
+ `sklearn` to get the confusion matrix of the sensitivity and subjectivity.
+ `tqdm` for progress bar while the network is training and to know how long it'll take to complete.

### Method
+ Load in the dataset using the `read_data_sets()` function in tensorflow.
+ `one_hot` is set to `True` for computational reasons.
+ Building the model is divided into two parts: – Building the computational graph and – Running the graph.
##### Building tensorflow's computational graph
+ Build the tensorflow's computational graph which defines all the computational operations (called ops in tensorflow's grammar).
+ Two placeholder variables are created; `X` and `y` which would be feed into the network later on during training.
+ Four helper functions are defined: 
* `weights()` for initializing random weights with a standard deviation of 0.04.
* `bias()` for initializing the biases to start with an initial value of 0.04.
* `conv2D()` for performing a convolution operation with a stride of 1x1 and padding: 'SAME' i.e there will be zero padding at the end. To make the input image and output image the same size.
* `max_pool()` for scaling down the image. With both kernel size and strides set to 2x2 and padding: 'SAME' for the same reason as the above.
+ The Convolutional Neural Network Model consist of two convolutional layers, one fully connected layer with dropout and the final readout or output layer.
+ The model's error is evaluated using the tensorflow's `softmax_cross_entropy_with_logits()` function and minimize via tensorflow's `AdamOptimizer().minimize()`, which iteratively decays/lowers the learning rate through time.
+ Finally, the accuracy is deetermined with the tensorflow's helper functions: `tf.equal()` and `tf.argmax()` function. The average of the correct and incorrect predicitons is gotten using the tensorflow's helper function: `tf.reduce_mean()`.
##### Running the computational graph.
+ In order to run the tensorflow's computational graph, we define a `Session()` variable which encapsulate the above operations into a default graph.
+ Using the tensorflow's `global_variables_initializer()` function, we initialize all the variables created in the above computational graph.
+ The network is trained for a set number of iterations.
+ The accuracy is being run through the `Session()` variable created.
+ More visualizations is done to evaluated visually the model's accuracy (correct predictions) and flaws (incorrect predictions).
+ The confusion matrix is also printed to the console to gain more insights on the model's behaviour

### Results
+ 99.3% Accuracy on the MNIST dataset (which is considered okay for the size of this dataset but could be better by tunning some hyperparameters of the model).
+ Demonstration of how _Forward Propagation_ and _Backward Propagation_ works.
+ Visualizing the so called _weights_ of the neural network.
+ Visualizing the confusion matrix.
