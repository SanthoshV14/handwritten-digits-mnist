# Overview
This code uses the MNIST dataset to train a simple neural network model in Keras and a convolutional neural network using Tensorflow.

# Dataset
The MNIST dataset contains images of handwritten digits (0-9). The dataset is divided into 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale image.

<img src="https://github.com/SanthoshV14/handwritten-digits-mnist/blob/main/img/dataset.png" />

# Model Architecture

<ol>
<li>MLP - (handwritten_digits_mnist_mlp.ipynb)</li>
The neural network model used in this code has the following architecture: </br>

<ol>
<li>Input layer</li> Flattens the input image into a 784-element vector. </br>
<li>Hidden layers</li> Three dense layers, each with 12 units and a ReLU activation function. </br>
<li>Output layer</li> Dense layer with 10 units and a softmax activation function. </br> </br>
</ol>

Training and Evaluation: </br>
The model is trained using mean squared error loss and stochastic gradient descent optimizer. The accuracy is used as a metric to evaluate the performance of the model. </br>

The model is trained for 30 epochs with a batch size of 32. The trained model is evaluated on the test set and achieves an accuracy of 77.52%. </br>

<li>CNN - (handwritten_digits_mnist_cnn.ipynb)</li>
The convolutional neural network model used in this code has the following architecture: </br>
<ol>
<li>Input layer</li> Takes an input image of size 28x28 with a depth of 1 (grayscale). </br>
<li>Hidden layers</li> The layer has 5 filters of size 2x2 with ReLU activation function, which creates 5 feature maps. The padding is set to "same" to ensure that the output has the same spatial dimensions as the input. The second layer is a max-pooling layer with a 2x2 pool size, which reduces the spatial dimensions of each feature map by a factor of two. The third layer is another convolutional layer with 5 filters of size 2x2 and ReLU activation function. The padding is again set to "same". </br>
<li>Output layer</li> The output is flattened to a one-dimensional array of 980 elements. A dropout layer with a rate of 0.5 is used to prevent overfitting, and a fully connected output layer with 10 nodes and softmax activation function is used to produce the output probabilities for each of the 10 classes. </br> </br>
</ol>

Training and Evaluation: </br>
The model is trained using mean squared error loss and stochastic gradient descent optimizer. The accuracy is used as a metric to evaluate the performance of the model. </br>

The model has a total of 9,940 trainable parameters, which is relatively small compared to more complex CNN architectures. The model was trained for 5 epochs using the Adam optimizer and mean squared error loss function. The training accuracy of the model improved significantly with each epoch, reaching 93.52% accuracy on the training set and 97.54% accuracy on the validation set after 5 epochs. The model was evaluated on the test set and achieved an accuracy of approximately 97.31%. </br>
</ol>

# Saving the Model
The trained model is saved in the current directory using the Keras save function.

# Dependencies
<ul>
<li>tensorflow</li>
<li>keras</li>
<li>numpy</li>
<li>matplotlib</li>
</ul>

# Reults
<ul>
<li>MLP</li>
The model has less than 10K learnable parameters and achieved an accuracy of 77.52% on the validation set.
<img src="https://github.com/SanthoshV14/handwritten-digits-mnist/blob/main/img/mlp-accuracy-plot.png" />

<li>CNN</li>
The model has less than 10K learnable parameters and achieved an accuracy of 97.31% on the validation set.
<img src="https://github.com/SanthoshV14/handwritten-digits-mnist/blob/main/img/cnn-accuracy-plot.png" />
</ul>

# Acknowledgements
The MNIST dataset is a widely used dataset for image classification tasks. The dataset was downloaded from the Keras library.

# Author
Santhos Vadivel </br>
Email - ssansh3@gmail.com </br>
LinkedIn - https://www.linkedin.com/in/santhosh-vadivel-2141b8126/ </br>
