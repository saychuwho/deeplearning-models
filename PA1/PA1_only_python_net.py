import numpy as np
from PA1_only_python_layers import *


class OnlyPythonThreeLayerNet:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, learning_rate=0.01):
        self.linear1 = OnlyPythonLinearLayer(n_input, n_hidden1)
        self.linear2 = OnlyPythonLinearLayer(n_hidden1, n_hidden2)
        self.linear3 = OnlyPythonLinearLayer(n_hidden2, n_output)
        self.softmaxlayer = OnlyPythonCrossEntrophyLoss()

        self.relu1 = OnlyPythonReLU()
        self.relu2 = OnlyPythonReLU()

        self.lr = learning_rate


    def __repr__(self):
        return "OnlyPythonThreeLayerNet"


    def forward(self, input):
        output = input
        output = self.linear1.forward(output)
        output = self.relu1.forward(output)
        output = self.linear2.forward(output)
        output = self.relu2.forward(output)
        output = self.linear3.forward(output)
        
        return output
    

    def loss(self, output, label):
        loss = self.softmaxlayer.forward(output, label)

        return loss
    
    

    def backward(self):
        dy = self.softmaxlayer.backward()
        dy = self.linear3.backward(dy)
        dy = self.relu2.backward(dy)
        dy = self.linear2.backward(dy)
        dy = self.relu1.backward(dy)
        dy = self.linear1.backward(dy)
        


    def optimizer(self):
        self.backward()

        self.linear1.weight -= self.lr * self.linear1.weight_gradient
        self.linear1.bias -= self.lr * self.linear1.bias_gradient

        self.linear2.weight -= self.lr * self.linear2.weight_gradient
        self.linear2.bias -= self.lr * self.linear2.bias_gradient

        self.linear3.weight -= self.lr * self.linear3.weight_gradient
        self.linear3.bias -= self.lr * self.linear3.bias_gradient


class OnlyPythonCNN:
    def __init__(self, lr=0.01, kernel_size=3, kernel_num=32):
        self.kernel_size = kernel_size
        self.input_channel_size = 1
        self.kernel_num = kernel_num
        self.output_size = 10
        self.learning_rate = lr

        self.output_shape = None

        self.conv1 = OnlyPythonConvolution(self.kernel_num, self.input_channel_size, self.kernel_size, self.kernel_size, 1, 1)
        self.relu1 = OnlyPythonReLU()
        self.maxpool1 = OnlyPythonMaxPooling(2, 2)
        self.conv2 = OnlyPythonConvolution(self.kernel_num, self.kernel_num, self.kernel_size, self.kernel_size)
        self.relu2 = OnlyPythonReLU()
        self.maxpool2 = OnlyPythonMaxPooling(2,2)
        self.linear = OnlyPythonLinearLayer(self.kernel_num * 6 * 6, self.output_size)
        self.softmaxlayer = OnlyPythonCrossEntrophyLoss()


    def __repr__(self):
        return "OnlyPythonCNN"


    def forward(self, input):
        output = input
        output = self.conv1.forward(output)
        output = self.relu1.forward(output)
        output = self.maxpool1.forward(output)
        output = self.conv2.forward(output)
        output = self.relu2.forward(output)
        output = self.maxpool2.forward(output)

        self.output_shape = output.shape
        output = output.reshape(output.shape[0], -1)

        output = self.linear.forward(output)
        
        return output
    

    def loss(self, output, label):
        loss = self.softmaxlayer.forward(output, label)

        return loss
    

    def backward(self):
        dy = self.softmaxlayer.backward()
        dy = self.linear.backward(dy)

        dy = dy.reshape(self.output_shape)

        dy = self.maxpool2.backward(dy)
        dy = self.relu2.backward(dy)
        dy = self.conv2.backward(dy)
        dy = self.maxpool1.backward(dy)
        dy = self.relu1.backward(dy)
        dy = self.conv1.backward(dy)


    def optimizer(self):
        self.backward()

        self.conv1.weight -= self.learning_rate * self.conv1.weight_gradient
        self.conv1.bias -= self.learning_rate * self.conv1.bias_gradient

        self.conv2.weight -= self.learning_rate * self.conv2.weight_gradient
        self.conv2.bias -= self.learning_rate * self.conv2.bias_gradient

        self.linear.weight -= self.learning_rate * self.linear.weight_gradient
        self.linear.bias -= self.learning_rate * self.linear.bias_gradient