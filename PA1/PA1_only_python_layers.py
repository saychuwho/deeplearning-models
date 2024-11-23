import numpy
from PA1_only_python_functions import *


class OnlyPythonLinearLayer:
    def __init__(self, input_size, output_size):
        self.weight = 0.01 * np.random.randn(input_size, output_size)
        self.bias = np.zeros(output_size)
        
        self.input = None
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weight) + self.bias
    
    def backward(self, before_gradient):
        gradient = np.dot(before_gradient, self.weight.T)
        self.weight_gradient = np.dot(self.input.T, before_gradient)
        self.bias_gradient = np.sum(before_gradient, axis=0)

        return gradient


class OnlyPythonReLU:
    def __init__(self):
        self.is_zero = None

    def forward(self, input:np.ndarray):
        self.is_zero = (input<=0)
        output = input.copy()
        output[self.is_zero] = 0

        return output
    
    def backward(self, before_gradient):
        before_gradient[self.is_zero] = 0
        gradient = before_gradient

        return gradient


class OnlyPythonCrossEntrophyLoss:
    def __init__(self):
        self.loss = None
        self.x_softmax: np.ndarray = None
        self.label: np.ndarray = None

    def forward(self, x, label):
        self.label = label
        self.x_softmax = only_python_softmax(x)
        self.loss = only_python_cross_entropy_loss(self.x_softmax, self.label)

        return self.loss
    
    def backward(self):
        batch_size = self.label.shape[0]
        gradient = (self.x_softmax - self.label) / batch_size

        return gradient



class OnlyPythonConvolution:
    def __init__(self, kernel_num, channel_size, kernel_h, kernel_w, stride=1, padding=0):
        self.weight = 0.01 * np.random.randn(kernel_num, channel_size, kernel_h, kernel_w)
        self.bias = np.zeros(kernel_num)
        self.stride = stride
        self.padding = padding

        self.input = None
        self.col = None
        self.col_W = None
        self.weight_gradient = None
        self.bias_gradient = None

    
    def forward(self, input: np.ndarray):
        kernel_num, _, kernel_h, kernel_w = self.weight.shape
        batch_size, _, height, width = input.shape
        out_h = (height + 2*self.padding - kernel_h) // self.stride + 1
        out_w = (width + 2*self.padding - kernel_w) // self.stride + 1

        col = img2col(input, kernel_h, kernel_w, self.stride, self.padding)
        col_W = self.weight.reshape(kernel_num, -1).T

        out = np.dot(col, col_W) + self.bias
        out = out.reshape(batch_size, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.input = input
        self.col = col
        self.col_W = col_W

        return out


    def backward(self, before_gradient: np.ndarray):
        kernel_num, channel_size, kernel_h, kernel_w = self.weight.shape
        before_gradient = before_gradient.transpose(0,2,3,1).reshape(-1, kernel_num)

        self.bias_gradient = np.sum(before_gradient, axis=0)
        self.weight_gradient = np.dot(self.col.T, before_gradient)
        self.weight_gradient = self.weight_gradient.transpose(1, 0).reshape(kernel_num, channel_size, kernel_h, kernel_w)

        col_gradient = np.dot(before_gradient, self.col_W.T)
        gradient = col2img(col_gradient, self.input.shape, kernel_h, kernel_w, self.stride, self.padding)

        return gradient
    

class OnlyPythonMaxPooling:
    def __init__(self, pool_h, pool_w):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = self.pool_h

        self.input = None
        self.arg_max = None

    
    def forward(self, input: np.ndarray):
        batch_size, channel_size, height, width = input.shape
        out_h = (height - self.pool_h) // self.stride + 1
        out_w = (width - self.pool_w) // self.stride + 1

        col = img2col(input, self.pool_h, self.pool_w, self.stride)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(batch_size, out_h, out_w, channel_size).transpose(0,3,1,2)

        self.input = input
        self.arg_max = arg_max

        return out
    

    def backward(self, before_gradient: np.ndarray):
        before_gradient = before_gradient.transpose(0,2,3,1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((before_gradient.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = before_gradient.flatten()
        dmax = dmax.reshape(before_gradient.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        gradient = col2img(dcol, self.input.shape, self.pool_h, self.pool_w, self.stride)

        return gradient
