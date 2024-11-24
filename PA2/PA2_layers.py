import numpy as np
from PA2_functions import sigmoid, softmax, cross_entropy_loss


# modified PA1's OnlyPythonLinearLayer
class LinearLayer:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        
        self.input = None

    def forward(self, input):
        self.input = input
        weight, bias = self.params
        return np.dot(self.input, weight) + bias
    
    def backward(self, before_gradient):
        weight, bias = self.params
        gradient = np.dot(before_gradient, weight.T)
        self.grads[0][...] = np.dot(self.input.T, before_gradient)
        self.grads[1][...] = np.sum(before_gradient, axis=0)

        return gradient


class LinearLayerWithoutBias:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        
        self.input = None

    def forward(self, input):
        self.input = input
        weight, = self.params
        return np.dot(self.input, weight)
    
    def backward(self, before_gradient):
        weight, = self.params
        gradient = np.dot(before_gradient, weight.T)
        self.grads[0][...] = np.dot(self.input.T, before_gradient)

        return gradient


# modified PA1's OnlyPythonRelu
class ReLU:
    def __init__(self):
        self.params, self.grads = [], []

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


# modified PA1's OnlyPythonCrossEntrophyLoss
class CrossEntrophyLoss:
    def __init__(self):
        self.params, self.grads = [], []

        self.loss = None
        self.x_softmax: np.ndarray = None
        self.label: np.ndarray = None

    def forward(self, x, label):
        self.label = label
        self.x_softmax = softmax(x)
        self.loss = cross_entropy_loss(self.x_softmax, self.label)

        return self.loss
    
    def backward(self, before_gradient=1):
        batch_size = self.label.shape[0]
        gradient = (self.x_softmax - self.label) / batch_size

        return gradient
    

class Tanh:
    def __init__(self):
        self.params, self.grads = [], []
        self.x = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out ** 2)



class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.x = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out



class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, is_train=True):
        if is_train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x
    
    def backward(self, dout):
        return dout * self.mask



class Embedding:
    def __init__(self, word_to_vec_map):
        self.map = word_to_vec_map

    def get_wordvec_sentence(self, sentence):
        sentence_list = sentence.split()
        sentence_word_vec = []
        for word in sentence_list:
            tmp_word_vec = self.map[word.lower()]
            sentence_word_vec.append(tmp_word_vec.reshape((1,50)))

        return sentence_word_vec



class VanilaRNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.x_layer = LinearLayerWithoutBias(self.params[0])
        self.h_layer = LinearLayerWithoutBias(self.params[1])
        self.tanh_layer = Tanh()


    def forward(self, x, h_prev):
        _, _, b = self.params
        h_next = self.tanh_layer.forward((self.h_layer.forward(h_prev) + self.x_layer.forward(x) + b))
        
        return h_next

    def backward(self, dh_next):
        dtanh = self.tanh_layer.backward(dh_next)
        db = np.sum(dtanh, axis=0)
        
        dh_prev = self.h_layer.backward(dtanh)
        dWh = self.h_layer.grads[0]
        
        dx = self.x_layer.backward(dtanh)
        dWx = self.x_layer.grads[0]

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev



class LSTM:
    def __init__(self, Wx, Wh, b):
        # Wx, Wh shape is N x 4H
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.x_layer = LinearLayerWithoutBias(self.params[0])
        self.h_layer = LinearLayerWithoutBias(self.params[1])
        self.forget_sigmoid_layer = Sigmoid()
        self.add_memory_tanh_layer = Tanh()
        self.input_sigmoid_layer = Sigmoid()
        self.output_sigmoid_layer = Sigmoid()
        self.h_next_tanh_layer = Tanh()

        self.cache = None

    def forward(self, x, h_prev, c_prev):
        _, _, b = self.params
        _, H = h_prev.shape

        affine_res = self.h_layer.forward(h_prev) + self.x_layer.forward(x) + b

        forget_gate = self.forget_sigmoid_layer.forward(affine_res[:,:H])
        add_memory = self.add_memory_tanh_layer.forward(affine_res[:, H:2*H])
        input_gate = self.input_sigmoid_layer.forward(affine_res[:, 2*H:3*H])
        output_gate = self.output_sigmoid_layer.forward(affine_res[:,3*H:4*H])
        
        c_next = forget_gate * c_prev + add_memory  * input_gate
        h_next = output_gate * self.h_next_tanh_layer.forward(c_next)

        self.cache = (output_gate, c_next, c_prev, input_gate, add_memory, forget_gate)

        return h_next, c_next


    def backward(self, dh_next, dc_next):
        output_gate, c_next, c_prev, input_gate, add_memory, forget_gate = self.cache

        d_1 = dc_next + self.h_next_tanh_layer.backward(dh_next * output_gate)

        d_forget = self.forget_sigmoid_layer.backward(d_1 * c_prev)
        d_add = self.add_memory_tanh_layer.backward(d_1 * input_gate)
        d_input = self.input_sigmoid_layer.backward(d_1 * add_memory)
        d_output = self.output_sigmoid_layer.backward(dh_next * np.tanh(c_next))

        d_affine_res = np.hstack((d_forget, d_add, d_input, d_output))

        dx = self.x_layer.backward(d_affine_res)
        dWx = self.x_layer.grads[0]
        
        dh_prev = self.h_layer.backward(d_affine_res)
        dWh = self.h_layer.grads[0]
        db = np.sum(d_affine_res, axis=0)

        dc_prev = d_1 * forget_gate

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db        
        
        return dx, dh_prev, dc_prev