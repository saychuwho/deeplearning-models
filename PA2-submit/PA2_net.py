import numpy as np
from PA2_functions import softmax
from PA2_layers import *


class PA2VanilaRNN:
    def __init__(self, word_to_vec_map, word_vec_size, is_dropout=True):
        self.word_vec_size = word_vec_size
        self.hidden_state_size = 128
        self.label_size = 5
        self.is_dropout = is_dropout

        # parameters
        rnn_1_Wx = (np.random.randn(self.word_vec_size, self.hidden_state_size)/np.sqrt(self.word_vec_size)).astype('f')
        rnn_1_Wh = (np.random.randn(self.hidden_state_size, self.hidden_state_size)/np.sqrt(self.hidden_state_size)).astype('f')
        rnn_1_b = np.zeros(self.hidden_state_size).astype('f')
    
        rnn_2_Wx = (np.random.randn(self.hidden_state_size, self.hidden_state_size)/np.sqrt(self.hidden_state_size)).astype('f')
        rnn_2_Wh = (np.random.randn(self.hidden_state_size, self.hidden_state_size)/np.sqrt(self.hidden_state_size)).astype('f')
        rnn_2_b = np.zeros(self.hidden_state_size).astype('f')
        
        fc_W = (np.random.randn(self.hidden_state_size, self.label_size)/np.sqrt(self.hidden_state_size)).astype('f')
        fc_b = np.zeros(self.label_size).astype('f')

        # layers
        self.embedding = Embedding(word_to_vec_map)
        self.sentence_len = None

        self.rnn_1 = VanilaRNN(rnn_1_Wx, rnn_1_Wh, rnn_1_b)
        self.rnn_1_h = np.zeros((1,self.hidden_state_size), dtype='f')
        self.rnn_1_dh = np.zeros((1,self.hidden_state_size), dtype='f')
        
        self.dropout_1 = Dropout()

        self.rnn_2 = VanilaRNN(rnn_2_Wx, rnn_2_Wh, rnn_2_b)
        self.rnn_2_h = np.zeros((1,self.hidden_state_size), dtype='f')
        self.rnn_2_dh = np.zeros((1,self.hidden_state_size), dtype='f')

        self.dropout_2 = Dropout()
        self.fc = LinearLayer(fc_W, fc_b)
        

        self.loss_layer = CrossEntrophyLoss()

        # add params, grads for optimizer
        self.params, self.grads = [], []
        self.params += self.rnn_1.params
        self.grads += self.rnn_1.grads
        self.params += self.rnn_2.params
        self.grads += self.rnn_2.grads
        self.params += self.fc.params
        self.grads += self.fc.grads


    def forward(self, sentence):
        wordvec_sentence = self.embedding.get_wordvec_sentence(sentence)
        self.sentence_len = len(wordvec_sentence)

        forward_res = None

        # first phase
        for wordvec in wordvec_sentence:
            forward_res = self.rnn_1.forward(wordvec, self.rnn_1_h)
            self.rnn_1_h = forward_res
            if self.is_dropout:
                forward_res = self.dropout_1.forward(forward_res)
            forward_res = self.rnn_2.forward(forward_res, self.rnn_2_h)
            self.rnn_2_h = forward_res
        
        # second phase
        if self.is_dropout:
            forward_res = self.dropout_2.forward(forward_res)
        forward_res = self.fc.forward(forward_res)
        
        return forward_res


    def loss(self, forward_res, ans_label):
        # change ans_label to one-hot vector
        one_hot_ans_label = np.zeros(5)
        one_hot_ans_label[ans_label] = 1
        one_hot_ans_label = one_hot_ans_label.astype('f')

        loss = self.loss_layer.forward(forward_res, one_hot_ans_label)

        return loss


    def predict(self, sentence):
        # do forward process with is_train=False
        wordvec_sentence = self.embedding.get_wordvec_sentence(sentence)
        forward_res = None

        # first phase
        for wordvec in wordvec_sentence:
            forward_res = self.rnn_1.forward(wordvec, self.rnn_1_h)
            self.rnn_1_h = forward_res
            if self.is_dropout:
                forward_res = self.dropout_1.forward(forward_res, False)
            forward_res = self.rnn_2.forward(forward_res, self.rnn_2_h)
            self.rnn_2_h = forward_res
        
        # second phase
        if self.is_dropout:
            forward_res = self.dropout_2.forward(forward_res, False)
        forward_res = self.fc.forward(forward_res)

        return softmax(forward_res)


    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dout = self.fc.backward(dout)
        if self.is_dropout:
            dout = self.dropout_2.backward(dout)
        
        self.rnn_2_dh = dout
        for _ in range(self.word_vec_size):
            dout, self.rnn_2_dh = self.rnn_2.backward(self.rnn_2_dh)
            if self.is_dropout:
                dout = self.dropout_1.backward(dout)
            self.rnn_1_dh = dout
            dout, self.rnn_1_dh = self.rnn_1.backward(self.rnn_1_dh)

        return dout


class PA2LSTM:
    def __init__(self, word_to_vec_map, word_vec_size, dropout_ratio=0.5, is_adam=False, is_dropout=True):
        self.word_vec_size = word_vec_size
        self.hidden_state_size = 128
        self.label_size = 5
        self.is_dropout = is_dropout
        self.dropout_ratio = dropout_ratio

        # parameters
        lstm_1_Wx = 0.01*np.random.randn(self.word_vec_size, 4*self.hidden_state_size).astype('f')
        lstm_1_Wh = 0.01*np.random.randn(self.hidden_state_size, 4*self.hidden_state_size).astype('f')
        lstm_2_Wx = 0.01*np.random.randn(self.hidden_state_size, 4*self.hidden_state_size).astype('f')
        lstm_2_Wh = 0.01*np.random.randn(self.hidden_state_size, 4*self.hidden_state_size).astype('f')
        fc_W = 0.01*np.random.randn(self.hidden_state_size, self.label_size).astype('f')
        
        if not is_adam:
            lstm_1_Wx = (np.random.randn(self.word_vec_size, 4*self.hidden_state_size)/np.sqrt(self.word_vec_size)).astype('f')
            lstm_1_Wh = (np.random.randn(self.hidden_state_size, 4*self.hidden_state_size)/np.sqrt(self.hidden_state_size)).astype('f')
            lstm_2_Wx = (np.random.randn(self.hidden_state_size, 4*self.hidden_state_size)/np.sqrt(self.hidden_state_size)).astype('f')
            lstm_2_Wh = (np.random.randn(self.hidden_state_size, 4*self.hidden_state_size)/np.sqrt(self.hidden_state_size)).astype('f')
            fc_W = (np.random.randn(self.hidden_state_size, self.label_size)/np.sqrt(self.hidden_state_size)).astype('f')
        

        lstm_1_b = np.zeros(4*self.hidden_state_size).astype('f')
        lstm_2_b = np.zeros(4*self.hidden_state_size).astype('f')
        fc_b = np.zeros(self.label_size).astype('f')

        # layers
        self.embedding = Embedding(word_to_vec_map)
        self.sentence_len = None

        self.lstm_1 = LSTM(lstm_1_Wx, lstm_1_Wh, lstm_1_b)
        self.lstm_1_h = np.zeros((1,self.hidden_state_size), dtype='f')
        self.lstm_1_c = np.zeros((1,self.hidden_state_size), dtype='f')
        self.lstm_1_dh = np.zeros((1,self.hidden_state_size), dtype='f')
        self.lstm_1_dc = np.zeros((1,self.hidden_state_size), dtype='f')
        
        self.dropout_1 = Dropout(self.dropout_ratio)

        self.lstm_2 = LSTM(lstm_2_Wx, lstm_2_Wh, lstm_2_b)
        self.lstm_2_h = np.zeros((1,self.hidden_state_size), dtype='f')
        self.lstm_2_c = np.zeros((1,self.hidden_state_size), dtype='f')
        self.lstm_2_dh = np.zeros((1,self.hidden_state_size), dtype='f')
        self.lstm_2_dc = np.zeros((1,self.hidden_state_size), dtype='f')

        self.dropout_2 = Dropout(self.dropout_ratio)
        self.fc = LinearLayer(fc_W, fc_b)
        

        self.loss_layer = CrossEntrophyLoss()

        # add params, grads for optimizer
        self.params, self.grads = [], []
        self.params += self.lstm_1.params
        self.grads += self.lstm_1.grads
        self.params += self.lstm_2.params
        self.grads += self.lstm_2.grads
        self.params += self.fc.params
        self.grads += self.fc.grads


    def forward(self, sentence):
        wordvec_sentence = self.embedding.get_wordvec_sentence(sentence)
        self.sentence_len = len(wordvec_sentence)

        forward_res = None

        # first phase
        for wordvec in wordvec_sentence:
            forward_res, self.lstm_1_c = self.lstm_1.forward(wordvec, self.lstm_1_h, self.lstm_1_c)
            self.lstm_1_h = forward_res
            if self.is_dropout:
                forward_res = self.dropout_1.forward(forward_res)
            forward_res, self.lstm_2_c = self.lstm_2.forward(forward_res, self.lstm_2_h, self.lstm_2_c)
            self.lstm_2_h = forward_res
        
        # second phase
        if self.is_dropout:
            forward_res = self.dropout_2.forward(forward_res)
        forward_res = self.fc.forward(forward_res)
        
        return forward_res


    def loss(self, forward_res, ans_label):
        # change ans_label to one-hot vector
        one_hot_ans_label = np.zeros(5)
        one_hot_ans_label[ans_label] = 1
        one_hot_ans_label = one_hot_ans_label.astype('f')

        loss = self.loss_layer.forward(forward_res, one_hot_ans_label)

        return loss


    def predict(self, sentence):
        # do forward process with is_train=False
        wordvec_sentence = self.embedding.get_wordvec_sentence(sentence)
        forward_res = None

        # first phase
        for wordvec in wordvec_sentence:
            forward_res, self.lstm_1_c = self.lstm_1.forward(wordvec, self.lstm_1_h, self.lstm_1_c)
            self.lstm_1_h = forward_res
            if self.is_dropout:
                forward_res = self.dropout_1.forward(forward_res, False)
            forward_res, self.lstm_2_c = self.lstm_2.forward(forward_res, self.lstm_2_h, self.lstm_2_c)
            self.lstm_2_h = forward_res
        
        # second phase
        if self.is_dropout:
            forward_res = self.dropout_2.forward(forward_res, False)
        forward_res = self.fc.forward(forward_res)

        return softmax(forward_res)


    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dout = self.fc.backward(dout)
        if self.is_dropout:
            dout = self.dropout_2.backward(dout)
        
        self.lstm_2_dh = dout
        for _ in range(self.word_vec_size):
            dout, self.lstm_2_dh, self.lstm_2_dc = self.lstm_2.backward(self.lstm_2_dh, self.lstm_2_dc)
            if self.is_dropout:
                dout = self.dropout_1.backward(dout)
            self.lstm_1_dh = dout
            dout, self.lstm_1_dh, self.lstm_1_dc = self.lstm_1.backward(self.lstm_1_dh, self.lstm_1_dc)

        return dout