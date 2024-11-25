import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr_init = lr
        self.lr = self.lr_init
        self.iter = 0
        self.k = 0.1
        self.step_to_decay = 50
        self.before_decay_step = 0

    def update(self, params, grads):
        self.iter += 1
        
        # my lr decay
        # if self.iter % 10 == 0: self.lr /= 2.0              
        
        # exponential rate scheduling    
        # self.lr = self.lr_init * np.exp(-self.k*self.iter)  
        
        if self.iter % self.step_to_decay == 0 and self.before_decay_step != 0:
            self.lr = self.lr_init * (self.k ** (self.iter / self.before_decay_step))
            self.before_decay_step = self.iter

        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

        

class Adam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iter = 0

        self.m = None
        self.v = None
        

    def update(self, params, grads):
        if self.m == None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta_2 ** self.iter) / (1.0 - self.beta_1 ** self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta_1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta_2) * (grads[i]**2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)