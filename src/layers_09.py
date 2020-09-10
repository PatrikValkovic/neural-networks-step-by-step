import numpy as np

class SoftmaxLayer:
    def __init__(self):
        self.params = []
        self.grads = []
    
    def __call__(self, inputs):
        inputs = inputs - np.max(inputs, axis=1)[:,np.newaxis]
        return np.exp(inputs) / np.sum(np.maximum(np.exp(inputs), 1e-15), axis=-1)[:,np.newaxis]
    
    def gradient(self, inputs, gradients):
        outputs = self(inputs)  # examples, classes
        examples, classes = outputs.shape
        diag = np.zeros((examples, classes, classes))  # examples, classes, classes
        diag[:, np.arange(classes), np.arange(classes)] = outputs # set the diagonal of each example
        my_gradient = diag - outputs[:,:,np.newaxis] * outputs[:,np.newaxis,:]  # examples, classes, classes
        return np.sum(gradients[:,np.newaxis,:] * my_gradient, axis=2) # examples, classes

    
class SigmoidLayer:
    def __init__(self):
        self.params = []
        self.grads = []
        
    def __call__(self, inputs):
        return 1 / (1 + np.exp(-inputs))
    
    def gradient(self, inputs, gradients):
        outputs = self(inputs)
        my_gradient = outputs * (1 - outputs)
        return my_gradient * gradients
    
    
class ReLULayer:
    def __init__(self):
        self.params = []
        self.grads = []
        
    def __call__(self, inputs):
        return np.maximum(inputs, 0)
    
    def gradient(self, inputs, gradients):
        return gradients * (inputs >= 0)
    
    
class DenseLayer:
    def __init__(self,inputs, outputs, random_seed=None):
        self._random_state = np.random.RandomState(random_seed)
        self._W = self._random_state.uniform(-2,2,size=(inputs, outputs))
        self._b = self._random_state.uniform(-2,2,size=(outputs,))
        self.params = [self._W, self._b]
        self.grads = [np.zeros_like(self._W), np.zeros_like(self._b)]
        self._cache = None
    
    def __call__(self, inputs):
        return inputs @ self._W + self._b[np.newaxis,:]
    
    def gradient(self, inputs, gradients):
        # create the cache
        if self._cache is None or self._cache.shape[0] != inputs.shape[0]:
            self._cache = np.ndarray((inputs.shape[0],inputs.shape[1], gradients.shape[1]))
        # gradient in respect to W
        w_grad = np.multiply(inputs[:,:,np.newaxis], gradients[:,np.newaxis,:], out=self._cache)  # examples, inputs, outputs
        np.add(self.grads[0], np.sum(w_grad, axis=0), out=self.grads[0])  # inputs, outputs
        # gradient in respect to b
        b_grad = gradients  # examples, outputs
        np.add(self.grads[1], np.sum(b_grad, axis=0), out=self.grads[1])  # outputs
        # gradient in respect to inputs
        in_grad = np.multiply(self._W[np.newaxis,:,:], gradients[:,np.newaxis,:], out=self._cache)  # examples, inputs, outputs
        #in_grad = np.add(in_grad, np.sign(self._b)[np.newaxis, np.newaxis, :], out=self._cache)  # examples, inputs, outputs
        return np.sum(in_grad, axis=2) # examples, inputs
