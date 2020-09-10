import numpy as np


class CategoricalCrossEntropyLoss:
    def __call__(self, target, predicted):
        indices = np.arange(len(target))
        return -np.log(np.maximum(predicted[indices,target], 1e-15))
    
    def gradient(self, target, predicted):
        grad = np.zeros((len(target), 10))
        indices = np.arange(len(target))
        grad[indices,target] = -1 / np.maximum(predicted[indices,target], 1e-15)
        return grad
    
    
class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def optim(self, model):
        for layer in model.layers:
            for params, grad in zip(layer.params, layer.grads):
                np.add(params, - self.learning_rate * grad, out=params)

                
class AccuracyMetric:
    def __init__(self, name=None):
        self.name = name or "accuracy"
        self.correct = 0
        self.num = 0
    def __call__(self, target, predicted):
        self.correct += np.sum(np.argmax(predicted,axis=1) == target)
        self.num += len(target)
    def summary(self):
        acc = self.correct / self.num
        self.correct = 0
        self.num = 0
        return acc
