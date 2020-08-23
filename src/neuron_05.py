import numpy as np
from progressbar import progressbar

class Neuron:
    def __init__(self, loss, metrices=[], epochs=100, random_state=None, learning_rate=0.001, batch_size=16):
        self.epochs = epochs
        self.loss = loss
        self.metrices = metrices
        self._rand = np.random.RandomState(random_state)
        self._weights = None
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        pass
    
    def _activation(self, vals):  # activation function
        return 1 / (1 + np.exp(-vals))
    
    def fit(self, X, y, Xtest=None, ytest=None, progress=False):
        # Initialize the weights
        self._weights = self._rand.uniform(-2, 2, X.shape[1])
        # get how many data we have
        n_train_data = len(X)
        n_test_data = 1 if ytest is None else len(ytest)
        # store gradients and losses
        train_losses = np.zeros((self.epochs,))
        test_losses = np.zeros((self.epochs,))
        train_metrices = np.zeros((len(self.metrices), self.epochs))
        test_metrices = np.zeros((len(self.metrices), self.epochs))
        # decide whatever to log progress
        epoch_counter = progressbar(range(self.epochs)) if progress else range(self.epochs)
        # Learning
        for epoch in epoch_counter:
            # shuffle the data
            permutation = self._rand.permutation(n_train_data)
            # for each batch
            for batch_start in range(0, n_train_data, self._batch_size):
                # get batch
                batch_data = X[permutation[batch_start:batch_start+self._batch_size]]
                batch_target = y[permutation[batch_start:batch_start+self._batch_size]]
                # predict the data
                prediction = self.predict(batch_data)
                # table of gradient for each weights and gradient in the shape (samples,weights)
                gradient = np.reshape(self.loss.gradient(batch_target, prediction) * (prediction * (1 - prediction)), newshape=(-1,1)) * batch_data
                # sum gradient over the samples
                op_gradient = self._learning_rate * np.sum(gradient, axis=0)
                # store loss
                train_losses[epoch] += self.loss(batch_target, prediction)
                # update the weights
                self._weights = self._weights - op_gradient
                # transform probs to predictions
                prediction[prediction < 0.5] = 0
                prediction[prediction >= 0.5] = 1
                # compute the metrices
                for metric in self.metrices:
                    metric(batch_target, prediction)
            # store train metrices
            for num_metric, metric in enumerate(self.metrices):
                train_metrices[num_metric, epoch] = metric.summary()
                
            # evaluate on the test set
            if Xtest is not None and ytest is not None:
                # for each batch
                for batch_start in range(0, n_test_data, self._batch_size):
                    # get batch
                    batch_data = Xtest[batch_start:batch_start+self._batch_size]
                    batch_target = ytest[batch_start:batch_start+self._batch_size]
                    # predict the data
                    prediction = self.predict(batch_data)
                    # store loss
                    test_losses[epoch] += self.loss(batch_target, prediction)
                    # transform probs to predictions
                    prediction[prediction < 0.5] = 0
                    prediction[prediction >= 0.5] = 1
                    # compute the metrices
                    for metric in self.metrices:
                        metric(batch_target, prediction)
                # store test metrices
                for num_metric, metric in enumerate(self.metrices):
                    test_metrices[num_metric, epoch] = metric.summary()
          
        results = {
            "train_loss": train_losses / n_train_data, 
            "test_loss": test_losses / n_test_data,      
        }
        results.update({f"train_{metric.name}": train_metrices[num_metric] for num_metric in range(len(self.metrices))})
        results.update({f"test_{metric.name}": test_metrices[num_metric] for num_metric in range(len(self.metrices))})
        return results
    
    def predict(self, X):
        return self._activation(X @ self._weights)
        
    def predict_classes(self, X):
        prediction = self.predict(X)
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
    
    def __call__(self, X):
        return self.predict(X)