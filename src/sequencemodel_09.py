import numpy as np
from progressbar import progressbar

class SequenceModel:
    def __init__(self, layers, loss, metrices = [], random_seed = None):
        self._rand = np.random.RandomState(random_seed);
        self.loss = loss
        self.metrices = metrices
        self.layers = layers
        
    def zero_grad(self):
        for layer in self.layers:
            for grad in layer.grads:
                grad.fill(0)
    
    def fit(self, X, y, optimizer, Xtest=None, ytest=None, epochs=100, batch_size=32, progress=False):
        # get how many data we have
        n_train_data = len(X)
        n_test_data = 1 if ytest is None else len(ytest)
        # store gradients and losses
        train_losses = np.zeros((epochs,))
        test_losses = np.zeros((epochs,))
        train_metrices = np.zeros((len(self.metrices), epochs))
        test_metrices = np.zeros((len(self.metrices), epochs))
        # decide whatever to log progress
        epoch_counter = progressbar(range(epochs)) if progress else range(epochs)
        # Learning
        outputs = [None] * (len(self.layers) + 1)
        for epoch in epoch_counter:
            # shuffle the data
            permutation = self._rand.permutation(n_train_data)
            # for each batch
            for batch_start in range(0, n_train_data, batch_size):
                # get batch
                batch_data = X[permutation[batch_start:batch_start+batch_size]]
                batch_target = y[permutation[batch_start:batch_start+batch_size]]
                # forward pass
                outputs[0] = batch_data
                for layer, i in zip(self.layers, range(len(self.layers))):
                    outputs[i+1] = layer(outputs[i])
                # backward pass
                self.zero_grad()
                current_grad = self.loss.gradient(batch_target, outputs[-1])
                for layer, layer_input in zip(self.layers[::-1], outputs[-2::-1]):
                    current_grad = layer.gradient(layer_input, current_grad)
                # update the weights
                optimizer.optim(self)
                # store loss
                train_losses[epoch] += np.sum(self.loss(batch_target, outputs[-1]))
                # compute the metrices
                for metric in self.metrices:
                    metric(batch_target, outputs[-1])
            # store train metrices
            for num_metric, metric in enumerate(self.metrices):
                train_metrices[num_metric, epoch] = metric.summary()
                
            # evaluate on the test set
            if Xtest is not None and ytest is not None:
                # for each batch
                for batch_start in range(0, n_test_data, batch_size):
                    # get batch
                    batch_data = Xtest[batch_start:batch_start+ batch_size]
                    batch_target = ytest[batch_start:batch_start + batch_size]
                    # predict the data
                    prediction = self.predict(batch_data)
                    # store loss
                    test_losses[epoch] += np.sum(self.loss(batch_target, prediction))
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
        for layer in self.layers:
            X = layer(X)
        return X
