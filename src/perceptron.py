import numpy as np
import sklearn.metrics

class Perceptron:
    def __init__(self, max_iters=100, random_state=None):
        self.max_iters = max_iters
        self.converged = False
        self._rand = np.random.RandomState(random_state)
        self._weights = None
        pass
    
    def _sign(self, vals):
        if np.isscalar(vals):
            return np.array([1 if vals >= 0 else 0])
        vals[vals >= 0] = 1
        vals[vals < 0] = 0
        return vals
    
    def fit(self, X, y):
        # Initialize the weights
        self.converged = False
        weights = self._rand.uniform(-2, 2, X.shape[1])
        
        best_correctly_classified = -1
        current_correctly_classified = -1
        best_weights = None

        old_weights = None
        for iteration in range(self.max_iters):
            # convergence check
            if (weights == old_weights).all():
                self.converged = True
                break
            old_weights = weights.copy()
            # data shuffle
            permutation = self._rand.permutation(len(y))
            # iterate over data
            for instance, target in zip(X[permutation], y[permutation]):
                # predict the output of the perceptron
                prediction = self._sign(instance @ weights)
                # update the weights
                weights = weights + (target - prediction) * instance
                # pocket algorithm
                current_correctly_classified = current_correctly_classified + 1 if target == prediction else 0
                if current_correctly_classified > best_correctly_classified:
                    best_correctly_classified = current_correctly_classified
                    best_weights = weights.copy()
        # store the best weights
        self._weights = best_weights
        return self
    
    def predict(self, X):
        return self._sign(X @ self._weights)
    
    def __call__(self, X):
        return self.predict(X)
    
    
def multiclass_perceptron(X, y, Xtest, ytest, iters=500, random_state=42):
    # train models
    models = np.empty((10,10), dtype=object)
    for i in range(10):
        for j in range(i):
            models[i][j] = Perceptron(max_iters=iters, random_state=random_state + i * 10 + j)
            mask = np.logical_or(y == i, y == j)
            current_X = X[mask]
            current_y = (y[mask] - j) / (i - j)
            models[i][j].fit(current_X, current_y)
            
    # predict
    train_predictions = np.zeros((y.shape[0], 10), dtype=int)
    test_predictions = np.zeros((ytest.shape[0], 10), dtype=int)
    for i in range(10):
        for j in range(i):
            prediction = models[i][j].predict(X)
            train_predictions[prediction == 0, j] += 1
            train_predictions[prediction == 1, i] += 1
            prediction = models[i][j].predict(Xtest)
            test_predictions[prediction == 0, j] += 1
            test_predictions[prediction == 1, i] += 1
    train_predictions = train_predictions.argmax(axis=1)
    test_predictions = test_predictions.argmax(axis=1)
    
    return (
        sklearn.metrics.accuracy_score(y, train_predictions), 
        sklearn.metrics.accuracy_score(ytest, test_predictions), 
    )
