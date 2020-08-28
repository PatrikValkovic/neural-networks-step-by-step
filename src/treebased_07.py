import numpy as np
from .neuron_05 import Neuron

class TreeInOrder:
    def __init__(self, loss, metrices=[], epochs=100, random_state=None, learning_rate=0.001, batch_size=16):
        self.loss = loss
        self.metrices = metrices
        self.epochs = epochs
        self.random_state = np.random.RandomState(random_state)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # models
        self.zerotofiveVSfivetoten = None
        self.zerotothreeVSthreetofive = None
        self.fivetoeightVSeighttoten = None
        self.zerototwoVStwo = None
        self.fivetosevenVSseven = None
        self.zeroVSone = None
        self.threeVSfour = None
        self.fiveVSsix = None
        self.eightVSnine = None
        
    def fit(self, X, y, progress=False):
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 0, target == 1, target == 2, target == 3, target == 4), axis=0)
        mask_ones = np.any((target == 5, target == 6, target == 7, target == 8, target == 9), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.zerotofiveVSfivetoten = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.zerotofiveVSfivetoten.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 0, target == 1, target == 2), axis=0)
        mask_ones = np.any((target == 3, target == 4), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.zerotothreeVSthreetofive = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.zerotothreeVSthreetofive.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 5, target == 6, target == 7), axis=0)
        mask_ones = np.any((target == 8, target == 9), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.fivetoeightVSeighttoten = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.fivetoeightVSeighttoten.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 0, target == 1), axis=0)
        mask_ones = np.any((target == 2,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.zerototwoVStwo = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.zerototwoVStwo.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 5, target == 6), axis=0)
        mask_ones = np.any((target == 7,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.fivetosevenVSseven = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.fivetosevenVSseven.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 0,), axis=0)
        mask_ones = np.any((target == 1,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.zeroVSone = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.zeroVSone.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 3,), axis=0)
        mask_ones = np.any((target == 4,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.threeVSfour = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.threeVSfour.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 5,), axis=0)
        mask_ones = np.any((target == 6,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.fiveVSsix = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.fiveVSsix.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 8,), axis=0)
        mask_ones = np.any((target == 9,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.eightVSnine = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.eightVSnine.fit(data, target, progress=progress)

    def predict_probbased(self, X):
        probs = np.ones((len(X),10))
        pred = self.zerotofiveVSfivetoten.predict(X)
        probs[:,0:5] *= 1 - pred[:, np.newaxis]
        probs[:,5:10] *= pred[:, np.newaxis]
        pred = self.zerotothreeVSthreetofive.predict(X)
        probs[:,0:3] *= 1 - pred[:, np.newaxis]
        probs[:,3:5] *= pred[:, np.newaxis]
        pred = self.fivetoeightVSeighttoten.predict(X)
        probs[:,5:8] *= 1 - pred[:, np.newaxis]
        probs[:,8:10] *= pred[:, np.newaxis]
        pred = self.zerototwoVStwo.predict(X)
        probs[:,0:2] *= 1 - pred[:, np.newaxis]
        probs[:,2] *= pred
        pred = self.fivetosevenVSseven.predict(X)
        probs[:,5:7] *= 1 - pred[:, np.newaxis]
        probs[:,7] *= pred
        pred = self.zeroVSone.predict(X)
        probs[:,0] *= 1 - pred
        probs[:,1] *= pred
        pred = self.threeVSfour.predict(X)
        probs[:,3] *= 1 - pred
        probs[:,4] *= pred
        pred = self.fiveVSsix.predict(X)
        probs[:,5] *= 1 - pred
        probs[:,6] *= pred
        pred = self.eightVSnine.predict(X)
        probs[:,8] *= 1 - pred
        probs[:,9] *= pred
        return np.argmax(probs, axis=1)
    
    def predict_direct(self, X):
        result = np.ones((len(X),)) * -1
        pred1 = self.zerotofiveVSfivetoten.predict(X)
        pred2 = self.zerotothreeVSthreetofive.predict(X)
        pred3 = self.fivetoeightVSeighttoten.predict(X)
        pred4 = self.zerototwoVStwo.predict(X)
        pred5 = self.fivetosevenVSseven.predict(X)
        pred6 = self.zeroVSone.predict(X)
        pred7 = self.threeVSfour.predict(X)
        pred8 = self.fiveVSsix.predict(X)
        pred9 = self.eightVSnine.predict(X)
        result[np.all([pred1 < 0.5, pred2 < 0.5, pred4 < 0.5, pred6 < 0.5], axis=0)] = 0
        result[np.all([pred1 < 0.5, pred2 < 0.5, pred4 < 0.5, pred6 >= 0.5], axis=0)] = 1
        result[np.all([pred1 < 0.5, pred2 < 0.5, pred4 >= 0.5], axis=0)] = 2
        result[np.all([pred1 < 0.5, pred2 >= 0.5, pred7 < 0.5], axis=0)] = 3
        result[np.all([pred1 < 0.5, pred2 >= 0.5, pred7 >= 0.5], axis=0)] = 4
        result[np.all([pred1 >= 0.5, pred3 < 0.5, pred5 < 0.5, pred8 < 0.5], axis=0)] = 5
        result[np.all([pred1 >= 0.5, pred3 < 0.5, pred5 < 0.5, pred8 >= 0.5], axis=0)] = 6
        result[np.all([pred1 >= 0.5, pred3 < 0.5, pred5 >= 0.5], axis=0)] = 7
        result[np.all([pred1 >= 0.5, pred3 >= 0.5, pred9 < 0.5], axis=0)] = 8
        result[np.all([pred1 >= 0.5, pred3 >= 0.5, pred9 >= 0.5], axis=0)] = 9
        return result
    

class TreeSpecialOrder:
    def __init__(self, loss, metrices=[], epochs=100, random_state=None, learning_rate=0.001, batch_size=16):
        self.loss = loss
        self.metrices = metrices
        self.epochs = epochs
        self.random_state = np.random.RandomState(random_state)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # models
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.model4 = None
        self.model5 = None
        self.model6 = None
        self.model7 = None
        self.model8 = None
        self.model9 = None
        
    def fit(self, X, y, progress=False):
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 1, target == 7, target == 4, target == 2, target == 3), axis=0)
        mask_ones = np.any((target == 5, target == 9, target == 0, target == 6, target == 8), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.model1 = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.model1.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 1, target == 7, target == 4), axis=0)
        mask_ones = np.any((target == 2, target == 3), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.model2 = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.model2.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 5, target == 9), axis=0)
        mask_ones = np.any((target == 0, target == 6, target == 8), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.model3 = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.model3.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 1, target == 7), axis=0)
        mask_ones = np.any((target == 4,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.model4 = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.model4.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 2,), axis=0)
        mask_ones = np.any((target == 3,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.model5 = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.model5.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 5,), axis=0)
        mask_ones = np.any((target == 9,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.model6 = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.model6.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 0,), axis=0)
        mask_ones = np.any((target == 6, target == 8), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.model7 = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.model7.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 1,), axis=0)
        mask_ones = np.any((target == 7,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.model8 = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.model8.fit(data, target, progress=progress)
        # model
        data = X.copy()
        target = y.copy()
        mask_zeros = np.any((target == 6,), axis=0)
        mask_ones = np.any((target == 8,), axis=0)
        data = data[mask_zeros | mask_ones]
        target[mask_zeros] = 0
        target[mask_ones] = 1
        target = target[mask_zeros | mask_ones]
        self.model9 = Neuron(self.loss, self.metrices, self.epochs, self.random_state.randint(0, 1000), self.learning_rate, self.batch_size)
        self.model9.fit(data, target, progress=progress)

    def predict_probbased(self, X):
        probs = np.ones((len(X),10))
        pred = self.model1.predict(X)
        probs[:,(1,7,4,2,3)] *= 1 - pred[:, np.newaxis]
        probs[:,(5,9,0,6,8)] *= pred[:, np.newaxis]
        pred = self.model2.predict(X)
        probs[:,(1,7,4)] *= 1 - pred[:, np.newaxis]
        probs[:,(2,3)] *= pred[:, np.newaxis]
        pred = self.model3.predict(X)
        probs[:,(5,9)] *= 1 - pred[:, np.newaxis]
        probs[:,(0,6,8)] *= pred[:, np.newaxis]
        pred = self.model4.predict(X)
        probs[:,(1,7)] *= 1 - pred[:, np.newaxis]
        probs[:,4] *= pred
        pred = self.model5.predict(X)
        probs[:,2] *= 1 - pred
        probs[:,3] *= pred
        pred = self.model6.predict(X)
        probs[:,5] *= 1 - pred
        probs[:,9] *= pred
        pred = self.model7.predict(X)
        probs[:,0] *= 1 - pred
        probs[:,(6,8)] *= pred[:, np.newaxis]
        pred = self.model8.predict(X)
        probs[:,1] *= 1 - pred
        probs[:,7] *= pred
        pred = self.model9.predict(X)
        probs[:,6] *= 1 - pred
        probs[:,8] *= pred
        return np.argmax(probs, axis=1)
    
    def predict_direct(self, X):
        result = np.ones((len(X),)) * -1
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        pred3 = self.model3.predict(X)
        pred4 = self.model4.predict(X)
        pred5 = self.model5.predict(X)
        pred6 = self.model6.predict(X)
        pred7 = self.model7.predict(X)
        pred8 = self.model8.predict(X)
        pred9 = self.model9.predict(X)
        result[np.all([pred1 < 0.5, pred2 < 0.5, pred4 < 0.5, pred8 < 0.5], axis=0)] = 1
        result[np.all([pred1 < 0.5, pred2 < 0.5, pred4 < 0.5, pred8 >= 0.5], axis=0)] = 7
        result[np.all([pred1 < 0.5, pred2 < 0.5, pred4 >= 0.5], axis=0)] = 4
        result[np.all([pred1 < 0.5, pred2 >= 0.5, pred5 < 0.5], axis=0)] = 2
        result[np.all([pred1 < 0.5, pred2 >= 0.5, pred5 >= 0.5], axis=0)] = 3
        result[np.all([pred1 >= 0.5, pred3 < 0.5, pred6 < 0.5], axis=0)] = 5
        result[np.all([pred1 >= 0.5, pred3 < 0.5, pred6 >= 0.5], axis=0)] = 9
        result[np.all([pred1 >= 0.5, pred3 >= 0.5, pred7 < 0.5], axis=0)] = 0
        result[np.all([pred1 >= 0.5, pred3 >= 0.5, pred7 >= 0.5, pred9 < 0.5], axis=0)] = 6
        result[np.all([pred1 >= 0.5, pred3 >= 0.5, pred7 >= 0.5, pred9 >= 0.5], axis=0)] = 8
        return result
