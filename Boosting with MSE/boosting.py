import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

class SimplifiedBoostingRegressor:
    def init(self):
        pass
        
    @staticmethod
    def loss(targets, predictions):
        loss = np.mean((targets - predictions)**2)
        return loss
    
    @staticmethod
    def loss_gradients(targets, predictions):
        gradients = 2 * (predictions - targets) / len(targets)
        assert gradients.shape == targets.shape
        return gradients
        
    def fit(self, model_constructor, data, targets, num_steps=10, lr=0.1, max_depth=5, verbose=False):
        new_targets = targets
        self.models_list = []
        self.lr = lr
        self.loss_log = []
        
        for step in range(num_steps):
            try:
                model = model_constructor(max_depth=max_depth)
            except TypeError:
                print('max_depth keyword is not found. Ignoring')
                model = model_constructor()
                
            fitted_model = model.fit(data, new_targets)
            self.models_list.append(fitted_model)
            
            predictions = self.predict(data)
            self.loss_log.append(self.loss(targets, predictions))
            gradients = self.loss_gradients(targets, predictions)
            new_targets = targets - predictions

            
        if verbose:
            print('Finished! Loss=', self.loss_log[-1])
        return self
            
    def predict(self, data):
        predictions = np.zeros(len(data))
        for model in self.models_list:
            predictions += self.lr * model.predict(data)
        return predictions
