from sklearn.base import TransformerMixin

class WrappedPredictor(TransformerMixin):
    def __init__(self, predictor):
        self.predictor = predictor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.predictor.predict_proba(X)[:,1].reshape(-1,1)

class SummedPredictors:
    def __init__(self, predictors, weights):
        self.predictors = predictors
        self.weights = weights

    def predict_proba(self, X):
        pos_prob = None
        for p, w in zip(self.predictors, self.weights):
            if pos_prob is None:
                pos_prob = p.predict_proba(X)
            else:
                pos_prob += p.predict_proba(X)
        pos_prob[:,0] = 1.0 - pos_prob[:,1]
        return pos_prob
