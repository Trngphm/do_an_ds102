from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from builders.model_builder import META_ARCHITECTURE  
import joblib

@META_ARCHITECTURE.register()
class LogisticRegression:
    def __init__(self, config):
        self.config = config
        self.lr = LogisticRegression(
            C=getattr(config, 'C', 1.0),
            penalty=getattr(config, 'penalty', 'l2'),
            solver=getattr(config, 'solver', 'lbfgs'),
            max_iter=getattr(config, 'max_iter', 1000),
            random_state=42
        )
    
    def get_base_model(self):
        return LogisticRegression(random_state=42, max_iter=1000)
    
    def update_params(self, params):
        self.lr = LogisticRegression(**params, random_state=42, max_iter=1000)
    
    def set_model(self, model):
        self.lr = model
    
    def fit(self, X, y):
        self.lr.fit(X, y)
        print("Logistic Regression Training completed.")
        print(f"Number of iterations: {self.lr.n_iter_}")
        
        return self.lr
    
    def predict(self, X):
        return self.lr.predict(X)
    
    def save(self, path):
        joblib.dump(self.lr, path)
    
    def load(self, path):
        self.lr = joblib.load(path)