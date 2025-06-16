from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from builders.model_builder import META_ARCHITECTURE
import joblib

@META_ARCHITECTURE.register()
class MyLogisticRegression:
    def __init__(self, config):
        self.config = config
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', SklearnLogisticRegression(
                C=getattr(config, 'C', 1.0),
                penalty=getattr(config, 'penalty', 'l2'),
                solver=getattr(config, 'solver', 'lbfgs'),
                max_iter=getattr(config, 'max_iter', 1000),
                random_state=42
            ))
        ])
    
    def get_base_model(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('lr', SklearnLogisticRegression(random_state=42, max_iter=1000))
        ])
    
    def update_params(self, params):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', SklearnLogisticRegression(**params, random_state=42, max_iter=1000))
        ])
        self.model = pipeline
    
    def set_model(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        print("Logistic Regression Training completed.")
        print(f"Number of iterations: {self.model.named_steps['lr'].n_iter_}")
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)
