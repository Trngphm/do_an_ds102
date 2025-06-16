from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from builders.model_builder import META_ARCHITECTURE
import joblib

@META_ARCHITECTURE.register()
class SVM:
    def __init__(self, config):
        self.config = config
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel=getattr(config, 'kernel', 'rbf'),
                C=getattr(config, 'C', 1.0),
                gamma=getattr(config, 'gamma', 'scale'),
                random_state=42
            ))
        ])
    
    def get_base_model(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42))
        ])
    
    def update_params(self, params):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**params, random_state=42))
        ])
        self.model = pipeline
    
    def set_model(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        print("SVM Training completed.")
        svm = self.model.named_steps['svm']
        print(f"Number of support vectors for each class: {svm.n_support_}")
        print(f"Total support vectors: {sum(svm.n_support_)}")
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)
