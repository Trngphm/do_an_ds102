from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from builders.model_builder import META_ARCHITECTURE  
import joblib


@META_ARCHITECTURE.register()
class SVM:
    def __init__(self, config):
        # Default parameters
        self.config = config
        self.svm = SVC(
            kernel=getattr(config, 'kernel', 'rbf'),
            C=getattr(config, 'C', 1.0),
            gamma=getattr(config, 'gamma', 'scale'),
            random_state=42
        )
    
    def get_base_model(self):
        """Return a fresh instance of the base model for hyperparameter tuning"""
        return SVC(random_state=42)
    
    def update_params(self, params):
        """Update model parameters after tuning"""
        self.svm = SVC(**params, random_state=42)
    
    def set_model(self, model):
        """Set the model instance"""
        self.svm = model
    
    def fit(self, X, y):
        self.svm.fit(X, y)
        print("SVM Training completed.")
        
        # Print information about support vectors
        print(f"Number of support vectors for each class: {self.svm.n_support_}")
        print(f"Total support vectors: {sum(self.svm.n_support_)}")
        
        return self.svm
    
    def predict(self, X):
        return self.svm.predict(X)
    
    def save(self, path):
        joblib.dump(self.svm, path)
    
    def load(self, path):
        self.svm = joblib.load(path)