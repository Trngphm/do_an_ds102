from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from builders.model_builder import META_ARCHITECTURE
import numpy as np

@META_ARCHITECTURE.register()
class RandomForest:
    def __init__(self, config):
        self.config = config
        self.rf = RandomForestClassifier(
            n_estimators=getattr(config, 'n_estimators', 100),
            max_depth=getattr(config, 'max_depth', None),
            min_samples_split=getattr(config, 'min_samples_split', 2),
            min_samples_leaf=getattr(config, 'min_samples_leaf', 1),
            random_state=42
        )
    
    def get_base_model(self):
        return RandomForestClassifier(random_state=42)
    
    def update_params(self, params):
        self.rf = RandomForestClassifier(**params, random_state=42)
    
    def set_model(self, model):
        self.rf = model
    
    def fit(self, X, y):
        self.rf.fit(X, y)
        print("Random Forest Training completed.")
        print(f"Number of trees: {self.rf.n_estimators}")
        
        # Feature importance
        if hasattr(self.rf, 'feature_importances_'):
            feature_importance = self.rf.feature_importances_
            print(f"Top 5 most important features: {np.argsort(feature_importance)[-5:]}")
        
        return self.rf
    
    def predict(self, X):
        return self.rf.predict(X)
    
    def save(self, path):
        joblib.dump(self.rf, path)
    
    def load(self, path):
        self.rf = joblib.load(path)
