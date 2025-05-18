from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class RandomForest:
    def __init__(self, config):
        # Khởi tạo mô hình sklearn
        self.rf = RandomForestClassifier(
            n_estimators = config.n_estimators,
            max_depth = config.max_depth,
            random_state = config.random_state
        )

    def fit(self, X, y):
        self.rf.fit(X, y)
        print("Training completed.")

        # Tính accuracy trên tập train
        y_pred = self.rf.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"Training accuracy: {acc:.4f}")

        return self.rf

    def predict(self, X):
        return self.rf.predict(X)

    def save(self, path):
        joblib.dump(self.rf, path)

    def load(self, path):
        self.rf = joblib.load(path)
