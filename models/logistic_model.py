from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from builders.model_builder import META_ARCHITECTURE  


@META_ARCHITECTURE.register()
class Logistic_Regression:
    def __init__(self, config):
        # Khởi tạo mô hình Logistic Regression
        self.model = LogisticRegression(
            penalty = config.penalty,
            C = config.C,
            solver = config.solver,
            max_iter = config.max_iter
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        print("Training completed.")

        # Tính accuracy trên tập train
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"Training accuracy: {acc:.4f}")

        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)
