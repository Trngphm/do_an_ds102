from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from builders.model_builder import META_ARCHITECTURE  


@META_ARCHITECTURE.register()
class SVM:
    def __init__(self, config):
        kernel = getattr(config.model, "kernel", "rbf")
        C = getattr(config.model, "C", 1.0)

        # Khởi tạo mô hình sklearn
        self.model = SVC(kernel=kernel, C=C)

    from sklearn.metrics import accuracy_score

    def fit(self, X, y):
        print("Starting training SVM model...")
        self.model.fit(X, y)
        print("Training completed.")

        # In thông tin về support vectors
        print(f"Number of support vectors for each class: {self.model.n_support_}")

        # Tính accuracy trên tập train ngay sau khi fit (optional)
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
