from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from builders.model_builder import META_ARCHITECTURE  
import joblib


@META_ARCHITECTURE.register()
class SVM:
    def __init__(self, config):
        # Khởi tạo mô hình sklearn
        self.svm = SVC(kernel = config.kernel, C = config.C)

    def fit(self, X, y):
        self.svm.fit(X, y)
        print("Training completed.")

        # In thông tin về support vectors
        print(f"Number of support vectors for each class: {self.svm.n_support_}")

        # Tính accuracy trên tập train ngay sau khi fit (optional)
        y_pred = self.svm.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"Training accuracy: {acc:.4f}")

        return self.svm

    def predict(self, X):
        return self.svm.predict(X)

    def save(self, path):
        joblib.dump(self.svm, path)

    def load(self, path):
        self.svm = joblib.load(path)
