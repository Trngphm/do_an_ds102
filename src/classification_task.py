import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from builders.task_builder import META_TASK
from builders.model_builder import build_model
import matplotlib.pyplot as plt


@META_TASK.register()
class MLClassificationTask():
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.checkpoint_path = config.checkpoint_path
        self.train_path = config.dataset.train.path
        self.test_path = config.dataset.test.path

        self.model = build_model(config)  
        self.load_datasets()

    def load_datasets(self):
        # Load csv train/test vào pandas dataframe
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

    def train(self):
        X_train = self.train_df.drop("label", axis=1)
        y_train = self.train_df["label"]

        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Model training completed.")

    def evaluate(self):
        X_test = self.test_df.drop("label", axis=1)
        y_test = self.test_df["label"]

        print("Predicting test set...")
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Accuracy: {acc:.4f}")
        # Có thể in hoặc log report chi tiết
        self.report = report
        self.accuracy = acc

        # Tạo và lưu ma trận nhầm lẫn
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()

        cm_path = os.path.join(self.checkpoint_path, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix to {cm_path}")
        return acc, report

    def save_model(self):
        import joblib
        os.makedirs(self.checkpoint_path, exist_ok=True)
        save_path = os.path.join(self.checkpoint_path, "best_model.joblib")
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self):
        import joblib
        load_path = os.path.join(self.checkpoint_path, "best_model.joblib")
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Model checkpoint not found: {load_path}")
        self.model = joblib.load(load_path)
        print(f"Loaded model from {load_path}")

    def get_predictions(self):
        self.load_model()
        X_test = self.test_df.drop("label", axis=1)
        y_test = self.test_df["label"]

        y_pred = self.model.predict(X_test)
        results = {}
        for i, pred in enumerate(y_pred):
            results[i] = {
                "prediction": int(pred),
                "target": int(y_test.iloc[i])
            }


        # Lưu kết quả ra file json
        json.dump(results, open(os.path.join(self.checkpoint_path, "predictions.json"), "w+"), indent=4)
        print("Saved predictions to predictions.json")
        return results
