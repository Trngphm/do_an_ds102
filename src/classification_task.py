import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from tqdm import tqdm
from builders.task_builder import META_TASK
from builders.model_builder import build_model
import matplotlib.pyplot as plt
import joblib
import time
import numpy as np

@META_TASK.register()
class MLClassificationTask():
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.checkpoint_path = config.checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.log_file = os.path.join(self.checkpoint_path, "task.log")
        self._init_log()

        self.train_path = config.dataset.train.path
        self.test_path = config.dataset.test.path
        self.dev_path = config.dataset.dev.path

        self.model = build_model(config.model)
        self.best_params = None
        self.best_score = None
        self.tuning_results = {}

        self.load_datasets()

    def _init_log(self):
        with open(self.log_file, "w") as f:
            f.write(f"Log file created at {time.ctime()}\n")

    def log(self, message: str):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        print(f"{timestamp} {message}")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} {message}\n")

    def load_datasets(self):
        self.train_df = pd.read_csv(self.train_path)
        self.dev_df = pd.read_csv(self.dev_path)
        self.test_df = pd.read_csv(self.test_path)
        self.log("Loaded datasets successfully.")

    def tune_hyperparameters(self):
        X_train_full = self.train_df.drop("label", axis=1)
        y_train_full = self.train_df["label"]

        self.log("Starting hyperparameter tuning...")
        self.log(f"Full training data shape: {X_train_full.shape}")

        if len(X_train_full) > 5000:
            self.log("Sampling 5,000 samples for hyperparameter tuning to reduce memory usage...")
            sampled = self.train_df.sample(n=5000, random_state=42, stratify=self.train_df["label"])
            X_train = sampled.drop("label", axis=1)
            y_train = sampled["label"]
        else:
            X_train, y_train = X_train_full, y_train_full

        self.log(f"Subset training data shape: {X_train.shape}")

        base_model = self.model.get_base_model()
        tuning_method = self.config.model.params.get('tuning_method', 'grid')
        raw_param_grid = {
            k: v for k, v in self.config.model.params.items() if k != 'tuning_method'
        }

        # Tự động kết hợp các combination hợp lệ nếu là LogisticRegression
        model_name = self.config.model.name
        if model_name == 'LogisticRegression':
            param_grid = []
            for C in raw_param_grid.get('C', [1.0]):
                for penalty in raw_param_grid.get('penalty', ['l2']):
                    for solver in raw_param_grid.get('solver', ['lbfgs']):
                        # Check parameter compatibility
                        if penalty == 'l1' and solver in ['newton-cg', 'lbfgs']:
                            continue  # l1 penalty not supported with these solvers
                        if penalty == 'elasticnet' and solver != 'saga':
                            continue  # elasticnet only supported with saga
                        param_grid.append({'C': C, 'penalty': penalty, 'solver': solver})
            self.log(f"Filtered valid hyperparameter combinations: {len(param_grid)}")
        else:
            param_grid = raw_param_grid


        self.log(f"Tuning method: {tuning_method}")
        self.log(f"Parameter grid: {param_grid}")

        if tuning_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=1,
                verbose=1
            )
        elif tuning_method == 'random':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=50,
                cv=5,
                scoring='accuracy',
                n_jobs=1,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported tuning method: {tuning_method}")

        start_time = time.time()
        search.fit(X_train, y_train)
        end_time = time.time()

        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.tuning_results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'tuning_time': end_time - start_time,
            'cv_results': search.cv_results_
        }

        # Save full cv_results_
        cv_results_path = os.path.join(self.checkpoint_path, "cv_results.json")
        cv_serializable = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in search.cv_results_.items()
        }
        with open(cv_results_path, "w") as f:
            json.dump(cv_serializable, f, indent=4)
        self.log(f"Saved full cross-validation results to {cv_results_path}")

        self.log(f"Tuning completed in {end_time - start_time:.2f} seconds")
        self.log(f"Best parameters: {self.best_params}")
        self.log(f"Best cross-validation score: {self.best_score:.4f}")

        self.model.update_params(self.best_params)
        return search.best_estimator_

    def train(self):
        X_train = self.train_df.drop("label", axis=1)
        y_train = self.train_df["label"]

        if hasattr(self.config.model, 'params') and len(self.config.model.params) > 1:
            self.log("Performing hyperparameter tuning...")
            best_model = self.tune_hyperparameters()
            self.model.set_model(best_model)

        self.log("Training final model with best parameters...")
        self.model.fit(X_train, y_train)
        self.log("Model training completed.")

        self.save_tuning_results()

    def save_tuning_results(self):
        if self.tuning_results:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            results_path = os.path.join(self.checkpoint_path, "tuning_results.json")

            serializable_results = {}
            for key, value in self.tuning_results.items():
                if key == 'cv_results':
                    continue
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_results[key] = value.item()
                else:
                    serializable_results[key] = value

            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            self.log(f"Tuning results saved to {results_path}")

    def evaluate(self, split="test"):
        if split == "dev":
            X = self.dev_df.drop("label", axis=1)
            y = self.dev_df["label"]
            self.log("Evaluating on dev set...")
        elif split == "test":
            X = self.test_df.drop("label", axis=1)
            y = self.test_df["label"]
            self.log("Predicting test set...")
        else:
            raise ValueError("split must be either 'dev' or 'test'")

        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')

        report = classification_report(y, y_pred, output_dict=True)

        self.log(f"{split.capitalize()} Accuracy: {acc:.4f}")
        self.log(f"{split.capitalize()} Precision (macro): {precision:.4f}")
        self.log(f"{split.capitalize()} Recall (macro): {recall:.4f}")
        self.log(f"{split.capitalize()} F1-score (macro): {f1:.4f}")
        self.log(f"{split.capitalize()} Classification Report:\n{json.dumps(report, indent=2)}")

        # Save classification report
        report_path = os.path.join(self.checkpoint_path, f"{split}_classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        self.log(f"Saved {split} classification report to {report_path}")

        # Save into attributes
        if split == "dev":
            self.dev_accuracy = acc
            self.dev_precision = precision
            self.dev_recall = recall
            self.dev_f1 = f1
            self.dev_report = report
        else:
            self.accuracy = acc
            self.precision = precision
            self.recall = recall
            self.f1 = f1
            self.report = report

        # Save metrics to file
        metrics = {
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        }
        metrics_path = os.path.join(self.checkpoint_path, f"{split}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.log(f"Saved {split} metrics to {metrics_path}")

        return acc, f1, report

    def save_model(self):
        save_path = os.path.join(self.checkpoint_path, "best_model.joblib")
        joblib.dump(self.model, save_path)
        self.log(f"Model saved to {save_path}")

    def load_model(self):
        load_path = os.path.join(self.checkpoint_path, "best_model.joblib")
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Model checkpoint not found: {load_path}")
        self.model = joblib.load(load_path)
        self.log(f"Loaded model from {load_path}")

    def get_predictions(self):
        self.load_model()
        X_test = self.test_df.drop("label", axis=1)
        y_test = self.test_df["label"]

        y_pred = self.model.predict(X_test)
        results = {
            i: {"prediction": int(pred), "target": int(y_test.iloc[i])}
            for i, pred in enumerate(y_pred)
        }

        json_path = os.path.join(self.checkpoint_path, "predictions.json")
        json.dump(results, open(json_path, "w+"), indent=4)
        self.log(f"Saved predictions to {json_path}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()

        cm_path = os.path.join(self.checkpoint_path, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        self.log(f"Saved confusion matrix to {cm_path}")

        return results
