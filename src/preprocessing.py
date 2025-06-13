import os
import zipfile
import numpy as np
import pandas as pd
from io import TextIOWrapper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import RandomOverSampler
import csv
from builders.task_builder import META_TASK

@META_TASK.register()
class Preprocessing():
    def __init__(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/raw_data/"))
        self.zip_path = os.path.join(self.data_dir, "AwA2-features.zip")
        self.npy_path = os.path.join(self.data_dir, "AwA2-features.npy")
        self.classes_txt_path = os.path.join(self.data_dir, "classes.txt")
        self.zip_internal_dir = "Animals_with_Attributes2/Features/ResNet101/"
        self.features_txt = self.zip_internal_dir + "AwA2-features.txt"
        self.labels_txt = self.zip_internal_dir + "AwA2-labels.txt"
        self.filenames_txt = self.zip_internal_dir + "AwA2-filenames.txt"

    def forward(self):
        # 1. Load features
        if not os.path.exists(self.npy_path):
            print("Đang đọc features từ .zip...")
            features = []
            with zipfile.ZipFile(self.zip_path) as zf:
                with zf.open(self.features_txt) as f:
                    for line in TextIOWrapper(f, encoding='utf-8'):
                        features.append([float(x) for x in line.strip().split()])
            features = np.array(features)
            np.save(self.npy_path, features)
        else:
            features = np.load(self.npy_path)

        # 2. Load labels & filenames
        with zipfile.ZipFile(self.zip_path) as zf:
            with zf.open(self.labels_txt) as f:
                labels = np.loadtxt(f).astype(int) - 1
            with zf.open(self.filenames_txt) as f:
                filenames = [line.strip() for line in TextIOWrapper(f, encoding='utf-8')]

        # 3. Load class names
        class_map = pd.read_csv(self.classes_txt_path, sep="\t", header=None, names=['id', 'name'])
        class_map['name'] = class_map['name'].str.replace('+', ' ', regex=False)
        class_map['id'] = class_map['id'] - 1
        id2name = dict(zip(class_map['id'], class_map['name']))

        # 4. Tạo DataFrame gốc
        feature_cols = [f"f{i}" for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=feature_cols)
        df['label'] = labels
        df['filename'] = filenames
        df['class_name'] = df['label'].map(id2name)

        # 5. Chia dữ liệu
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

        # 6. Chuẩn hóa
        print("Đang chuẩn hóa bằng PowerTransformer...")
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        X_train_scaled = scaler.fit_transform(train_df[feature_cols])
        X_val_scaled = scaler.transform(val_df[feature_cols])
        X_test_scaled = scaler.transform(test_df[feature_cols])

        # 7. Oversample trên tập train
        print("Đang oversample tập train để cân bằng lớp...")
        y_train = train_df['label']
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)

        # Lấy đúng filename từ chỉ số được oversample
        original_indices = ros.sample_indices_
        resampled_filenames = train_df.iloc[original_indices]['filename'].values

        train_oversampled = pd.DataFrame(X_resampled, columns=feature_cols)
        train_oversampled['label'] = y_resampled
        train_oversampled['class_name'] = pd.Series(y_resampled).map(id2name)
        train_oversampled['filename'] = resampled_filenames

        # 8. Gán lại cho val/test
        val_df_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
        val_df_scaled['label'] = val_df['label'].values
        val_df_scaled['filename'] = val_df['filename'].values
        val_df_scaled['class_name'] = val_df['class_name'].values

        test_df_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
        test_df_scaled['label'] = test_df['label'].values
        test_df_scaled['filename'] = test_df['filename'].values
        test_df_scaled['class_name'] = test_df['class_name'].values

        # 9. Lưu lại
        train_oversampled['split'] = 'train'
        val_df_scaled['split'] = 'dev'
        test_df_scaled['split'] = 'test'

        os.makedirs("datasets/clean_data", exist_ok=True)
        train_oversampled.to_csv("datasets/clean_data/train_data.csv", index=False)
        val_df_scaled.to_csv("datasets/clean_data/dev_data.csv", index=False)
        test_df_scaled.to_csv("datasets/clean_data/test_data.csv", index=False)

        print("Đã lưu train/dev/test đã xử lý tại datasets/clean_data/")

        # Gộp tất cả thành một DataFrame
        full_df = pd.concat([train_oversampled, val_df_scaled, test_df_scaled], ignore_index=True)
        output_path = "datasets/preprocessing_data/clean_data.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        full_df.to_csv(output_path, index=False)

        print(f"Đã lưu toàn bộ dữ liệu đã xử lý tại {output_path}")
