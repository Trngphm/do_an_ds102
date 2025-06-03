import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import TextIOWrapper
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import RandomOverSampler
import csv
from builders.task_builder import META_TASK

@META_TASK.register()
class Preprocessing():
    def __init__(self):
         # === 1. Thiết lập đường dẫn ===
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/raw_data/"))
        self.zip_path = os.path.join(data_dir, "AwA2-features.zip")
        self.npy_path = os.path.join(data_dir, "AwA2-features.npy")
        self.classes_txt_path = os.path.join(data_dir, "classes.txt")  # File ngoài .zip

        # Các đường dẫn nội bộ trong file zip
        zip_internal_dir = "Animals_with_Attributes2/Features/ResNet101/"
        self.features_txt = zip_internal_dir + "AwA2-features.txt"
        self.labels_txt = zip_internal_dir + "AwA2-labels.txt"
        self.filenames_txt = zip_internal_dir + "AwA2-filenames.txt"

    def forward(self):
        # === 2. Đọc features từ zip hoặc .npy ===
        if not os.path.exists(self.npy_path):
            print(" Đang đọc features từ .zip (mất vài phút)...")
            features = []
            with zipfile.ZipFile(self.zip_path) as zf:
                with zf.open(self.features_txt) as f:
                    for line in TextIOWrapper(f, encoding='utf-8'):
                        features.append([float(x) for x in line.strip().split()])
            features = np.array(features)
            os.makedirs(os.path.dirname(self.npy_path), exist_ok=True)
            np.save(self.npy_path, features)
            print(" Đã lưu file features.npy")
        else:
            features = np.load(self.npy_path)
            print(" Đã tải features từ .npy")

        # === 3. Đọc các file nhãn và metadata từ zip và local ===
        with zipfile.ZipFile(self.zip_path) as zf:
            with zf.open(self.labels_txt) as f:
                labels = np.loadtxt(f).astype(int) - 1
            with zf.open(self.filenames_txt) as f:
                filenames = [line.strip() for line in TextIOWrapper(f, encoding='utf-8')]

        # Đọc file classes.txt từ hệ thống file, không phải từ zip
        class_map = pd.read_csv(self.classes_txt_path, sep="\t", header=None, names=['id', 'name'])

        # === 4. Tạo DataFrame ===
        df = pd.DataFrame(features, columns=[f"f{i}" for i in range(features.shape[1])])
        df['label'] = labels
        df['filename'] = filenames
        df['class_name'] = df['filename'].apply(lambda x: x.split('_')[0])

        # === 5. Ghép tên lớp đầy đủ ===
        class_map['name'] = class_map['name'].str.replace('+', ' ', regex=False)
        class_map['id'] = class_map['id'] - 1
        id2name = dict(zip(class_map['id'], class_map['name']))
        df['class_fullname'] = df['label'].map(id2name)

        # === 6. Phân chia train/test ===
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class_fullname'], random_state=42)

        # Gán cột 'split' cho từng tập
        train_df['split'] = 'train'
        test_df['split'] = 'test'

        # Kết hợp lại
        df = pd.concat([train_df, test_df], ignore_index=True)

        # Kiểm tra lại xem cả hai tập 'train' và 'test' có trong DataFrame df không
        print(df['split'].value_counts())  # Kiểm tra số lượng phân chia


        # === 7. Kiểm tra dữ liệu ===
        print(" Thiếu dữ liệu:\n", df.isnull().sum())
        print(" Dòng trùng lặp:", df.duplicated().sum())

        # === 8. Chuẩn hóa ===
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        X_scaled = scaler.fit_transform(features)
        scaled_df = pd.DataFrame(X_scaled, columns=[f'f{i}' for i in range(features.shape[1])])

        # === 8. Oversampling tập train để cân bằng lớp ===
        X_train = train_df.iloc[:, :-4]  # loại bỏ label, filename, class_name, class_fullname
        y_train = train_df['label']
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

        # Tạo lại DataFrame đã oversample
        df_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        df_resampled['label'] = y_resampled
        df_resampled['split'] = 'train_oversampled'

        # === 9. Ghép test và train đã oversample ===
        test_df['split'] = 'test'
        final_df = pd.concat([df_resampled, test_df], ignore_index=True)

        # Thêm tên lớp
        final_df['class_name'] = final_df['label'].map(lambda x: id2name[x])
        final_df['class_fullname'] = final_df['class_name']

        # === 10. PCA cho trực quan hóa ===
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(final_df.iloc[:, :-5])
        final_df['PCA1'] = X_pca[:, 0]
        final_df['PCA2'] = X_pca[:, 1]

        # === . Lưu dữ liệu ===
        output_path = "datasets/preprocessing_data/clean_data.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        print(" Đã lưu dữ liệu xử lý tại:", output_path)
