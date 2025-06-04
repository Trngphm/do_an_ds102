import pandas as pd
import os
from sklearn.model_selection import train_test_split

class SplitData():
    def __init__(self):
        file_path = "datasets/preprocessing_data/clean_data.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file tại: {file_path}")
        self.df = pd.read_csv(file_path)

    def forward(self):
        df = self.df

        if 'filename' in df.columns:
            df = df.drop(columns=['filename'])

        # Lọc các cột bắt đầu bằng 'f' và cột 'label'
        feature_columns = [col for col in df.columns if col.startswith('f')]
        selected_columns = feature_columns + ['label']

        # Lấy dữ liệu cần dùng
        data = df[selected_columns]

        # Chia dữ liệu: 70% train, 15% val, 15% test
        train_df, temp_df = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Lưu train, val, test
        train_df.to_csv("datasets/clean_data/train_data.csv", index=False)
        val_df.to_csv("datasets/clean_data/dev_data.csv", index=False)
        test_df.to_csv("datasets/clean_data/test_data.csv", index=False)

