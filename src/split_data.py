import pandas as pd
from builders.task_builder import META_TASK
import os


@META_TASK.register()
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

        # Tạo tập train và test dựa vào cột 'split'
        train_df = df[df['split'] == 'train'][selected_columns]
        test_df = df[df['split'] == 'test'][selected_columns]

        # Lưu train data
        train_df.to_csv("datasets/clean_data/train_data.csv", index=False)

        # Lưu test data
        test_df.to_csv("datasets/clean_data/test_data.csv", index=False)