import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import TextIOWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# === 1. Thiết lập đường dẫn ===
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datastes/raw_data/"))
zip_path = os.path.join(data_dir, "AwA2-features.zip")
npy_path = os.path.join(data_dir, "AwA2-features.npy")
classes_txt_path = os.path.join(data_dir, "classes.txt")  # File ngoài .zip

# Các đường dẫn nội bộ trong file zip
zip_internal_dir = "Animals_with_Attributes2/Features/ResNet101/"
features_txt = zip_internal_dir + "AwA2-features.txt"
labels_txt = zip_internal_dir + "AwA2-labels.txt"
filenames_txt = zip_internal_dir + "AwA2-filenames.txt"

# === 2. Đọc features từ zip hoặc .npy ===
if not os.path.exists(npy_path):
    print(" Đang đọc features từ .zip (mất vài phút)...")
    features = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(features_txt) as f:
            for line in TextIOWrapper(f, encoding='utf-8'):
                features.append([float(x) for x in line.strip().split()])
    features = np.array(features)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, features)
    print(" Đã lưu file features.npy")
else:
    features = np.load(npy_path)
    print(" Đã tải features từ .npy")

# === 3. Đọc các file nhãn và metadata từ zip và local ===
with zipfile.ZipFile(zip_path) as zf:
    with zf.open(labels_txt) as f:
        labels = np.loadtxt(f).astype(int) - 1
    with zf.open(filenames_txt) as f:
        filenames = [line.strip() for line in TextIOWrapper(f, encoding='utf-8')]

# Đọc file classes.txt từ hệ thống file, không phải từ zip
class_map = pd.read_csv(classes_txt_path, sep="\t", header=None, names=['id', 'name'])

# === 4. Tạo DataFrame ===
df = pd.DataFrame(features)
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
scaled_df = pd.DataFrame(X_scaled, columns=[f'f{i}' for i in range(features.shape[1])])
df_scaled = pd.concat([scaled_df, df[['label', 'filename', 'class_name', 'class_fullname', 'split']]], axis=1)

# === 9. PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_df)
df_scaled['PCA1'] = X_pca[:, 0]
df_scaled['PCA2'] = X_pca[:, 1]

# === 10. Lưu dữ liệu ===
output_path = "datasets/preprocessing_data/clean_data.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_scaled.to_csv(output_path, index=False)
print(" Đã lưu dữ liệu xử lý tại:", output_path)
