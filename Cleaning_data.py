import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === 1. Thiết lập thư mục dữ liệu ===
data_dir = os.path.dirname(__file__)  # Thư mục chứa file .py

# === 2. Đường dẫn các file ===
features_path = os.path.join(data_dir, "AwA2-features.txt")
labels_path = os.path.join(data_dir, "AwA2-labels.txt")
filenames_path = os.path.join(data_dir, "AwA2-filenames.txt")
classes_path = os.path.join(data_dir, "classes.txt")
trainclasses_path = os.path.join(data_dir, "trainclasses.txt")
testclasses_path = os.path.join(data_dir, "testclasses.txt")

# === 3. Đọc dữ liệu ===
#features = np.loadtxt(features_path) không thể do dữ liệu quá lớn

# Đường dẫn file AwA2-features.txt
data_dir = os.path.dirname(__file__)  # thư mục chứa file .py
txt_path = os.path.join(data_dir, "AwA2-features.txt")
npy_path = os.path.join(data_dir, "AwA2-features.npy")

# Chuyển đổi và lưu
print(" Đang đọc file .txt (mất vài phút)...")
features = []
with open(txt_path, "r") as f:
    for line in f:
        features.append([float(x) for x in line.strip().split()])
features = np.array(features)

print(" Đang lưu file .npy...")
np.save(npy_path, features)

print(f" Đã lưu file: {npy_path}")

features = np.load(os.path.join(data_dir, "AwA2-features.npy"))
labels = np.loadtxt(labels_path).astype(int) - 1
filenames = [line.strip() for line in open(filenames_path)]

# === 4. Tạo DataFrame ban đầu ===
df = pd.DataFrame(features)
df['label'] = labels
df['filename'] = filenames
df['class_name'] = df['filename'].apply(lambda x: x.split('_')[0])

# === 5. Ghép tên lớp đầy đủ từ classes.txt ===
class_map = pd.read_csv(classes_path, sep="\t", header=None, names=['id', 'name'])
class_map['name'] = class_map['name'].str.replace('+', ' ', regex=False)
class_map['id'] = class_map['id'] - 1
id2name = dict(zip(class_map['id'], class_map['name']))
df['class_fullname'] = df['label'].map(id2name)

# === 6. Phân chia train/test ===
train_classes = [line.strip().replace('+', ' ') for line in open(trainclasses_path)]
test_classes = [line.strip().replace('+', ' ') for line in open(testclasses_path)]
df['split'] = df['class_fullname'].apply(lambda x: 'train' if x in train_classes else 'test')

# === 7. Kiểm tra dữ liệu thiếu và trùng ===
print(" Thiếu dữ liệu:\n", df.isnull().sum())
print(" Dòng trùng lặp:", df.duplicated().sum())

# === 8. Chuẩn hóa đặc trưng ===
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(df.iloc[:, :-4].values)
#scaled_df = pd.DataFrame(X_scaled, columns=[f'f{i}' for i in range(features.shape[1])])
#df_scaled = pd.concat([scaled_df, df[['label', 'filename', 'class_name', 'class_fullname', 'split']]], axis=1)

# === Chuẩn hóa đặc trưng (chuẩn nhất) ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Tạo DataFrame đặc trưng đã chuẩn hóa
scaled_df = pd.DataFrame(X_scaled, columns=[f'f{i}' for i in range(features.shape[1])])

# Ghép với các cột thông tin
df_scaled = pd.concat([scaled_df, df[['label', 'filename', 'class_name', 'class_fullname', 'split']]], axis=1)

# === 9. Trực quan hóa: Phân phối lớp ===
plt.figure(figsize=(12, 5))
sns.countplot(data=df_scaled, x='class_fullname', order=df_scaled['class_fullname'].value_counts().index)
plt.xticks(rotation=90)
plt.title(" Phân phối mẫu theo lớp")
plt.tight_layout()
#plt.show()
plt.savefig("class_distribution.png")  

# === 10. Ma trận tương quan ===
plt.figure(figsize=(10, 8))
sns.heatmap(scaled_df.corr(), cmap='coolwarm', center=0)
plt.title(" Ma trận tương quan giữa các đặc trưng")
plt.tight_layout()
#plt.show()
plt.savefig("matrix.png")

# === 11. PCA giảm chiều để trực quan hóa ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_df)
df_scaled['PCA1'] = X_pca[:, 0]
df_scaled['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_scaled, x='PCA1', y='PCA2', hue='split', style='split', alpha=0.6)
plt.title(" PCA (Train vs Test)")
plt.tight_layout()
#plt.show()
plt.savefig("PCA.png")

# === 12. Lưu dữ liệu đã xử lý (nếu cần) ===
output_path = os.path.join(data_dir, "AwA2_processed_full.csv")
df_scaled.to_csv(output_path, index=False)
print(" Đã lưu dữ liệu xử lý tại:", output_path)
