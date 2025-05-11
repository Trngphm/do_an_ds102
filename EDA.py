import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# === I. EDA CƠ BẢN ===

# 1. Tổng quan dữ liệu
print("Đọc dữ liệu từ file CSV...")
data_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(data_dir, "AwA2_processed_full.csv"))

print("\n1. Kích thước dữ liệu:", df.shape)
print("\n2. 5 dòng đầu:")
print(df.head())

print("\n3. Thông tin dữ liệu:")
df.info()

print("\n4. Thống kê mô tả:")
print(df.describe(include='all'))

# 2. Kiểm tra kiểu dữ liệu
print("\n5. Kiểu dữ liệu từng cột:")
print(df.dtypes)

# 3. Dữ liệu thiếu
missing = df.isnull().sum()
print("\n6. Số lượng giá trị thiếu:")
print(missing[missing > 0])

# 4. Dữ liệu trùng
print("\n7. Dòng bị trùng:", df.duplicated().sum())

# 5. Thống kê phân phối
plt.figure(figsize=(10, 4))
sns.histplot(df['f0'], bins=30, kde=True)
plt.title("Phân phối của đặc trưng f0")
plt.tight_layout()
plt.savefig("hist_f0.png")

plt.figure(figsize=(10, 4))
df['class_fullname'].value_counts().plot(kind='bar')
plt.title("Tần suất theo lớp")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("bar_class_dist.png")

# 6. Biểu đồ đơn giản
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, y='f0')
plt.title("Boxplot đặc trưng f0")
plt.tight_layout()
plt.savefig("boxplot_f0.png")

plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x='split', y='f10')
plt.title("Violinplot f10 theo tập split")
plt.tight_layout()
plt.savefig("violinplot_f10.png")

# === II. EDA TRUNG BÌNH ===

# 7. Tương quan
feature_cols = [col for col in df.columns if col.startswith('f') and col[1:].isdigit()]
plt.figure(figsize=(10, 8))
sns.heatmap(df[feature_cols].corr(), cmap='coolwarm', center=0)
plt.title("Ma trận tương quan giữa các đặc trưng")
plt.tight_layout()
plt.savefig("correlation_matrix.png")

# 8. So sánh giữa các nhóm
print("\nTrung bình đặc trưng theo label:")
print(df.groupby('label')[feature_cols].mean().head())

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='split', y='f20')
plt.title("So sánh f20 giữa tập train/test")
plt.tight_layout()
plt.savefig("boxplot_f20_split.png")

# 9. Phân tích đa biến
sns.pairplot(df, vars=['f0', 'f1', 'f2'], hue='split', corner=True)
plt.savefig("pairplot_f012.png")

# PCA trực quan hóa
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[feature_cols])
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='split')
plt.title("PCA 2D")
plt.tight_layout()
plt.savefig("pca_scatter.png")

# === III. EDA NÂNG CAO ===

# 10. Xử lý outliers - hiển thị giá trị bất thường theo IQR
Q1 = df['f0'].quantile(0.25)
Q3 = df['f0'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['f0'] < Q1 - 1.5 * IQR) | (df['f0'] > Q3 + 1.5 * IQR)]
print(f"\nSố lượng outlier f0 theo IQR: {len(outliers)}")

# 11. Biến đổi dữ liệu (Scaling đã làm ở preprocessing, Encoding không cần vì class đã là số)
print("\nEDA hoàn tất.")
