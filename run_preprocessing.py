import os
from preprocessing import Preprocessing

def main():
    # Kiểm tra xem dữ liệu đã tồn tại chưa
    required_files = [
        "datasets/clean_data/train_data.csv",
        "datasets/clean_data/dev_data.csv",
        "datasets/clean_data/test_data.csv"
    ]
    
    if all(os.path.exists(f) for f in required_files):
        print("Data already preprocessed. Skipping...")
        return
    
    # Chạy tiền xử lý
    print("Running preprocessing...")
    preprocessor = Preprocessing()
    preprocessor.forward()

if __name__ == "__main__":
    main()