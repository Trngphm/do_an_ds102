from src.preprocessing import Preprocessing
from builders.register import Registry
from builders.task_builder import build_task

# Hàm main để chạy toàn bộ quá trình
def main():
    preprocessing = Preprocessing()
    preprocessing.forward()

if __name__ == "__main__":
    main()