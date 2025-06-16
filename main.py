from src.preprocessing import Preprocessing
from src.split_data import SplitData
from builders.register import Registry
from builders.task_builder import build_task
import yaml
import argparse
from easydict import EasyDict as edict

from builders.task_builder import META_TASK
from builders.model_builder import build_model

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return edict(config_dict)


def main(config):
    # preprocessing = Preprocessing()
    # preprocessing.forward()

    # split_data = SplitData()
    # split_data.forward()

    # Build task
    task_class = META_TASK.get("MLClassificationTask")
    task = task_class(config)

    # Gán model từ builder vào task
    task.model = build_model(config.model)

    # Train
    task.train()

    # Evaluate trên dev
    dev_acc, dev_f1, dev_report = task.evaluate(split="dev")

    # Evaluate
    acc, f1, report = task.evaluate(split="test")
    # Save model
    task.save_model()

    task.get_predictions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)

# python3 main.py --config configs/SVM.yaml
# pip install -r requirements.txt
