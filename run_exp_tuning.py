import os
from pathlib import Path

def run_exp_tuning(learning_rates, batch_sizes):
    """run the experiments automatically and save all results"""
    pwd = os.getcwd()
    if not os.path.isfile(f"{pwd}/setup.py"):
        print("Please run this script in the root /fairmotion directory.")
        return

    # TODO setup paths and other constants
    epoch = 10
    device = "cuda"  # cpu
    preprocessed_path = f"../preprocessed/aa/" # insert path where preprocessed training data is stored
    architecture = "rnn" # seq2seq transformer

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            model_path = f"../experiments/{architecture}_learning_rate_{learning_rate}_batch_size_{batch_size}"  # TODO make sure to modify save path if tuning more hyperparams so you dont overwrite existing

            cmd = f"mkdir -p {model_path}"
            os.system(f"/bin/bash -c \"{cmd}\"")

            cmd = f"python fairmotion/tasks/motion_prediction/training.py --save-model-path {model_path} --preprocessed-path {preprocessed_path} --epochs {epoch} --device {device} --architecture {architecture} --lr {learning_rate} --batch-size {batch_size} |& tee {model_path}/log.txt"
            os.system(f"/bin/bash -c \"{cmd}\"")

    return

if __name__ == "__main__":
    # TODO setup hyperparameters to tune
    learning_rates = [0.1, 0.01]
    batch_sizes = [64]

    run_exp_tuning(learning_rates, batch_sizes)
