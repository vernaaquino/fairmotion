# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import numpy as np

from fairmotion.tasks.motion_prediction import generate, utils, test
from fairmotion.utils import utils as fairmotion_utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_seeds():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    fairmotion_utils.create_dir_if_absent(args.save_model_path)
    logging.info(args._get_kwargs())
    utils.log_config(args.save_model_path, args)

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device if args.device else device
    logging.info(f"Using device: {device}")

    logging.info("Preparing dataset...")
    _, rep = os.path.split(args.preprocessed_path.strip("/")) # Added for input compression
    dataset, mean, std = utils.prepare_dataset(
        *[
            os.path.join(args.preprocessed_path, f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        batch_size=args.batch_size,
        device=device,
        shuffle=args.shuffle,
        as_int=(rep=="compressed") # Added for input compression
    )
    # Loss per epoch is the average loss per sequence
    num_training_sequences = len(dataset["train"]) * args.batch_size

    # number of predictions per time step = num_joints * angle representation
    # shape is (batch_size, seq_len, num_predictions)
    _, tgt_len, num_predictions = next(iter(dataset["train"]))[1].shape

    model = utils.prepare_model(
        input_dim=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
        k=args.k
    )

    criterion = nn.MSELoss()
    model.init_weights()
    training_losses, val_losses = [], []
    mae_val_losses_dict = dict()

    epoch_loss = 0
    with torch.no_grad():
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
            model.eval()
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            outputs = model(src_seqs, tgt_seqs, teacher_forcing_ratio=1,)
            loss = criterion(outputs, tgt_seqs)
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / num_training_sequences
        val_loss = generate.eval(
            model, criterion, dataset["validation"], args.batch_size, device,
        )
        logging.info(
            "Before training: "
            f"Training loss {epoch_loss} | "
            f"Validation loss {val_loss}"
        )

    logging.info("Training model...")
    torch.autograd.set_detect_anomaly(True)
    opt = utils.prepare_optimizer(model, args.optimizer, args.lr)
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        teacher_forcing_ratio = np.clip(
            (1 - 2 * epoch / args.epochs), a_min=0, a_max=1,
        )
        logging.info(
            f"Running epoch {epoch} | "
            f"teacher_forcing_ratio={teacher_forcing_ratio}"
        )
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
            opt.optimizer.zero_grad()
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            outputs = model(
                src_seqs, tgt_seqs, teacher_forcing_ratio=teacher_forcing_ratio
            )
            outputs = outputs.double()
            loss = criterion(
                outputs,
                utils.prepare_tgt_seqs(args.architecture, src_seqs, tgt_seqs),
            )
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / num_training_sequences
        training_losses.append(epoch_loss)
        val_loss = generate.eval(
            model, criterion, dataset["validation"], args.batch_size, device,
        )
        val_losses.append(val_loss)
        opt.epoch_step(val_loss=val_loss)
        logging.info(
            f"Training loss {epoch_loss} | "
            f"Validation loss {val_loss} | "
            f"Iterations {iterations + 1}"
        )
        if epoch % args.save_model_frequency == 0:
            _, rep = os.path.split(args.preprocessed_path.strip("/"))
            _, mae = test.test_model(
                model=model,
                dataset=dataset["validation"],
                rep=rep,
                device=device,
                mean=mean,
                std=std,
                max_len=tgt_len,
            )
            logging.info(f"Validation MAE: {mae}")
            torch.save(
                model.state_dict(), f"{args.save_model_path}/{epoch}.model"
            )
            if len(val_losses) == 0 or val_loss <= min(val_losses):
                torch.save(
                    model.state_dict(), f"{args.save_model_path}/best.model"
                )

    return training_losses, val_losses, mae_val_losses_dict


def plot_curves(args, training_losses, val_losses, mae_val_losses_dict):
    plt.title('MSE Loss Curve')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epochs')
    plt.plot(training_losses, label="Train")
    plt.plot(val_losses, label="Valid")
    plt.legend()
    plt.savefig(f"{args.save_model_path}/mse_loss.png", format="png")
    plt.clf()

    # plt.title('MAE Validation Loss Curve')
    # plt.ylabel('MAE Loss')
    # plt.xlabel('Epochs')
    #
    # for key, value in mae_val_losses_dict.items():
    #     plt.plot(value, label=f"{key} frames")
    #
    # plt.legend()
    # plt.savefig(f"{args.save_model_path}/mae_validation_loss.png", format="png")
    # plt.clf()


def save_to_csv(args, training_losses, val_losses, mae_val_losses_dict):
    np.savetxt(f"{args.save_model_path}/mse_loss_dump.csv", np.column_stack((training_losses, val_losses)), header="Train,Validation", fmt='%f',
               delimiter=',', comments='')

    # frame6 = mae_val_losses_dict[6]
    # frame12 = mae_val_losses_dict[12]
    # frame18 = mae_val_losses_dict[18]
    # frame24 = mae_val_losses_dict[24]
    #
    # np.savetxt(f"{args.save_model_path}/mae_val_loss_dump.csv", np.column_stack((frame6, frame12, frame18, frame24)), header="6 Frames, 12 Frames, 18 Frames, 24 Frames", fmt='%f',
    #            delimiter=',', comments='')

def main(args):
    train_losses, val_losses, mae_val_losses_dict = train(args)
    save_to_csv(args, train_losses, val_losses, mae_val_losses_dict)
    plot_curves(args, train_losses, val_losses, mae_val_losses_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence to sequence motion prediction training"
    )
    parser.add_argument(
        "--preprocessed-path",
        type=str,
        help="Path to folder with pickled " "files",
        required=True,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training", default=64
    )
    parser.add_argument(
        "--shuffle", action='store_true',
        help="Use this option to enable shuffling",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=1024,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=1,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        help="Path to store saved models",
        required=True,
    )
    parser.add_argument(
        "--save-model-frequency",
        type=int,
        help="Frequency (in terms of number of epochs) at which model is "
        "saved",
        default=5,
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=200
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device",
        default=None,
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Seq2Seq archtiecture to be used",
        default="seq2seq",
        choices=[
            "seq2seq",
            "tied_seq2seq",
            "transformer",
            "transformer_encoder",
            "rnn",
            "cnn_seq2seq",  # Added for CNN Seq2Seq model (EJ edited)
            "tcn_seq2seq",   # Added for tcn_model (EJ edited)
            "transformer2", # Added for learned input compression
            "transformerv1", # Added for Transformer with temporal conv v1
            "transformerv2" # Added for Transformer with temporal conv v2
        ],
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=None,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Torch optimizer",
        default="sgd",
        choices=["adam", "sgd", "noamopt"],
    )
    parser.add_argument(
        "--k",
        type=int,
        help="kernel size for temporal conv transformer",
        default=3,
    )

    args = parser.parse_args()
    main(args)
