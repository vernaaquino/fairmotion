# Copyright (c) Facebook, Inc. and its affiliates.


import numpy as np
import os
import torch
from functools import partial
from multiprocessing import Pool

from fairmotion.models import (
    decoders,
    encoders,
    optimizer,
    rnn,
    seq2seq,
    transformer,
    cnn_decoders,
    cnn_encoders,
    cnn_seq2seq,
    tcn_decoders,
    tcn_encoders,
    tcn_seq2seq,  #cnn, tcn added for CNN Seq2Seq and TCN architecture (EJ edited)
    transformer2, # Added for learned input compression
)
from fairmotion.tasks.motion_prediction import dataset as motion_dataset
from fairmotion.utils import constants
from fairmotion.ops import conversions


def apply_ops(input, ops):
    """
    Apply series of operations in order on input. `ops` is a list of methods
    that takes single argument as input (single argument functions, partial
    functions). The methods are called in the same order provided.
    """
    output = input
    for op in ops:
        output = op(output)
    return output


def unflatten_angles(arr, rep):
    """
    Unflatten from (batch_size, num_frames, num_joints*ndim) to
    (batch_size, num_frames, num_joints, ndim) for each angle format
    """
    if rep == "aa":
        return arr.reshape(arr.shape[:-1] + (-1, 3))
    elif rep == "quat":
        return arr.reshape(arr.shape[:-1] + (-1, 4))
    elif rep == "rotmat":
        return arr.reshape(arr.shape[:-1] + (-1, 3, 3))
    elif rep == "compressed":
        return arr.reshape(arr.shape[:-1] + (-1, 1))


def flatten_angles(arr, rep):
    """
    Unflatten from (batch_size, num_frames, num_joints, ndim) to
    (batch_size, num_frames, num_joints*ndim) for each angle format
    """
    if rep == "aa":
        return arr.reshape(arr.shape[:-2] + (-1))
    elif rep == "quat":
        return arr.reshape(arr.shape[:-2] + (-1))
    elif rep == "rotmat":
        # original dimension is (batch_size, num_frames, num_joints, 3, 3)
        return arr.reshape(arr.shape[:-3] + (-1))
    elif rep == "compressed":
        return arr


def multiprocess_convert(arr, convert_fn):
    pool = Pool(40)
    result = list(pool.map(convert_fn, arr))
    return result


def convert_fn_to_R(rep):
    ops = [partial(unflatten_angles, rep=rep)]
    if rep == "aa":
        ops.append(partial(multiprocess_convert, convert_fn=conversions.A2R))
    elif rep == "quat":
        ops.append(partial(multiprocess_convert, convert_fn=conversions.Q2R))
    elif rep == "rotmat":
        ops.append(lambda x: x)
    ############
    # MARK
    # code added for input compression
    elif rep == "compressed":
        ops.append(partial(multiprocess_convert, convert_fn=conversions.C2R))
    # end input compression code
    ############
    ops.append(np.array)
    return ops


def identity(x):
    return x


def convert_fn_from_R(rep):
    if rep == "aa":
        convert_fn = conversions.R2A
    elif rep == "quat":
        convert_fn = conversions.R2Q
    elif rep == "rotmat":
        convert_fn = identity
    ############
    # MARK
    # code added for input compression
    elif rep == "compressed":
        convert_fn = conversions.R2C
    # end input compression code
    ############
    return convert_fn


def unnormalize(arr, mean, std):
    return arr * (std + constants.EPSILON) + mean


def prepare_dataset(
    train_path, valid_path, test_path, batch_size, device, shuffle=False,
    as_int=False # Added for input compression
):
    dataset = {}
    for split, split_path in zip(
        ["train", "test", "validation"], [train_path, valid_path, test_path]
    ):
        mean, std = None, None
        if split in ["test", "validation"]:
            mean = dataset["train"].dataset.mean
            std = dataset["train"].dataset.std
        dataset[split] = motion_dataset.get_loader(
            split_path, batch_size, device, mean, std, shuffle,
            as_int # Added for input compression
        )
    return dataset, mean, std


def prepare_model(
    input_dim, hidden_dim, device, num_layers=1, architecture="seq2seq", k=3
):
    if architecture == "rnn":
        model = rnn.RNN(input_dim, hidden_dim, num_layers)
    if architecture == "seq2seq":
        enc = encoders.LSTMEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim
        ).to(device)
        dec = decoders.LSTMDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            device=device,
        ).to(device)
        model = seq2seq.Seq2Seq(enc, dec)
    elif architecture == "tied_seq2seq":
        model = seq2seq.TiedSeq2Seq(input_dim, hidden_dim, num_layers, device)
    elif architecture == "transformer_encoder":
        model = transformer.TransformerLSTMModel(
            input_dim, hidden_dim, 4, hidden_dim, num_layers,
        )
    elif architecture == "transformer":
        model = transformer.TransformerModel(
            input_dim, hidden_dim, 4, hidden_dim, num_layers,
        )
    # Added for CNN seq2seq
    elif architecture == "cnn_seq2seq":
        enc = cnn_encoders.LSTMEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim
        ).to(device)
        dec = cnn_decoders.LSTMDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            device=device,
        ).to(device)
        model = cnn_seq2seq.Seq2Seq(enc, dec)
    # Added for tcn_model
    elif architecture == "tcn_seq2seq":
        enc = tcn_encoders.TCNEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim
        ).to(device)
        dec = tcn_decoders.TCNDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            device=device,
        ).to(device)
        model = tcn_seq2seq.TCNSeq2Seq(enc, dec)
    ############
    # MARK
    # Code added for learned input compression
    elif architecture == "transformer2":
        model = transformer2.TransformerModel2(
            input_dim, hidden_dim, 4, hidden_dim, num_layers,
        )
    # End of code added for learned input compression
    ###########
    elif architecture == "transformerv1":
        model = transformer.TransformerModelV1(
            input_dim, hidden_dim, 4, hidden_dim, num_layers, k=k
        )
    elif architecture == "transformerv2":
        model = transformer.TransformerModelV2(
            input_dim, hidden_dim, 4, hidden_dim, num_layers, k=k
        )
    model = model.to(device)
    model.zero_grad()
    model.double()
    return model


def log_config(path, args):
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, value in args._get_kwargs():
            f.write(f"{key}:{value}\n")


def prepare_optimizer(model, opt="sgd", lr=None):      #lr edit
    kwargs = {}
    if lr is not None:
        kwargs["lr"] = lr

    if opt == "sgd":
        return optimizer.SGDOpt(model, **kwargs)
    elif opt == "adam":
        return optimizer.AdamOpt(model, **kwargs)
    elif opt == "noamopt":
        return optimizer.NoamOpt(model)


def prepare_tgt_seqs(architecture, src_seqs, tgt_seqs):
    if architecture == "st_transformer" or architecture == "rnn":
        return torch.cat((src_seqs[:, 1:], tgt_seqs), axis=1)
    else:
        return tgt_seqs
