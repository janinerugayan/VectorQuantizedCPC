#!/usr/bin/env python

"""
Perform phone segmentation.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from hydra import utils
from pathlib import Path
from tqdm import tqdm
import hydra
import numpy as np

import phoneseg_algorithms


@hydra.main(config_path="conf", config_name="phoneseg")
def segment(cfg):
    # print(cfg)

    # Algorithm
    segment_func = getattr(
        phoneseg_algorithms, cfg.phoneseg_algorithm.function
        )

    # Read pre-quantisation
    input_dir = (
        Path(utils.to_absolute_path("exp"))/cfg.model/cfg.dataset/cfg.split
        )
    z_dir = input_dir/"auxiliary_embedding2"
    print("Reading: {}".format(z_dir))
    assert z_dir.is_dir(), "missing directory: {}".format(z_dir)
    z_dict = {}
    if cfg.input_format == "npy":
        for input_fn in tqdm(z_dir.glob("*.npy")):
            z_dict[input_fn.stem] = np.load(input_fn)
    elif cfg.input_format == "txt":
        for input_fn in tqdm(z_dir.glob("*.txt")):
            z_dict[input_fn.stem] = np.loadtxt(input_fn)
    else:
        assert False, "invalid input format"

    # Read embedding matrix
    embedding_fn = input_dir.parent/"embedding.npy"
    print("Reading: {}".format(embedding_fn))
    embedding = np.load(embedding_fn)

    # Segmentation
    boundaries_dict = {}
    code_indices_dict = {}
    kwargs = dict(cfg.phoneseg_algorithm)
    kwargs.pop("function")
    print("Running {}:".format(cfg.phoneseg_algorithm.function))
    for utt_key in tqdm(z_dict):
        
        # Segment
        z = z_dict[utt_key]
        if z.ndim == 1:
            continue
        boundaries, code_indices = segment_func(embedding, z, **kwargs)

        # Convert boundaries to same frequency as reference
        if cfg.downsample_factor > 1:
            boundaries_upsampled = np.zeros(
                len(boundaries)*cfg.downsample_factor, dtype=bool
                )
            for i, bound in enumerate(boundaries):
                boundaries_upsampled[i*cfg.downsample_factor + 1] = bound
            boundaries = boundaries_upsampled

            code_indices_upsampled = []
            for start, end, index in code_indices:
                code_indices_upsampled.append((
                    start*cfg.downsample_factor, 
                    end*cfg.downsample_factor,
                    index
                    ))
            code_indices = code_indices_upsampled
        
        boundaries_dict[utt_key] = boundaries_upsampled
        code_indices_dict[utt_key] = code_indices

    output_base_dir = input_dir/cfg.output_tag
    output_base_dir.mkdir(exist_ok=True, parents=True)
    print("Writing to: {}".format(output_base_dir))

    # Write code indices
    output_dir = output_base_dir/"indices"
    output_dir.mkdir(exist_ok=True, parents=True)
    # print("Writing to: {}".format(output_dir))
    for utt_key in tqdm(code_indices_dict):
        np.save(
            (output_dir/utt_key).with_suffix(".npy"),
            np.array([i[-1] for i in code_indices_dict[utt_key]], dtype=np.int)
            )

    # Write boundaries
    output_dir = output_base_dir/"boundaries"
    output_dir.mkdir(exist_ok=True, parents=True)
    # print("Writing to: {}".format(output_dir))
    for utt_key in tqdm(code_indices_dict):
        np.save(
            (output_dir/utt_key).with_suffix(".npy"),
            np.array(boundaries_dict[utt_key], dtype=np.bool)
            )

    # Write intervals
    output_dir = output_base_dir/"intervals"
    output_dir.mkdir(exist_ok=True, parents=True)
    # print("Writing to: {}".format(output_dir))
    for utt_key in tqdm(code_indices_dict):
        with open((output_dir/utt_key).with_suffix(".txt"), "w") as f:
            for start, end, index in code_indices_dict[utt_key]:
                f.write("{:d} {:d} {:d}\n".format(start, end, index))


if __name__ == "__main__":
    segment()
