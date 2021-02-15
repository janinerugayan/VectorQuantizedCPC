#!/usr/bin/env python

"""
Convert phone segmentation output to the ABX evaluation format.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "model", type=str, help="the model type", choices=["vqvae", "vqcpc"]
        )
    parser.add_argument("seg_tag", type=str, help="phone segmentation tag")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Directories and filenames
    base_dir = Path("exp")/args.model/"english2019"/"test"/args.seg_tag
    indices_dir = base_dir/"indices"
    abx_dir = base_dir/"abx"/"2019"/"english"/"test"
    embedding_fn = (Path("exp")/args.model/"english2019"/"embedding.npy")

    # Read embeddings
    embeddings = np.load(embedding_fn)

    # Create output directory
    abx_dir.mkdir(exist_ok=True, parents=True)

    # Read indices and write codes
    print("Writing: {}".format(abx_dir))
    for indices_fn in tqdm(indices_dir.glob("*.npy")):
        indices = np.load(indices_fn)
        codes = embeddings[indices]
        codes_fn = (abx_dir/indices_fn.stem).with_suffix(".txt")
        np.savetxt(codes_fn, codes, fmt="%.16f")

    print("Now run:")
    print("conda activate zerospeech2020")
    print("export ZEROSPEECH2020_DATASET="
        "/media/kamperh/endgame/datasets/zerospeech2020/2020/"
        )
    print("cd {}".format(base_dir))
    print("zerospeech2020-evaluate 2019 -j4 abx/ -o abx_results.json")
    print("cat abx_results.json")
    print("cd -")

if __name__ == "__main__":
    main()
