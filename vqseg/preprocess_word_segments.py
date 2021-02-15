#!/usr/bin/env python

"""
Create a new data set by cutting out word segments from encoded utterances.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import shutil
import sys

from utils import cut_segments


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "segments_dir", type=str, help="directory containing segment files"
        )
    parser.add_argument(
        "input_dir", type=str, help="directory containing encoded sequences"
        )
    parser.add_argument(
        "output_dir", type=str, help="directory for new data set"
        )
    parser.add_argument(
        "--input_txt", action="store_true",
        help="input is .txt instead of the default .npy format"
        )
    parser.add_argument(
        "--downsample_factor", type=int, default=1,
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    for split in ["train", "val", "test"]:
        input_basedir = Path(args.input_dir)/split
        segments_fn = (Path(args.segments_dir)/split).with_suffix(".segments")
        if not input_basedir.is_dir() or not segments_fn.is_file():
            continue

        # Read input segments
        print("Reading: {}".format(segments_fn))
        segments = [
            i.strip() for i in segments_fn.read_text().strip().split("\n")
            ]

        # Cut segments
        for feat_type in ["auxiliary_embedding1", "auxiliary_embedding2",
                "codes", "indices"]:
            input_dir = input_basedir/feat_type
            if not input_dir.is_dir():
                continue

            # Read inputs
            input_dict = {}
            print("Reading: {}".format(input_dir))
            if args.input_txt:
                for input_fn in tqdm(input_dir.glob("*.txt")):
                    input_dict[input_fn.stem] = np.loadtxt(input_fn)
                    # break
            else:
                for input_fn in tqdm(input_dir.glob("*.npy")):
                    input_dict[input_fn.stem] = np.load(input_fn)

            # Cut segments
            output_dict = cut_segments.cut_segments_from_utterances(
                input_dict, segments, args.downsample_factor
                )

            # Create output directory
            output_dir = Path(args.output_dir)/split/feat_type
            output_dir.mkdir(exist_ok=True, parents=True)

            # Write output
            print("Writing: {}".format(output_dir))
            for fn in tqdm(output_dict):
                np.save(
                    (output_dir/fn).with_suffix(".npy"), output_dict[fn]
                    )

        # Copy embedding matrix
        src_embedding_fn = Path(args.input_dir)/"embedding.npy"
        tgt_embedding_fn = Path(args.output_dir)/"embedding.npy"
        shutil.copy(src_embedding_fn, tgt_embedding_fn)


if __name__ == "__main__":
    main()
