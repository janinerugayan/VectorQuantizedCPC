#!/usr/bin/env python

"""
Cut segments from within longer sequences (normally an utterance).

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
        "segments_fn", type=str, help="segments to be cut out"
        )
    parser.add_argument(
        "input_dir", type=str, help="directory containing encoded sequences"
        )
    parser.add_argument(
        "output_dir", type=str, help="directory for new data set"
        )
    parser.add_argument(
        "--downsample_factor", type=int, default=1,
        )
    parser.add_argument(
        "--input_txt", action="store_true",
        help="input is .txt instead of the default .npy format"
        )
    parser.add_argument(
        "--output_txt", action="store_true",
        help="store as .txt instead of the default .npy format"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def cut_segments_from_utterances(utterance_dict, segments_list,
        downsample_factor=1):
    """
    Cut segments from a Numpy array dictionary and return a new dictionary.

    The dictionaries use the key format: "label_spkr_utterance_start-end" or
    "label_utterance_start-end".
    """

    # Create input segments dictionary
    utterance_segs = {}  # utterance_segs["sw04058-A_025640-026722"]
                         # is (25640, 26722)
    for key in utterance_dict.keys():
        utterance_segs[key] = tuple(
            [int(i) for i in key.split("_")[-1].split("-")]
            )
    
    # Create target segments dictionary
    target_segs = {}  # target_segs["probably_sw02557-A_059653-059737"]
                      # is ("sw02557-A", 59653, 59737)
    for segment in segments_list:
        segment_split = segment.split("_")
        if len(segment_split) == 4:
            utterance = segment_split[-3] + "_" + segment_split[-2]
        else:
            utterance = segment_split[-2]
        start, end = segment_split[-1].split("-")
        start = int(start)
        end = int(end)
        target_segs[segment.strip()] = (utterance, start, end)

    print("Extracting segments:")
    segments_dict = {}
    n_target_segs = 0
    for target_seg_key in tqdm(sorted(target_segs)):
        utterance, target_start, target_end = target_segs[target_seg_key]
        for utterance_key in [
                i for i in utterance_segs.keys() if i.startswith(utterance)]:
            utterannce_start, utterance_end = utterance_segs[utterance_key]
            if (target_start >= utterannce_start and target_start <=
                    utterance_end):
                if downsample_factor == 1:
                    start = target_start - utterannce_start
                    end = target_end - utterannce_start
                else:
                    start = int(
                        round((target_start -
                        utterannce_start)/downsample_factor)
                        )
                    end = int(
                        round((target_end -
                        utterannce_start)/downsample_factor)
                        )
                segments_dict[target_seg_key] = utterance_dict[
                    utterance_key
                    ][start:end]
                n_target_segs += 1
                break
    print(
        "Extracted {} out of {} segments".format(n_target_segs,
        len(target_segs))
        )

    return segments_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Read inputs
    input_dir = Path(args.input_dir)
    input_dict = {}
    print("Reading: {}".format(input_dir))
    if args.input_txt:
        for input_fn in tqdm(input_dir.glob("*.txt")):
            input_dict[input_fn.stem] = np.loadtxt(input_fn)
            # break
    else:
        for input_fn in tqdm(input_dir.glob("*.npy")):
            input_dict[input_fn.stem] = np.load(input_fn)

    # Read input segments
    segments_fn = Path(args.segments_fn)
    print("Reading: {}".format(segments_fn))
    segments = [i.strip() for i in segments_fn.read_text().strip().split("\n")]
    
    # Cut segments
    output_dict = cut_segments_from_utterances(
        input_dict, segments, args.downsample_factor
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Write output
    print("Writing: {}".format(output_dir))
    for fn in tqdm(output_dict):
        if args.output_txt:
            np.savetxt(
                (output_dir/fn).with_suffix(".txt"), output_dict[fn], fmt="%.16f"
                )
        else:
            np.save(
                (output_dir/fn).with_suffix(".npy"), output_dict[fn]
                )


if __name__ == "__main__":
    main()
