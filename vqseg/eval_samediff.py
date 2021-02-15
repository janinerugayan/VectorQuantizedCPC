#!/usr/bin/env python

"""
Perform same-different evaluation.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from collections import Counter
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import sys

sys.path.append(str(Path("../src/speech_dtw/speech_dtw")))
sys.path.append(str(Path("../src/speech_dtw/utils")))

import _dtw
import samediff

dtw_cost_func = _dtw.multivariate_dtw_cost_cosine


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("input_dir", type=str)
    parser.add_argument(
        "--indices_to_codes_embedding", type=str,
        help="if provided, the indices in `input_dir` are first converted to "
        "codes given this .npy embedding matrix"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def eval_samediff(segments_dict):
    """Returns average precision and recision-recall breakeven."""

    # Generate list of pairs
    segment_keys = sorted(segments_dict.keys())
    pairs = []
    m = len(segment_keys)
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            pairs.append((segment_keys[i], segment_keys[j]))
    # print("No. pairs: {}".format(len(pairs)))

    print("Calculating distances:")
    costs = np.zeros(len(pairs))
    for i_pair, pair in enumerate(tqdm(pairs)):
        utt_id_1, utt_id_2 = pair
        costs[i_pair] = dtw_cost_func(
            np.array(segments_dict[utt_id_1], dtype=np.double, order="c"),
            np.array(segments_dict[utt_id_2], dtype=np.double, order="c"), True
            )

    # Same-different
    distances_vec = np.asarray(costs)
    labels = [key.split("_")[0] for key in segment_keys]
    matches = samediff.generate_matches_array(labels)
    ap, prb = samediff.average_precision(
        distances_vec[matches == True], distances_vec[matches == False], False
        )
    return (ap, prb)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Read inputs
    input_dir = Path(args.input_dir)
    segments_dict = {}
    print("Reading: {}".format(input_dir))
    for input_fn in tqdm(input_dir.glob("*.npy")):
        segments_dict[input_fn.stem] = np.load(input_fn)

    # Convert indices to codes
    if args.indices_to_codes_embedding is not None:
        embeddings = np.load(args.indices_to_codes_embedding)
        code_counts = Counter()
        duration = 0

        print("Converting to codes:")
        codes_dict = {}
        for utt_key in tqdm(segments_dict):
            codes_dict[utt_key] = (
                embeddings[np.asarray(segments_dict[utt_key], dtype=np.int)]
                )

            # For bitrate
            code_counts.update(segments_dict[utt_key])
            start, end = utt_key.split("_")[-1].split("-")
            start = int(start)
            end = int(end)
            duration += (end - start)*10e-3

        segments_dict = codes_dict

        # Bitrate
        probs = []
        n_symbols = sum(code_counts.values())
        for i_symbol in code_counts:
            probs += [code_counts[i_symbol]/n_symbols, ]
        probs=np.array(probs)
        bits = -n_symbols*np.sum(probs*np.log2(probs))
        bitrate = bits/duration
        print("Max bitrate (with 2x downsampling): {:.4f} bits/sec".format(
            2*n_symbols*np.log2(2*n_symbols)/duration
            ))
        print("Bitrate: {:.4f} bits/sec".format(bitrate))

    # Evaluate
    ap, prb = eval_samediff(segments_dict)
    print("Average precision: {:.4f}%".format(ap*100))
    print("Precision-recall breakeven: {:.4f}%".format(prb*100))


if __name__ == "__main__":
    main()
