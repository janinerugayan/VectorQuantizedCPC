#!/usr/bin/env python

"""
Perform word segmentation.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from hydra import utils
from pathlib import Path
from tqdm import tqdm
import hydra

from eval_segmentation import get_intervals_from_dir
import wordseg_algorithms


@hydra.main(config_path="conf", config_name="wordseg")
def segment(cfg):

    # Algorithm
    segment_func = getattr(wordseg_algorithms, cfg.wordseg_algorithm.function)

    # Directories
    phoneseg_dir = (
        Path(utils.to_absolute_path("exp"))/cfg.model/cfg.dataset/cfg.split/
        cfg.phoneseg_tag/"intervals"
        )

    # Read phone intervals
    phoneseg_interval_dict = {}
    print("Reading: {}".format(phoneseg_dir))
    phoneseg_interval_dict = get_intervals_from_dir(phoneseg_dir)
    utterances = phoneseg_interval_dict.keys()

    # Segmentation
    print("Segmenting:")
    prepared_text = []
    for utt_key in utterances:
        prepared_text.append(
            " ".join([i[2] + "_" for i in phoneseg_interval_dict[utt_key]])
            )
    kwargs = dict(cfg.wordseg_algorithm)
    kwargs.pop("function")
    word_segmentation = segment_func(prepared_text, **kwargs)
    # print(prepared_text[:10])
    wordseg_interval_dict = {}
    for i_utt, utt_key in tqdm(enumerate(utterances)):
        words_segmented = word_segmentation[i_utt].split(" ")
        word_start = 0
        word_label = ""
        i_word = 0
        wordseg_interval_dict[utt_key] = []
        for (phone_start,
                phone_end, phone_label) in phoneseg_interval_dict[utt_key]:
            word_label += phone_label + "_"
            if words_segmented[i_word] == word_label:
                wordseg_interval_dict[utt_key].append((
                    word_start, phone_end, word_label
                    ))
                word_label = ""
                word_start = phone_end
                i_word += 1

    # Write intervals
    output_dir = (
        Path(utils.to_absolute_path("exp"))/cfg.model/cfg.dataset/cfg.split/
        cfg.output_tag/"intervals"
        )
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Writing to: {}".format(output_dir))
    for utt_key in tqdm(wordseg_interval_dict):
        with open((output_dir/utt_key).with_suffix(".txt"), "w") as f:
            for start, end, label in wordseg_interval_dict[utt_key]:
                f.write("{:d} {:d} {}\n".format(start, end, label))


if __name__ == "__main__":
    segment()
