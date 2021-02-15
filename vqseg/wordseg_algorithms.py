"""
Word segmentation algorithms.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from wordseg.algos import tp, puddle, dpseg, baseline, dibs, ag
import wordseg.algos


def ag(utterance_list, **kwargs):
    return list(wordseg.algos.ag.segment(utterance_list, **kwargs))


def tp(utterance_list, **kwargs):
    return list(wordseg.algos.tp.segment(utterance_list, **kwargs))


def dpseg(utterance_list, **kwargs):
    return list(wordseg.algos.dpseg.segment(utterance_list, **kwargs))
