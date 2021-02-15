Phone and Word Segmentation using Vector-Quantised Neural Networks
==================================================================

Disclaimer
----------
The code provided here is not pretty. But I believe that research should be
reproducible. I provide no guarantees with the code, but please let me know if
you have any problems, find bugs or have general comments.


Installation
------------
You will require the following:

- [Python 3](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/)
- [tqdm](https://tqdm.github.io/)
- [hydra](https://hydra.cc/)
- [wordseg](https://wordseg.readthedocs.io/)
- [speech_dtw](https://github.com/kamperh/speech_dtw/)
- [VectorQuantizedCPC fork](https://github.com/kamperh/VectorQuantizedCPC)
- [VectorQuantizedVAE fork (ZeroSpeech)](https://github.com/kamperh/ZeroSpeech)

To install `speech_dtw`, `VectorQuantizedCPC` and `VectorQuantizedVAE`, run
`./install_local.sh`.

Make sure all dependencies for `VectorQuantizedCPC` are also satisfied.


Data and preprocessing
----------------------
**To-do.** Maybe describe formats used in `data` and what would be required for
a new dataset. Could also point to the `dev_buckeye_transcripts` branch? Maybe
also include download link for Buckeye.


Train VQ-CPC and encode data
----------------------------
Change directory to `../VectorQuantizedCPC` and then run the following there.

Pre-process audio and extract log-Mel spectrograms:

    # Switchboard
    python preprocess.py in_dir=../datasets/swb300-wavs/ dataset=swbd preprocessing=8khz

    # Buckeye
    python preprocess.py in_dir=../datasets/buckeye/ dataset=buckeye

    # ZeroSpeech2019
    python preprocess.py in_dir=/media/kamperh/endgame/datasets/zerospeech2019/shared/databases dataset=2019/english

Train the VQ-CPC model:

    python train_cpc.py checkpoint_dir=checkpoints/cpc/swbd5 dataset=swbd training.sample_frames=128 preprocessing=8khz training.n_epochs=22000

Encode the data and write it to the `vqseg/exp/` directory. This should
be performed for all splits (`train`, `val` and `test`):

    # Switchboard
    python encode.py checkpoint=checkpoints/cpc/swbd5/model.ckpt-22000.pt split=val save_indices=True save_auxiliary=True save_embedding=../vqseg/exp/vqcpc/swbd/embedding.npy out_dir=../vqseg/exp/vqcpc/swbd/val/ dataset=swbd preprocessing=8khz

    # Buckeye
    python encode.py checkpoint=checkpoints/cpc/english2019/model.ckpt-22000.pt split=train save_indices=True save_auxiliary=True save_embedding=../vqseg/exp/vqcpc/buckeye/embedding.npy out_dir=../vqseg/exp/vqcpc/buckeye/train/ dataset=buckeye

    # Buckeye Felix split
    python encode.py checkpoint=checkpoints/cpc/english2019/model.ckpt-22000.pt split=val save_indices=True save_auxiliary=True save_embedding=../vqseg/exp/vqcpc/buckeye_felix/embedding.npy out_dir=../vqseg/exp/vqcpc/buckeye_felix/val/ dataset=buckeye_felix

    # ZeroSpeech2019
    python encode.py checkpoint=checkpoints/cpc/english2019/model.ckpt-22000.pt split=test save_indices=True save_auxiliary=True save_embedding=../vqseg/exp/vqcpc/english2019/embedding.npy out_dir=../vqseg/exp/vqcpc/english2019/test/ dataset=2019/english

Move back to the root `vqseg/` directory.

*Optional.* For the same-different evaluation below, it is useful to split the
encoded utterances into indivudal word segments. This in effect creates a new
data set. As an example, you could run the following in the root `vqseg/`
directory:

    # Switchboard
    python preprocess_word_segments.py --downsample_factor 2 --input_txt data/swbd_segments/ exp/vqcpc/swbd/ exp/vqcpc/swbd_segments/

    # Buckeye
    python preprocess_word_segments.py --downsample_factor 2 data/buckeye_segments/ exp/vqcpc/buckeye/ exp/vqcpc/buckeye_segments/


Use VQ-VAE to encode data
-------------------------
Change directory to `../VectorQuantizedVAE` and then run the following there.

The audio can be pre-processed again (as above), or alternatively you can
simply link to the audio from `VectorQuantizedCPC`:

    ln -s ../VectorQuantizedCPC/datasets/ .

Encode the data and write it to the `vqseg/exp/` directory. This should
be performed for all splits (`train`, `val` and `test`):

    # Buckeye
    python encode.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt split=train save_indices=True save_auxiliary=True save_embedding=../vqseg/exp/vqvae/buckeye/embedding.npy out_dir=../vqseg/exp/vqvae/buckeye/train/ dataset=buckeye

    # Buckeye Felix split
    python encode.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt split=val save_indices=True save_auxiliary=True save_embedding=../vqseg/exp/vqvae/buckeye_felix/embedding.npy out_dir=../vqseg/exp/vqvae/buckeye_felix/val/ dataset=buckeye_felix

    # ZeroSpeech2019
    python encode.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt split=test save_indices=True save_auxiliary=True save_embedding=../vqseg/exp/vqvae/english2019/embedding.npy out_dir=../vqseg/exp/vqvae/english2019/test/ dataset=2019/english


Phone segmentation
------------------
Phone segmentation can then be performed on the encoded data.

Greedy phone segmentation:

    # Switchboard (VQ-CPC)
    python phoneseg.py split=val dataset=swbd_segments output_tag=phoneseg_greedy

    # ZeroSpeech2019 (VQ-VAE) greedy
    python phoneseg.py input_format=txt split=test dataset=english2019 output_tag=phoneseg_greedy_min_segs phoneseg_algorithm=greedyseg model=vqvae

    # ZeroSpeech2019 (VQ-VAE) DP N-seg
    python phoneseg.py input_format=txt split=test dataset=english2019 output_tag=phoneseg_dp_nseg phoneseg_algorithm=dp_nseg model=vqvae

DP N-seg segmentation:

    # Buckeye (VQ-CPC)
    python phoneseg.py input_format=txt split=val dataset=buckeye output_tag=phoneseg_dp_nseg phoneseg_algorithm=dp_nseg

L2 segmentation:

    # Switchboard (VQ-CPC)
    python phoneseg.py split=val dataset=swbd_segments output_tag=phoneseg_l2 phoneseg_algorithm=l2seg

    # Buckeye (VQ-CPC)
    python phoneseg.py input_format=txt split=val dataset=buckeye output_tag=phoneseg_l2 phoneseg_algorithm=l2seg

    # Buckeye (VQ-VAE)
    python phoneseg.py input_format=txt split=val dataset=buckeye output_tag=phoneseg_l2 phoneseg_algorithm=l2seg phoneseg_algorithm.dur_weight=3 model=vqvae

    # Buckeye - Felix split (VQ-VAE)
    python phoneseg.py input_format=txt split=test dataset=buckeye_felix output_tag=phoneseg_l2 phoneseg_algorithm=l2seg phoneseg_algorithm.dur_weight=3 model=vqvae

    # ZeroSpeech2019 (VQ-CPC)
    python phoneseg.py input_format=txt split=test dataset=english2019 output_tag=phoneseg_l2 phoneseg_algorithm=l2seg

    # ZeroSpeech2019 (VQ-VAE)
    python phoneseg.py input_format=txt split=test dataset=english2019 output_tag=phoneseg_l2 phoneseg_algorithm=l2seg phoneseg_algorithm.dur_weight=3 model=vqvae

Evaluate the segmentation:

    # Buckeye (VQ-CPC)
    python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_l2
    
    # Buckeye (VQ-VAE)
    python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_l2 model=vqvae


Word segmentation
-----------------
Word segmentation can be performed on the segmented phone sequences.

Adaptor grammar word segmentation:

    conda activate wordseg
    python word_seg.py split=val dataset=buckeye phoneseg_tag=phoneseg_l2 output_tag=wordseg_l2_tp2

Evaluate the segmentation:

    python eval_segmentation.py split=val dataset=buckeye seg_tag=wordseg_l2_tp


Same-different evaluation
-------------------------
To perform a same-different evaluation, individual word segments needs to be
encoded separately. If an entire utterances have been encoded, the separate
word segments can be cut out using `utils/cut_segments.py`. An example using
this utility is given above.

Given a set of encoded isolated words, same-different evaluation can be
performed:

    python eval_samediff.py exp/vqcpc/swbd_segments/val/codes/

On the Switchboard data this should give:

    # Validation
    Average precision: 16.5983%
    Precision-recall breakeven: 24.6215%

    # Test    
    Average precision: 21.3922%
    Precision-recall breakeven: 29.1720%

You can also provide the indices and then convert those to codes before doing
the same-differente evaluation. For instance:

    python eval_samediff.py --indices_to_codes_embedding exp/vqcpc/swbd_segments/embedding.npy exp/vqcpc/swbd_segments/val/indices/

More results are given in `results.md`.


ABX evaluation
--------------
Convert the phone segmentation output to the ABX evaluation format:

    python phoneseg_to_abx.py vqcpc phoneseg_l2

Install the [ZeroSpeech2020](https://github.com/bootphon/zerospeech2020) tools.
Before running the `python setup.py install` step, I had to go into the
`zerospeech2020/validation` and `zerospeech2020/evaluation` directories and
change all the `tdev2` import statements in the code to `tde`.

Then perform the evaluation by activating the conda environment and running the
steps as below:

    conda activate zerospeech2020
    export ZEROSPEECH2020_DATASET=/media/kamperh/endgame/datasets/zerospeech2020/2020/
    cd exp/vqcpc/english2019/test/phoneseg_l2
    zerospeech2020-evaluate 2019 -j4 abx/ -o abx_results.json


Notebooks
---------
- `view_phoneseg_examples.ipynb` - Shows example segmentation outputs.
- `phoneseg_examples.ipynb` - Performs segmentation in the notebook.


Directory structure
-------------------
To-do


Depricated
----------
*Optional.* The encoded data can be split into smaller segments (e.g. for
cutting out words). This in effect creates a new data set. As an example, you
could run the following in the root `vqseg/` directory:

    python utils/cut_segments.py --input_txt --downsample_factor 2 data/swbd_segments/val.segments exp/vqcpc/swbd/val/codes/ exp/vqcpc/swbd_segments/val/codes/
