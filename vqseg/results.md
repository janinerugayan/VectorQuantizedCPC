Results Notebook
================

Buckeye
-------

### Segmentation on validation data

Collapse VQ-CPC:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_collapse
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 29.73%
    Recall: 98.87%
    F-score: 45.72%
    OS: 232.56%
    R-value: -98.90%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 7.67%
    Recall: 99.31%
    F-score: 14.24%
    OS: 1194.50%
    R-value: -919.81%
    ---------------------------------------------------------------------------

Greedy VQ-CPC:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_greedy
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 51.50%
    Recall: 65.63%
    F-score: 57.72%
    OS: 27.43%
    R-value: 56.16%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 13.53%
    Recall: 67.15%
    F-score: 22.53%
    OS: 396.23%
    R-value: -250.49%
    ---------------------------------------------------------------------------

DP N-seg VQ-CPC:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_dp_nseg
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 55.16%
    Recall: 70.27%
    F-score: 61.81%
    OS: 27.39%
    R-value: 59.59%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 15.49%
    Recall: 76.82%
    F-score: 25.78%
    OS: 395.94%
    R-value: -246.49%
    ---------------------------------------------------------------------------

L2 VQ-CPC:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_l2
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 55.72%
    Recall: 74.68%
    F-score: 63.82%
    OS: 34.04%
    R-value: 57.80%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 15.54%
    Recall: 81.03%
    F-score: 26.08%
    OS: 421.37%
    R-value: -266.58%
    ---------------------------------------------------------------------------

Collapse VQ-VAE:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_collapse model=vqvae
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 32.01%
    Recall: 98.31%
    F-score: 48.29%
    OS: 207.17%
    R-value: -77.43%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 8.42%
    Recall: 97.73%
    F-score: 15.50%
    OS: 1060.89%
    R-value: -806.33%
    ---------------------------------------------------------------------------

Greedy VQ-VAE:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_greedy model=vqvae
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 57.06%
    Recall: 72.94%
    F-score: 64.03%
    OS: 27.83%
    R-value: 61.19%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 13.53%
    Recall: 65.42%
    F-score: 22.43%
    OS: 383.42%
    R-value: -240.27%
    ---------------------------------------------------------------------------

DP N-seg VQ-VAE:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_dp_nseg model=vqvae
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 61.47%
    Recall: 78.50%
    F-score: 68.95%
    OS: 27.71%
    R-value: 65.07%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 14.89%
    Recall: 71.89%
    F-score: 24.67%
    OS: 382.80%
    R-value: -237.19%
    ---------------------------------------------------------------------------

L2 VQ-VAE:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=phoneseg_l2 model=vqvae
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 66.42%
    Recall: 75.77%
    F-score: 70.79%
    OS: 14.08%
    R-value: 72.44%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 15.81%
    Recall: 68.13%
    F-score: 25.66%
    OS: 330.90%
    R-value: -194.47%
    ---------------------------------------------------------------------------


TP word segmentation on top of L2 VQ-CPC segmentation:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=wordseg_l2_tp
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 66.40%
    Recall: 28.33%
    F-score: 39.72%
    OS: -57.33%
    R-value: 49.04%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 19.86%
    Recall: 32.95%
    F-score: 24.78%
    OS: 65.91%
    R-value: 5.98%
    ---------------------------------------------------------------------------

TP word segmentation on top of L2 VQ-VAE segmentation:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=wordseg_l2_tp model=vqvae
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 72.27%
    Recall: 28.67%
    F-score: 41.06%
    OS: -60.33%
    R-value: 49.40%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.07%
    Recall: 27.07%
    F-score: 21.67%
    OS: 49.83%
    R-value: 12.43%
    ---------------------------------------------------------------------------

AG word segmentation on top of L2 VQ-CPC segmentation:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=wordseg_l2_ag
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 63.86%
    Recall: 48.65%
    F-score: 55.22%
    OS: -23.82%
    R-value: 61.96%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.24%
    Recall: 54.05%
    F-score: 27.27%
    OS: 196.37%
    R-value: -86.51%
    ---------------------------------------------------------------------------

AG word segmentation on top of L2 VQ-VAE segmentation:

    # python eval_segmentation.py split=val dataset=buckeye seg_tag=wordseg_l2_ag model=vqvae
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 69.19%
    Recall: 63.23%
    F-score: 66.08%
    OS: -8.61%
    R-value: 71.17%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 16.44%
    Recall: 56.75%
    F-score: 25.49%
    OS: 245.18%
    R-value: -126.46%
    ---------------------------------------------------------------------------

DPSeg (broken?) word segmentation on top of L2 VQ-CPC segmentation:

    # ./eval_segmentation.py split=val dataset=buckeye seg_tag=wordseg_l2_dpseg
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 55.66%
    Recall: 69.47%
    F-score: 61.80%
    OS: 24.82%
    R-value: 60.76%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 15.47%
    Recall: 75.10%
    F-score: 25.65%
    OS: 385.52%
    R-value: -238.26%
    ---------------------------------------------------------------------------


### Segmentation on Felix test data

Our evaluation of Kreuk et al. (https://arxiv.org/abs/2007.13465):

    # python eval_segmentation.py split=test dataset=buckeye_felix seg_tag=. model=felix
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 65.77%
    Recall: 74.40%
    F-score: 69.82%
    OS: 13.12%
    R-value: 71.93%
    ---------------------------------------------------------------------------

    # python eval_segmentation.py split=test dataset=buckeye_felix seg_tag=. model=felix phone_tolerance=3
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 71.77%
    Recall: 81.19%
    F-score: 76.19%
    OS: 13.12%
    R-value: 77.24%
    ---------------------------------------------------------------------------

L2 VQ-VAE DP N-seg segmentation:

    # python eval_segmentation.py split=test dataset=buckeye_felix seg_tag=phoneseg_dp_nseg model=vqvae phone_tolerance=3
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 61.74%
    Recall: 90.90%
    F-score: 73.53%
    OS: 47.24%
    R-value: 56.03%
    ---------------------------------------------------------------------------

L2 VQ-VAE segmentation:

    # python eval_segmentation.py split=test dataset=buckeye_felix seg_tag=phoneseg_l2 model=vqvae
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 63.80%
    Recall: 77.14%
    F-score: 69.83%
    OS: 20.91%
    R-value: 69.03%
    ---------------------------------------------------------------------------

    # python eval_segmentation.py split=test dataset=buckeye_felix seg_tag=phoneseg_l2 model=vqvae phone_tolerance=3
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 70.82%
    Recall: 85.63%
    F-score: 77.53%
    OS: 20.91%
    R-value: 74.84%
    ---------------------------------------------------------------------------



### Same-different on validation data

Unsegmented VQ-CPC:

    # python eval_samediff.py exp/vqcpc/buckeye_segments/val/codes/
    Bitrate: 429.0082 bits/sec
    Average precision: 45.6473%
    Precision-recall breakeven: 47.2122%

Unsegmented VQ-VAE:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqvae/buckeye_segments/embedding.npy exp/vqvae/buckeye_segments/val/indices/
    Bitrate: 409.6981 bits/sec
    Average precision: 50.4481%
    Precision-recall breakeven: 50.4388%

Merged VQ-CPC codes:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqcpc/buckeye_segments/embedding.npy exp/vqcpc/buckeye_segments/val/phoneseg_merge/
    Bitrate: 349.7209 bits/sec
    Average precision: 47.1283%
    Precision-recall breakeven: 47.9360%

Greedy VQ-CPC segmentation:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqcpc/buckeye_segments/embedding.npy exp/vqcpc/buckeye_segments/val/phoneseg_greedy/
    Average precision: 18.3154%
    Precision-recall breakeven: 25.1936%

Greedy VQ-VAE segmentation:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqvae/buckeye_segments/embedding.npy exp/vqvae/buckeye_segments/val/phoneseg_greedy/indices/
    Average precision: 26.4246%
    Precision-recall breakeven: 31.6211%

L2 VQ-CPC segmentation:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqcpc/buckeye_segments/embedding.npy exp/vqcpc/buckeye_segments/val/phoneseg_l2/
    Average precision: 35.5092%
    Precision-recall breakeven: 40.1394%

Merged VQ-VAE:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqvae/buckeye_segments/embedding.npy exp/vqvae/buckeye_segments/val/phoneseg_merge/indices/
    Bitrate: 310.3472 bits/sec
    Average precision: 47.8736%
    Precision-recall breakeven: 48.2452%

DP N-seg VQ-VAE:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqvae/buckeye_segments/embedding.npy exp/vqvae/buckeye_segments/val/phoneseg_dp_nseg/indices/
    Bitrate: 134.9551 bits/sec
    Average precision: 36.0942%
    Precision-recall breakeven: 39.3599%

L2 VQ-VAE:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqvae/buckeye_segments/embedding.npy exp/vqvae/buckeye_segments/val/phoneseg_l2/indices/
    Bitrate: 118.0934 bits/sec
    Average precision: 35.6942%
    Precision-recall breakeven: 38.9520%

MFCCs: 

    # projects/stellenbosch/bucktsong_awe_py3/samediff
    Average precision: 37.5848%
    Precision-recall breakeven: 40.4233%

Filterbanks:

    # projects/stellenbosch/bucktsong_awe_py3/samediff
    Average precision: 19.2113%
    Precision-recall breakeven: 25.6582%


###Same-different on test data

MFCCs:

    # projects/stellenbosch/bucktsong_awe_py3/samediff
    Average precision: 38.9556%
    Precision-recall breakeven: 42.3245%

Filterbanks:

    # projects/stellenbosch/bucktsong_awe_py3/samediff
    Average precision: 19.3433%
    Precision-recall breakeven: 26.2088%

Unsegmented VQ-VAE:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqvae/buckeye_segments/embedding.npy exp/vqvae/buckeye_segments/test/indices/
    Bitrate: 410.6826 bits/sec
    Average precision: 47.8423%
    Precision-recall breakeven: 49.1149%

Greedy VQ-VAE segmentation:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqvae/buckeye_segments/embedding.npy exp/vqvae/buckeye_segments/test/phoneseg_greedy/indices/
    Bitrate: 135.4884 bits/sec
    Average precision: 25.6669%
    Precision-recall breakeven: 32.7787%


Merged VQ-VAE:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqvae/buckeye_segments/embedding.npy exp/vqvae/buckeye_segments/test/phoneseg_merge/indices/
    Bitrate: 301.2490 bits/sec   
    Average precision: 46.8264%
    Precision-recall breakeven: 48.7677%

L2 VQ-VAE:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqvae/buckeye_segments/embedding.npy exp/vqvae/buckeye_segments/test/phoneseg_l2/indices/
    Bitrate: 118.2304 bits/sec
    Average precision: 34.5029%
    Precision-recall breakeven: 39.4971%


Switchboard
-----------

### Same-different on validation data

Unsegmented VQ-CPC:

    # python eval_samediff.py exp/vqcpc/swbd_segments/val/codes/
    Average precision: 16.5983%
    Precision-recall breakeven: 24.6215%

Unsegmented codes using model trained on ZeroSpeech 2019:

    Average precision: 3.7601%
    Precision-recall breakeven: 9.7699%

Merged codes:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqcpc/swbd_segments/embedding.npy exp/vqcpc/swbd_segments/val/phoneseg_merge/
    Average precision: 16.4563%
    Precision-recall breakeven: 24.4263%

Greedy segmentation:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqcpc/swbd_segments/embedding.npy exp/vqcpc/swbd_segments/val/phoneseg_greedy/
    Average precision: 3.8499%
    Precision-recall breakeven: 10.0877%

L2 segmentation:

    # python eval_samediff.py --indices_to_codes_embedding exp/vqcpc/swbd_segments/embedding.npy exp/vqcpc/swbd_segments/val/phoneseg_l2/
    Average precision: 11.3473%
    Precision-recall breakeven: 19.6982%

MFCCs:

    # projects/ttic/samediff/swbd/dtw_baselines
    Average precision: 14.1686%
    Precision-recall breakeven: 21.1157%


###Same-different on test data

MFCCs:

    # projects/ttic/samediff/swbd/dtw_baselines
    Average precision: 15.6335%
    Precision-recall breakeven: 23.3177%


ZeroSpeech 2019 (ABX) [final]
-----------------------------
MFCCs:

    # https://arxiv.org/abs/1904.07556
    abx: 22.7
    bitrate: 1738

ABX on unsegmented VQ-CPC codes:

    abx: 13.444869807551896
    bitrate: 421.3347459545065

ABX on greedy VQ-CPC segmentation:

    # exp/vqcpc/english2019/test/phoneseg_greedy/abx_results.json
    abx: 28.721012318333095
    bitrate: 146.77067232697695

ABX on greedy VQ-VAE segmentation:

    # exp/vqvae/english2019/test/phoneseg_greedy/abx_results.json
    abx: 23.06589252244493,
    bitrate: 142.2438934407938

ABX on DP N-seg VQ-VAE segmentation:

    exp/vqvae/english2019/test/phoneseg_dp_nseg/abx_results.json
    abx: 19.605026309535546,
    bitrate: 142.20968273941133

ABX on L2 VQ-CPC segmentation:

    # exp/vqcpc/english2019/test/phoneseg_l2/abx_results.json
    abx: 19.556202404470035
    bitrate: 123.87736667909118

ABX on L2 VQ-VAE segmentation:

    # exp/vqvae/english2019/test/phoneseg_l2/abx_results.json
    abx: 18.53130069166441
    bitrate: 106.18019946190225


ZeroSpeech 2019 (ABX) as we vary bitrate [final]
------------------------------------------------
phoneseg_l2_durweight0.001:

    abx: 14.783352842852027
    bitrate: 297.6536026917606

phoneseg_l2_durweight1:

    abx: 16.63861353312477
    bitrate: 167.97090274884843

phoneseg_l2_durweight2:

    abx: 17.292994735999624
    bitrate: 126.52557997689566

phoneseg_l2_durweight3:

    abx: 18.5313006916644
    bitrate: 106.18019946190225

phoneseg_l2_durweight4:

    abx: 19.294567231555103
    bitrate: 95.45371409165645

phoneseg_l2_durweight10:

    abx: 22.573069255655554
    bitrate: 67.64514351141277

phoneseg_l2_durweight15:

    abx: 26.613988629208983
    bitrate: 55.86614128863321

phoneseg_l2_durweight20:

    abx: 27.826613340237405
    bitrate: 47.4691960523425

phoneseg_l2_durweight40:

    abx: 27.636218746209195
    bitrate: 32.45933238315029


phoneseg_dp_nseg_nframesperseg1:

    abx: 14.043611615570672
    bitrate: 412.2387509949519

phoneseg_dp_nseg_nframesperseg2:

    abx: 17.422246462506575
    bitrate: 212.49314905289904

phoneseg_dp_nseg_nframesperseg3:

    abx: 19.605026309535546
    bitrate: 142.20968273941133

phoneseg_dp_nseg_nframesperseg4, phoneseg_algorithm.n_min_segments=3:

    abx: 20.230498008421726
    bitrate: 106.4947841688317

phoneseg_dp_nseg_nframesperseg5, phoneseg_algorithm.n_min_segments=3:

    abx: 20.04260112435944
    bitrate: 84.95737340430989

phoneseg_dp_nseg_nframesperseg5, phoneseg_algorithm.n_min_segments=0:

    abx: 29.29963286925432
    bitrate: 84.95737340430989

phoneseg_dp_nseg_nframesperseg6, phoneseg_algorithm.n_min_segments=3:

    abx: 20.101767852580434
    bitrate: 70.50274016980106

phoneseg_dp_nseg_nframesperseg6, phoneseg_algorithm.n_min_segments=0:

    abx: 28.939557263512583
    bitrate: 70.50274016980106

phoneseg_dp_nseg_nframesperseg7, phoneseg_algorithm.n_min_segments=3:

    abx: 27.636218746209195
    bitrate: 32.45933238315029

phoneseg_dp_nseg_nframesperseg7, phoneseg_algorithm.n_min_segments=0:

    abx: 28.17615825490798
    bitrate: 60.233424750163216
