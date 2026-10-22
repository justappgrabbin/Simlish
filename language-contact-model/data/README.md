# Public data and attribution

Retrieved 22 October 2026. Raw source files are included in the standalone ZIP for offline reproduction.
The GitHub branch uses `python -m neural.datasets` to retrieve them in six bounded
downloads and verifies each against `data/SHA256.json`.
All three original UCI dataset pages list CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/ . Credit the original creators when
redistributing data or derived forms. These are independent collections, not
synchronized multimodal recordings or evidence of universal communication.

## Gesture Phase Segmentation

Madeo, R., Wagner, P., and Peres, S. (2013). Gesture Phase Segmentation [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5Z32C

Files: `gesture/a1_raw.csv`, `a2_raw.csv`, `a3_raw.csv`.
Retrieved from the public mirror:
https://github.com/srdjop/Gesture-Phase-Segmentation/tree/main/Dataset

The mirror presents these as the raw UCI recordings. This project has not
byte-compared them to a direct UCI download. Line endings were normalized to LF.
Hashes of the included files are in `artifacts/gesture-benchmark.json`.
Train on a1/a2; hold out a3. Windows overlap within splits, but no recording crosses
splits. No claim of a participant-independent split. Labels are gesture phases,
not emotional state or communicative intent. Coordinates are made body-relative,
and normalization is fit only on training observations.

## Japanese Vowels

Kudo, M., Toyama, J., and Shimbo, M. (1999). Japanese Vowels [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5NS47

Files: `speech/JapaneseVowels_TRAIN.ts`, `JapaneseVowels_TEST.ts`. These `.ts` files
are time-series data, not TypeScript. Retrieved from:
https://github.com/aeon-toolkit/aeon/tree/main/aeon/datasets/data/JapaneseVowels

Preserves the published 270-train / 370-test utterance split. Twelve LPC acoustic
features per frame. Nine known speakers, recorded uttering /ae/. The test is
speaker classification across utterances, not unseen-speaker identification,
speech transcription or emotional inference. Normalization uses training data only.

## LIBRAS Movement

Dias, D., Peres, S., and Bscaro, H. (2009). Libras Movement [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5GC82

File: `sign/movement_libras.data`. Retrieved from:
https://github.com/IST-DASLab/QRGD/blob/master/movement_libras.data

360 sequences, 15 movement classes, 45 two-dimensional hand-centroid positions per
sequence. The source describes this as movement types in Brazilian Sign Language;
it is not a complete signed-language corpus. Two-thirds of unique trajectories
per class go to training, with seed 7; exact duplicate trajectories are grouped
in one split. Result: 238 training / 122 testing. Row indices are saved in
`artifacts/multimodal-normalization.json`. Signer identifiers are unavailable in
this representation, so this is not signer-independent evaluation. Labels remain
opaque numeric identifiers; no invented word translations are attached.

## Other located sources (not used for training)

- Auslan high-quality signs: https://doi.org/10.24432/C5VG6R . UCI documents a
  single native signer and 22 sensor channels. Current download listings show tiny
  archive placeholders; no usable training archive was retrieved in this session.
- Free Spoken Digit Dataset: https://github.com/Jakobovski/free-spoken-digit-dataset .
  Its README lists CC BY-SA 4.0; binary WAV retrieval through the connected GitHub
  fetch action was rejected. No FSDD data is included or used.
- LibriSpeech: https://openslr.org/12/ . Large-scale read speech, CC BY 4.0.
  Not downloaded or used here.

`neural.multimodal_benchmark.prepare()` reproduces the preprocessing and splits
from these raw files without network access. Generated JSONL files are excluded
from GitHub because they duplicate the raw data; the benchmark regenerates them.
