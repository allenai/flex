# About

Baseline from
```
@inproceedings{
Bao2020Few-shot,
title={Few-shot Text Classification with Distributional Signatures},
author={Yujia Bao and Menghua Wu and Shiyu Chang and Regina Barzilay},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=H1emfT4twB}
}
```

The `bao/` subdirectory was adapted from <https://github.com/YujiaBao/Distributional-Signatures/tree/c613ed070af3e7ae4967b9942fde16864af28cde>.

`conf/bao.yaml` args were obtained from running [bin/our.sh](https://github.com/YujiaBao/Distributional-Signatures/blob/c613ed070af3e7ae4967b9942fde16864af28cde/bin/our.sh)

# Instructions

Default parameters used by [Hydra](https://hydra.cc) are set in `conf/bao.yaml`.

## Installation
- Clone the main repo
- Download and install Miniconda
- Change to this directory
- Create the `bao` virtual environment: `conda env create -n bao -f environment.yml`
- Activate the virtual environment: `conda activate bao`
- Install the fewshot package: `pip install -e ../../`
- Download the spacy package data: `python -m spacy download en`

## Training

To train on dataset `{DATASET}` and shot `{SHOT}`, from the project root run
```
python baselines/bao/train.py dataset={DATASET} shot={SHOT} hydra.run.dir=outputs/train/{SHOT}/{DATASET}`
```

To reproduce results from the above paper, run the above command for 1-shot and 5-shot, with `{DATASET}` corresponding each of:
- amazon
- fewrel
- huffpost
- newsgroup
- reuters

## Testing

Based on training outputs, edit `conf/model/{DATASET_SHOT}.yaml` and from the project root run
```
python baselines/bao/test.py model={DATASET_SHOT} challenge={CHALLENGE_SHOT}
```

`{CHALLENGE_SHOT}` should correspond to a challenge (see project README.md).

By default, this will save to `outputs/eval/{CHALLENGE_SHOT}/model={DATASET_SHOT}/predictions.json`. The output directory configuration is specified in `conf/test.yaml` and can be overridden via CLI.

See main project README.md for details on scoring the predictions file.
