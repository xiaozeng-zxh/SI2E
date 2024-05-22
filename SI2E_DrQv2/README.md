# Instructions

- The two packages below are not included in `conda_env.yml`.
- Install MuJoCo 2.1.0.
- Install `dm_control`.

## Install the following libraries:

```bash
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

## Install dependencies

```bash
conda env create -f conda_env.yml
conda activate si2e

## Train the SI2E framework

```bash
python train.py
