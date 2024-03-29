# DMControl Evaluation
Follow the steps below to set up the necessary environments and dependencies for the SI2E framework.

- The two packages below are not included in `conda_env.yml`.
- Install MuJoCo 2.1.0.
- Install `dm_control`.

## Install the following libraries:

```bash
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

## Install dependencies

```bash
conda env create -f conda_env.yml
conda activate drqv2
```

## Train the SI2E framework

```bash
python train.py task=quadruped_walk do_vcse=True
```
