# MiniGrid Evaluation

Follow the steps below to set up the necessary environments and dependencies for the SI2E framework.

## rl-starter-files

First, install the dependencies for the rl-starter-files.

```bash
cd rl-starter-files/rl-starter-files
pip3 install -r requirements.txt
```

## gym_minigrid

Clone the gym_minigrid repository and install it.

```bash
git clone https://github.com/Farama-Foundation/Minigrid.git
cd Minigrid
git checkout 116fa65bf9584149f9a23c2b61c95fd84c25e467
pip3 install -e .
```

## torch-ac

Install the torch-ac library.

```bash
cd torch-ac
pip3 install -e .
```

## Train the SI2E framework

Finally, run the following script to train the SI2E framework.

```bash
cd rl-starter-files/rl-starter-files
source run_si2e.sh
```
