# Installation

## rl-starter-files

```bash
cd rl-starter-files/rl-starter-files
pip3 install -r requirements.txt

## gym_minigrid

```bash
git clone https://github.com/Farama-Foundation/Minigrid.git
cd Minigrid
git checkout 116fa65bf9584149f9a23c2b61c95fd84c25e467
pip3 install -e .

## torch-ac

```bash
cd torch-ac
pip3 install -e .

## Train the SI2E framework
```bash
cd rl-starter-files/rl-stater-files
source run_si2e.sh
