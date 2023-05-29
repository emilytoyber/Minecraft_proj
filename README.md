# Minecraft_proj

Environment checked on a RHEL4.8.5 cluster.

## Download conda
https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/ (if not downloaded)

## Install
```git clone https://github.com/emilytoyber/Minecraft_proj.git```

## Setup environments
```
cd Minecraft_proj/
conda env create -f bc_env.yaml
conda env create -f imi_env.yaml
```

## Running
In order to run the training part of the models, activate the relevant environment and cd to the respective directory (agents/MineRL2020/ for imi_env and agents/minecraft_bc/ for bc_env), then run ```python train.py```.
For testing, running the respective colab notebook is needed (colab because of the virtual frame buffer in google colab) after cloning the repository to colab.

## Evaluation
Run comparison_environment.ipynb after uploading the respective jsons obtained from the colab notebooks.

## Project Explanation
Our project is a comparison project between different algorithms in Artificial Intelligence trained and tested on environments from the MineRL competitions.
Most of the algorithms are submissions of different teams from the 2019-2022 competitions.

This README still needs the final third algorithm.
