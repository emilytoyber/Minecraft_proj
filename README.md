# Minecraft_proj

Environment checked on a RHEL4.8.5 cluster.

## Download conda
https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/ (if not downloaded)

## Install
```git clone https://github.com/emilytoyber/Minecraft_proj.git```

## Setup environments
```
cd Minecraft_proj/
conda env create -f imi_env.yaml
```

## Running

*The files state_to_transition.json, pov_cluster_to_transition_with_30K_pov.ipynb and files in jsons_actions.zip are created in preprocess_data.ipynb*

In order to run the training part of the models, activate the relevant environment and cd to the respective directory (agents/MineRL2020/ for imi_env), then run ```python train.py``` for imitation, run agents/basic_BC.ipynb for behavioral cloning algorithm.

For testing, running the respective colab notebook is needed (colab is needed because of the virtual frame buffer that exists in google colab, other platforms with a virtual frame buffer may also be optional) after cloning the repository to colab and uploading the relevant trained model (basic_BC.ipynb includes both training and testing of BC, Imitation_test.ipynb includes the test of the Imitation agent).

Random_Agent.ipynb runs the random baseline of the environment, scripted_ironpickaxe.ipynb runs the scripted version of our algorithm.

Our main new algorithm uses a clustering model of DBSCAN+KNN trained by running POVs_clustering.ipynb (it is now limited to only 30K POVs per player in the data, because the code crashed due to not having enough resources in colab, you may drop the slicing of 30K if you have more resources).

Testing this agent is done by running choose_actions.ipynb.

## Evaluation
Run comparison_environment.ipynb after uploading the respective jsons obtained from the colab notebooks.

### Adding new algorithms
In order to add new algorithms to the comparison environment, one has to export a json file of execution statistics from the algorithm. Json will be constructed in the following way: 
```
stats['runtime'].append(time() - start)
stats['reward'].append(reward_sum)
stats['reward_at'].append(rewards)
```
reward_sum is equal to the total reward of the episode, rewards is a list of tuples (steps, reward) where reward is the immediate (non-zero) reward of the action.

## Project Explanation
Our project is a comparison project between different algorithms in Artificial Intelligence trained and tested on environments from the MineRL competitions.
Some of the algorithms are submissions of different teams from the 2019-2022 competitions.

![Alt Text](https://github.com/emilytoyber/Minecraft_proj/blob/main/avg.gif)
