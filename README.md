# DQN
This is a basic implementation of Deep Q Learning. Currently we have implemented a linear version, i.e. we are taking environment state 
as input to a linear neural network, rather than using a convolutional network on raw pixels.

Soon, we aim to implement:
* Convolutional DQN trained on raw pixels for Atari games, as in the original paper
* DDQN
* Prioritized Experience Replay

## To run
To begin, setup [OpenAI gym](https://gym.openai.com/) and install the packages in `requirements.txt`.

We have an example script which trains a model on the CartPoleSwingUp environment (this requires gym <= 0.9.4).
Run `python -m examples.cartpoleswingup_linear` in the top-level directory.

## Results
The best models trained on each env are present in `results/models/`. There you will find the saved pytorch model as a `.pth` file and
a graph comparing the reward per episode against random play.

**Linear models**

| Model | Env             |      Score      |
|-------|-----------------|:---------------:|
| DQN | CartPole-v1     |  500            |
| DQN | CartPoleSwingUp\* |  700            |

\**Note: Training on CartPoleSwingUp is very unstable, even when training for millions of frames and with a large (100000) capacity memory. 
It is not clear why this is the case.*

**Conv model**

| Model | Env | Score |
|-------|-----|:-----:|
