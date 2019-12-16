## Flappy Bird AI

  This repository is a Pytorch DQN implementation with Expecimax algorithm making decision. The work referred to [1](https://github.com/yenchenlin/DeepLearningFlappyBird), [2](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch) and [3](https://github.com/xmfbit/DQN-FlappyBird). Thanks!



#### Creative ideas

1. Reward *= (1.05 ** score)
2. Modify the replay memory to record the whole game. Expectimax algorithm is based on it.



#### Usage

1. Install all the requirements by

   `pip install -r requirements.txt`

2. Train the new network by the following. If you use the pretrained model, set --use_model to True.

   `python -u dqn_ai.py --mode train --model_path /your/model/path/ --model_name name.pkl --use_model False` 

3. Test the network by 

   `python -u dqn_ai.py --mode test --model_path /your/model/path/ --model_name name.pkl --use_model True` `

  You are able to modify the training parameters in start() of dqn_ai.py.

