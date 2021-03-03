# Atari Data Scraper
This is a tool that can be used to collect data during the training of Atari agents in OpenAI Gym and Stable Baselines. The data is output as a CSV file that can then be used in creating visualizations as seen in the sample Jupyter notebook above. 

Getting Started
------

1. Clone the repo `https://github.com/j-vent/data_scraper_clean.git `

2. Create a virtual environment and install the requirements using  `pip install -r requirements.txt`

Running the code
------

To collect the minimum information (state, action, lives, reward) from an agent while training, run the following command:
``` python data_scraper.py --num_envs # --num_steps # --algo [DQN, A2C, PPO2] --environment [Atari game environment] ```

Additional flags:

⋅⋅⋅--lives include if the game environment has lives
⋅⋅⋅--collect_extra include if you want to collect additional information such as location of items in the game (i.e. the agent), and computed rewards for plotting. See below on how to run this separately.
⋅⋅⋅--save include if you want the training model to be saved for future use
⋅⋅⋅--model [Model Name] if you want to use a pre-existing and pre-trained model, some examples can be found in the models folder

Example Usage:
``` python data_scraper.py --num_envs 1 --num_steps 2000 --algo DQN --save --environment PongNoFrameskip-v4 --collect_extra ```

```python data_scraper.py --lives --num_envs 2 --num_steps 20 --algo A2C --save --environment MsPacmanNoFrameskip-v4 ```


To collect additional information (location of agents/items, distance between item and agent, computed rewards) from an existing CSV file, run the following command:
``` python collector_additional_information.py --filepath [directory that has csv] --num_steps [same as used in training] --num_envs [same as used in training] --algo [same as used in training] --env_name [same as used in training] ```

Example Usage:
``` python collector_additional_information.py --filepath A2C_Pong_data_2021-02-28_14-32-31 --num_envs 2 --num_steps 20 --algo A2C --env_name Pong ```


Additional Notes
------

Currently data scraper works in the Atari environments, MsPacman and Pong. To add additional environments, please 
see comments in colour_detection.py and visit https://stable-baselines.readthedocs.io/en/master/guide/examples.html#id2 for more details.

Currently the DQN, A2C and PPO2 algorithms are supported. For more information please visit: https://stable-baselines.readthedocs.io/en/master/modules/base.html and https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html

