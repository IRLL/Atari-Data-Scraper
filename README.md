# data_scraper_clean
Clone the repo.
Pip install any required libraries (I plan to create a requirements.txt file soon)

To collect the minimum information from a training algorithm, run the following command:
python stable_baselines_a2c.py --num_envs # --num_steps # --algo <DQN, A2C, PPO2> --environment <Atari game environment>

Additional flags:
--lives include if the game environment has lives
--collect_extra include if you want to collect additional information such as location of items in the game (i.e. the agent), and computed rewards for plotting
--save include if you want the training model to be saved for future use
--model <Model Name> if you want to use a pre-existing and pre-trained model 

Example Usage:
python stable_baselines_a2c.py --num_envs 1 --num_steps 2000 --algo DQN --save --environment PongNoFrameskip-v4 --collect_extra

python stable_baselines_a2c.py --lives --num_envs 2 --num_steps 20 --algo A2C --save --environment MsPacmanNoFrameskip-v4 

To collect additional information from an existin CSV file, run the following command:
python collector_additional_information.py --filepath <directory that has csv> --num_steps <same as used in training> --num_envs <same as used in training> --algo <same as used in training> --env_name <same as used in training> 

Example Usage:
python collector_additional_information.py --filepath A2C_Pong_data_2021-02-28_14-32-31 --num_envs 2 --num_steps 20 --algo A2C --env_name Pong
