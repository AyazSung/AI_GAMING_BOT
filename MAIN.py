from typing import List

import gym as gym
from GA import GA
from MODEL_CNN import ModelCNN

# install necessary libraries and packages
# !pip install gym[atari]
# !pip install gym[accept-rom-license]


# function to write to file "total_reward" and list of weights.
# Possible to load and test the model.
def save_weights_to_file(reward: int, weights: List[float]):
    # save best weights to txt file
    with open(f"best_weights_{reward}.txt", "w") as f:
        for weight in weights:
            f.write(str(weight) + "\n")


# loading weights from file "name"
def load_weights_from_file(name) -> List[float]:
    with open(name, "r") as f:
        return [float(weight) for weight in f.readlines()]


# creating initial game environment
env = gym.make('ALE/AirRaid-v5',
               obs_type="grayscale",
               frameskip=7,
               repeat_action_probability=0)

# set fps=120 for rendering
env.metadata['video.frames_per_second'] = 120

# creating CNN model for GA
MODEL = ModelCNN()
# Genetic Algorithm itself
GA = GA(population_size=10,
        gen_limit=10,
        p_crossover=0.5,
        mut_prob=0.7,
        p_mutation=0.8,
        env=env,
        number_of_weights=MODEL.get_number_of_weights(),
        model=MODEL)

# running GA and finding best set of weights
best_weights = GA.find_best_weights()

# Modify only render_mode parameter of replay_env
replay_env = gym.make('ALE/AirRaid-v5',
                      obs_type="grayscale",
                      render_mode='human',
                      frameskip=7,
                      repeat_action_probability=0)

# calculate reward with the set of weights "best_weights"
total_reward = MODEL.play(replay_env, best_weights)

print(f"Total reward: {total_reward}")
save_weights_to_file(int(total_reward), best_weights)
