from typing import List

import gym as gym
from GA import GA
from MODEL_CNN import ModelCNN

def load_weights_from_file(name) -> List[float]:
    with open(name, "r") as f:
        return [float(weight) for weight in f.readlines()]



env = gym.make('ALE/AirRaid-v5',
               obs_type="grayscale",
               render_mode='human',
               frameskip=7,
               repeat_action_probability=0)
env.metadata['video.frames_per_second'] = 120

MODEL = ModelCNN()

best_weights = load_weights_from_file("best_weights_1425.txt")
total_reward = MODEL.play(env, best_weights)
print(f"Total reward: {total_reward}")
