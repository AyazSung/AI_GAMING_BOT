# Air Raid
![](https://www.gymlibrary.dev/_images/air_raid.gif)

We create AI that play Air Raid. In this game, you control a ship that can move sideways. You must protect two buildings (one on the right and one on the left side of the screen) from flying saucers that are trying to drop bombs on them. To play this game,  AI interact with an game environment that is part of the [Atari environments](https://www.gymlibrary.dev/environments/atari/).
# How we achive better AI gaming?
We have CNN that play in our game. This CNN input layer take in the current frame of the game. Output layer of this CNN contain probabilities for the next actions.
Actions can be following:
|Num|Action   |
|---|---------|
|0  |NOOP     |
|1  |FIRE     |
|2  |RIGHT    |
|3  |LEFT     |
|4  |RIGHTFIRE|
|5  |LEFTFIRE |

For our AI to improve, we run the genetic algorithm that will try select the best weights for the model.
# How this genetic algorithm works exactly?
For this genetic algorithm, chromosome is a set of parameters of CNN and gene is some weight in CNN model.
Fitness function returns the game reward. For this reason, the greater the value of the fitness function means the better model.
