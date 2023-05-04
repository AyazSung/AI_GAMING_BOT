# Air Raid
![](https://www.gymlibrary.dev/_images/air_raid.gif)

We create AI gaming bot that play Air Raid. In this game, you control a ship that can move sideways. You must protect two buildings (one on the right and one on the left side of the screen) from flying saucers that are trying to drop bombs on them. To play this game, gaming bot interact with an game environment that is part of the [Atari environments](https://www.gymlibrary.dev/environments/atari/).
# How we achive better AI gaming bot?
We have CNN that play in our game. Code for CNN structure:
![image](https://user-images.githubusercontent.com/90756690/236259159-9e991da8-09ff-427b-bab4-b92082de1583.png)

This CNN input layer take in the current frame of the game. Output layer of this CNN contain probabilities for the next actions.
Actions can be following:
|Num|Action   |
|---|---------|
|0  |NOOP     |
|1  |FIRE     |
|2  |RIGHT    |
|3  |LEFT     |
|4  |RIGHTFIRE|
|5  |LEFTFIRE |

For our gaming bot to improve, we run the genetic algorithm that will try select the best weights for the model.
# How this genetic algorithm works exactly?
For this genetic algorithm, chromosome is a set of parameters of CNN and gene is some weight in CNN model.
Fitness function returns the game reward. For this reason, the greater the value of the fitness function means the better model.
# How to run this project?
First, you should download Main.ipynb file. Second, you should run this code. To run code, you should install needed libraries by running third cell:
![image](https://user-images.githubusercontent.com/90756690/236258583-dc131e1d-ae3e-42d3-bfc2-4d4385ee69ed.png)
So, just uncommented this cell before your first run of this project.
