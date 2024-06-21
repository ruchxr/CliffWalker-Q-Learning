A Q Learning Implementation of the Frozen Lake OpenAI for Reinforcement Learning

Game Description

The game starts with the player at location [0,0] of the frozen lake grid world with the goal located at far extent of the world e.g. [3,3] for the 4x4 environment.
Holes in the ice are distributed in set locations when using a pre-determined map or in random locations when a random map is generated.
The player makes moves until they reach the goal or fall in a hole.
The lake is slippery (unless disabled) so the player may move perpendicular to the intended direction sometimes.
Randomly generated worlds will always have a path to the goal.

Action Space (env.action_space)

The action shape is (1,) in the range {0, 3} indicating which direction to move the player.
0: Move left
1: Move down
2: Move right
3: Move up


Observation Space (env.observation_space)

The observation is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0).
For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15. The number of possible observations is dependent on the size of the map.
The observation is returned as an int().

Starting State

The episode starts with the player in state [0] (location [0, 0]).

Rewards

Reward schedule:
Reach goal: +1
Reach hole: 0
Reach frozen: 0

Episode End

The episode ends if the following happens:

Termination:

The player moves into a hole.
The player reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, max(ncol)-1]).

Truncation (when using the time_limit wrapper):

The length of the episode is 100 for 4x4 environment, 200 for FrozenLake8x8-v1 environment.

The Q-Table after the training is 

[[ 0.02772213  0.02879532  0.02842857  0.02513591]

 [ 0.01587821  0.01837487  0.02378215  0.02843866]
 
 [ 0.03954103  0.03606027  0.04317324  0.02796177]
 
 [ 0.01717991  0.01247999  0.0152297   0.02224992]
 
 [ 0.03888076  0.02616801  0.02381908  0.01624075]
 
 [ 0.          0.          0.          0.        ]
 
 [ 0.03860827  0.03375704  0.05930751  0.01328616]
 
 [ 0.          0.          0.          0.        ]
 
 [ 0.02307825  0.03912429  0.0332827   0.05364913]
 
 [ 0.03889504  0.08531859  0.06413145  0.04529939]
 
 [ 0.11702617  0.08587701  0.07285017  0.03961923]
 
 [-0.04217655  0.          0.          0.        ]
 
 [ 0.          0.          0.          0.        ]
 
 [ 0.06928331  0.0980362   0.11490705  0.04927568]
 
 [ 0.10789541  0.20363864  0.14964606  0.13325162]
 
 [-0.83928232 -0.97675271 -1.47584948 -1.65611597]]

Each row depicts the q-value of a particular action in a particular state calculated using the Bellman Equation.
