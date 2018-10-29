# drl-grasping

## Some findings:

- Constraining CEM search to (-1, 1) helps stabilize network
- When using CEM and supervised learning, since we used an initial random policy that emphasized moving the gripper downwards, there may be a significant amount of space in the z-dim that we haven't explored. Can we protect against sampling in this space by limiting variables to spacific ranges? 
- Should we scale the supervised values to be between [-1, 1]? Otherwise, compounding the values will give very large values along the z-dim compared to the X, Y, and rotation variables
- CEM saturates easy, so try and prevent it from sticking to a particular value quickly early on
- DDPG seems to need a lower gamma value (i.e. ~0.85) in order to converge on stable values, in comparison to DQN and DDQN which use values >= 0.9
- Need a large L2 penalty (i.e. 0.01) in supervised learning to prevent overfitting 

- Important: When using supervised learning, constrain the optimization on z-dim; otherwise, the model may choose z's that take the gripper away from the object, even though the dataset contains very few instances of this behaviour
