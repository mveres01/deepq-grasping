# drl-grasping

## Some findings:

- Constraining CEM search to (-1, 1) helps stabilize network
- When using CEM and supervised learning, since we used an initial random policy that emphasized moving the gripper downwards, there may be a significant amount of space in the z-dim that we haven't explored. Can we protect against sampling in this space by limiting variables to spacific ranges? 
- Should we scale the supervised values to be between [-1, 1]? Otherwise, compounding the values will give very large values along the z-dim compared to the X, Y, and rotation variables
- CEM saturates easy, so try and prevent it from sticking to a particular value quickly early on
