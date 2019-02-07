# drl-grasping

Implements off-policy models from: https://arxiv.org/abs/1802.10264. 

Work in progress; more details and instructions to come. 

## Requirements

__Note__: Must be running Linux due to dependence on Ray package for parallel policy execution.

* pytorch 1.0 (https://pytorch.org/)
* matplotlib
* ```pip install ray```
* ```pip install gym```
* ```pip install pybullet```

## To Run:

Running the following command will spawn N remote servers (running on different cores) that uses a biased downward policy to grasp objects

```python collect.py --remotes=1 --outdir=data100K```

Once data has been collected, you can begin training off-policy DQL models by selecting one from the list:

```python parallel.py --model=[dqn, ddqn, ddpg, supervised, mcre, cmcre]``` 

The command line can be used to specify a number of arguments; See parallel.py for details. 

If running a visdom server, you can replace ```parallel.py``` with ```parallel_vis.py``` to watch task execution. 
