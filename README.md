# deepq-grasping

Implements off-policy models from: https://arxiv.org/abs/1802.10264. 

Can use the robot &  environment from [here](https://github.com/google-research/google-research/tree/master/dql_grasping)

WIP. More details and instructions to come. 

## Requirements

__Note__: Must be running Linux due to dependence on Ray package for parallel policy execution.

* pytorch 1.0 (https://pytorch.org/)
* matplotlib
* ```pip install ray```
* ```pip install gym```
* ```pip install pybullet```

# To Run:

## Collect Experience

First collect experience using a biased downward policy. The following command will spawn _N_ remote servers that each run a different environment instance, and are used to collect experience in parallel.

```python collect.py --remotes=1 --outdir=data100K```

## Train a Model

Once data has been collected, you can begin training off-policy DQL models by selecting one from the list:

```python parallel.py --remotes=1 --data-dir=data100K --model=[dqn, ddqn, ddpg, supervised, mcre, cmcre]``` 

## Notes

The command line can be used to specify a number of additional arguments. See parallel.py for details. 

If running a visdom server, you can replace ```parallel.py``` with ```parallel_vis.py``` to watch task execution. 
