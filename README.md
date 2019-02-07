# drl-grasping

## Requirements

* pytorch 1.0 (https://pytorch.org/)
* matplotlib
* ```pip install ray```
* ```pip install gym```
* ```pip install pybullet```

## To Run

Note: Must be running Linux due to dependence on Ray package. Running the following command will spawn N remote servers (running on different cores) that uses a biased downward policy to grasp objects

```python collect.py --remotes=1 --outdir=data100K```

Once data has been collected, you can begin training off-policy DQL models:

```python parallel.py --model=dqn```
```python parallel.py --model=ddqn```
```python parallel.py --model=ddpg```
```python parallel.py --model=supervised```
```python parallel.py --model=mcre```
```python parallel.py --model=cmcre```

The command line can be used to specify a number of arguments; See parallel.py for details. 

If running a visdom server, you can replace ```parallel.py``` with ```parallel_vis.py``` to watch task execution. 

More details and instructions to come
