import pybullet_envs.bullet.kuka_diverse_object_gym_env as e


def make_env(max_steps, is_test, render):
    """Makes a new environment given a config file."""

    # Defines parameters for distributed evaluation
    env_config = {'actionRepeat':80,
                  'isEnableSelfCollision':True,
                  'renders':render,
                  'isDiscrete':False,
                  'maxSteps':max_steps,
                  'dv':0.06,
                  'removeHeightHack':True,
                  'blockRandom':0.3,
                  'cameraRandom':0,
                  'width':64,
                  'height':64,
                  'numObjects':5,
                  'isTest':is_test}

    def create():
        return e.KukaDiverseObjectEnv(**env_config)
    return create


def make_model(args, device):
    """Makes a new model given a config file."""

    # Defines parameters for network generator
    config = {'action_size':4, 'bounds':(-1, 1), 'device':device}
    config.update(vars(args))

    if args.model == 'dqn':
        from models.dqn import DQN as Model
    elif args.model == 'ddqn':
        from models.ddqn import DDQN as Model
    elif args.model == 'ddpg':
        from models.ddpg import DDPG as Model
    elif args.model == 'supervised':
        from models.supervised import Supervised as Model
    elif args.model == 'mcre':
        from models.mcre import MCRE as Model
    elif args.model == 'cmcre':
        from models.cmcre import CMCRE as Model
    else:
        raise NotImplementedError('Model <%s> not implemented' % args.model)

    def create():
        return Model(config)
    return create


def make_memory(model, buffer_size):
    """Initializes a memory structure.

    Some models require slight modifications to the replay buffer,
    such as sampling a full episode, setting discounted rewards, or
    altering the action. in these cases, the base.memory module gets
    overridden in the respective files.
    """

    if model == 'supervised':
        from models.supervised import Memory
    elif model == 'mcre':
        from models.mcre import Memory
    elif model == 'cmcre':
        from models.cmcre import Memory
    else:
        from models.base.memory import BaseMemory as Memory

    return Memory(buffer_size)
