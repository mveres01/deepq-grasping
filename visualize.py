"""
File from:
https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py
"""

from graphviz import Digraph
import torch
from torch.autograd import Variable
from collections import namedtuple


Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


# For traces

def replace(name, scope):
    return '/'.join([scope[name], name])


def parse(graph):
    scope = {}
    for n in graph.nodes():
        inputs = [i.uniqueName() for i in n.inputs()]
        for i in range(1, len(inputs)):
            scope[inputs[i]] = n.scopeName()

        uname = next(n.outputs()).uniqueName()
        assert n.scopeName() != '', '{} has empty scope name'.format(n)
        scope[uname] = n.scopeName()
    scope['0'] = 'input'

    nodes = []
    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}
        attrs = str(attrs).replace("'", ' ')
        inputs = [replace(i.uniqueName(), scope) for i in n.inputs()]
        uname = next(n.outputs()).uniqueName()
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': n.kind(),
                             'inputs': inputs,
                             'attr': attrs}))

    for n in graph.inputs():
        uname = n.uniqueName()
        if uname not in scope.keys():
            scope[uname] = 'unused'
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': 'Parameter',
                             'inputs': [],
                             'attr': str(n.type())}))

    return nodes


def make_dot_from_trace(trace):
    """ Produces graphs of torch.jit.trace outputs
    Example:
    >>> trace, = torch.jit.trace(model, args=(x,))
    >>> dot = make_dot_from_trace(trace)
    """
    torch.onnx._optimize_trace(trace, False)
    graph = trace.graph()
    list_of_nodes = parse(graph)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for node in list_of_nodes:
        dot.node(node.name, label=node.name.replace('/', '\n'))
        if node.inputs:
            for inp in node.inputs:
                dot.edge(inp, node.name)
    return dot


def collect_experience(env, memory, print_status_every=25, data_dir=None):

    import matplotlib.pyplot as plt
    from utils import ContinuousDownwardBiasPolicy

    # Initialize the experience replay buffer with memory
    policy = ContinuousDownwardBiasPolicy()

    total_step = 0
    while True and not memory.is_full:

        terminal = False
        state = env.reset()
        state = state.transpose(2, 0, 1)[np.newaxis]

        step = 0
        while not terminal and not memory.is_full:

            action = policy.sample_action(state, .1)

            next_state, reward, terminal, _ = env.step(action)
            next_state = next_state.transpose(2, 0, 1)[np.newaxis]

            step = step + 1

        if reward > 0:

            memory.add(state, action, reward, next_state, terminal, step)

            '''
            plt.subplot(1, 2, 1)
            plt.imshow(state[0].transpose(1, 2, 0))
            plt.subplot(1, 2, 2)
            plt.imshow(next_state[0].transpose(1, 2, 0))
            plt.show()
            '''

            print('Memory capacity: %d/%d' % (memory.cur_idx, memory.max_size))

            total_step = total_step + 1
            if total_step % 50 == 0:
                print('Saving current experience')
                memory.save(data_dir)


if __name__ == '__main__':

    import numpy as np
    from utils import ReplayMemoryBuffer
    import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

    """
    ENV OPTIONS
    -----------
    urdfRoot=pybullet_data.getDataPath(),
    actionRepeat=80,
    isEnableSelfCollision=True,
    renders=False,
    isDiscrete=False,
    maxSteps=8,
    dv=0.06,
    removeHeightHack=False,
    blockRandom=0.3,
    cameraRandom=0,
    width=48,
    height=48,
    numObjects=5,
    isTest=False
    """

    max_buffer_size = 10000
    render_env = False
    remove_height_hack = True
    use_precollected = True
    data_dir = 'successful_grasps'
    max_num_steps = 8
    num_rows = num_cols = 64
    action_size = 4
    state_space = (3, num_rows, num_cols)
    action_space = (action_size,)

    memory = ReplayMemoryBuffer(max_buffer_size, state_space, action_space)

    env = e.KukaDiverseObjectEnv(height=num_rows,
                                 width=num_cols,
                                 removeHeightHack=remove_height_hack,
                                 maxSteps=max_num_steps,
                                 renders=render_env,
                                 isDiscrete=False)

    collect_experience(env, memory, print_status_every=100, data_dir=data_dir)
    memory.save(data_dir)

    #memory.load(data_dir, max_buffer_size)

    print('memory: ', np.sum(memory.reward))
