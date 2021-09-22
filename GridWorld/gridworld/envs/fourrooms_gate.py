"""
Fourrooms with gates
One gate between two adjacent rooms
    Gate type: coin/water/wall
    Four gate with same color
    gates_pos = [25, 51, 62, 88]
Transition dynamic is basic, i.e. no further step and direction turning.
Model description is a vector with dim=4 in field ('coin', 'water', 'wall')

Init pos and goal are at diagonal rooms.
Wall number is at most 1, so the agent can reach the goal.
"""
from .fourrooms import *
from ..utils.env_utils.rtfm_util import *
import numpy as np
from vocab import Vocab
from ..utils.env_utils.fourrooms_util import *


class FourroomsGateState(FourroomsBaseState):
    def __init__(self, position_n, current_steps, goal_n, done, num_pos, cum_reward, description, gate_list):
        # description is a 4-dim list in field ('coin', 'water', 'wall')
        # gate_list is a 4-dim list in field (0, 1), 0 means a coin gate is eaten
        super(FourroomsGateState, self).__init__(position_n, current_steps, goal_n, done, num_pos)
        self.cum_reward = cum_reward
        self.descr = description
        self.gate_list = gate_list

    @classmethod
    def frombase(cls, base: FourroomsBaseState, cum_reward, description, gate_list):
        return cls(base.position_n, base.current_steps, base.goal_n, base.done, base.num_pos, cum_reward,
                   description, gate_list)

    @property
    def to_obs(self):
        if len(self.gate_list) == 0:  # initial 0
            return 0
        descr_value = [0 if gate == 'coin' else 1 if gate == 'water' else 2 for gate in self.descr]
        multiplier_descr = np.dot(descr_value, [3 ** i for i in range(4)])
        multiplier_gate = np.dot(self.gate_list, [2 ** i for i in range(4)])
        multiplier = multiplier_descr * 16 + multiplier_gate
        return np.array(multiplier * self.num_pos + self.position_n)

    def to_tuple(self):
        return self.position_n, self.current_steps, self.goal_n, self.done, tuple(self.gate_list)


class FourroomsGate(FourroomsBase):
    def __init__(self, Model=None, max_epilen=100, init_pos=None, goal=None, seed=None, mode='train', gate_list=None):
        super(FourroomsGate, self).__init__(max_epilen, goal, seed)
        # Model: dict pos -> gate_class type
        self.init_pos = init_pos
        self.goal = goal
        self.Model = Model
        self.mode = mode  # 'train' or 'test'
        self.gate_list = gate_list or [1, 1, 1, 1]
        self.state = FourroomsGateState(0, 0, 0, False, 0, [], [], [])
        self.open = False
        self.random_init = False if init_pos is not None else True
        self.random_goal = False if goal is not None else True
        self.random_model = False if Model is not None else True
        self.fix_gate_list = gate_list if gate_list else False
    
    def reset(self):
        super().reset()
        # random chooce initial position and goal, in diagonal rooms
        if self.random_init:
            init_room = np.random.choice([0, 1, 2, 3])
            init_pos = np.random.choice(rooms_pos[init_room])
            self.init_pos = init_pos
        else:
            room_idx = [self.init_pos in room for room in rooms_pos]
            init_room = int(np.where(room_idx)[0])
        if self.random_goal:
            goal_pos = np.random.choice(rooms_pos[3 - init_room])
            self.goal = goal_pos
        self.state.position_n = self.init_pos
        self.state.goal_n = self.goal
        
        if self.fix_gate_list:
            self.gate_list = deepcopy(self.fix_gate_list)
        else:
            self.gate_list = [1, 1, 1, 1]
        if self.random_model:
            if self.mode == 'train':
                self.Model = np.random.choice(train_models)
            else:
                self.Model = np.random.choice(test_models)
        self.state = FourroomsGateState.frombase(self.state, [], self.todecr(), self.gate_list)
        return self.state.to_obs
    
    def todecr(self):
        descr = [self.Model[pos] for pos in gates_pos]
        return descr
    
    def step(self, action):
        if self.state.done:
            raise Exception("Environment should be reseted")
        currentcell = self.tocell[self.state.position_n]
        try:
            nextcell = tuple(currentcell + self.directions[action])
        except TypeError:
            nextcell = tuple(currentcell + self.directions[action[0]])
        
        if not self.occupancy[nextcell] and self.Model.get(self.tostate[nextcell], None) != 'wall':
            currentcell = nextcell
        position_n = self.tostate[currentcell]
        
        if position_n == self.state.goal_n:
            reward = 10
        elif position_n in gates_pos and self.gate_list[gates_pos.index(position_n)] == 1:
            if self.Model[position_n] == 'water':
                reward = -10
            else:  # coin
                reward = 10
                self.gate_list[gates_pos.index(position_n)] = 0
            self.state.gate_list = self.gate_list
        else:
            reward = -0.1
        
        self.state.current_steps += 1
        self.state.done = (position_n == self.state.goal_n) or (self.state.current_steps >= self.max_epilen)
        self.state.position_n = position_n
        self.state.cum_reward.append(reward)
        info = {}
        
        if self.state.done:
            best = best_return(self.Model, self.init_pos, self.goal)
            r = np.sum(self.state.cum_reward)
            info = {'episode': {'r': r, 'l': self.state.current_steps},
                    'win': True if self.state.position_n == self.state.goal_n else False,
                    'best': best,
                    'regret': best - r}
        
        return self.state.to_obs, reward, self.state.done, info
    
    def render(self, mode=0):
        blocks = []
        for i in range(4):
            if self.gate_list[i] == 0:
                continue
            x, y = self.tocell[gates_pos[i]]
            blocks.append(self.make_block(x, y, (0, 1, 0)))
        blocks.extend(self.make_basic_blocks())
        return self.render_with_blocks(self.origin_background, blocks)
    
    def color_render(self):
        blocks = []
        for i in range(4):
            if self.gate_list[i] == 0:
                continue
            x, y = self.tocell[gates_pos[i]]
            if self.Model[gates_pos[i]] == 'coin':
                color = (1, 1, 0)
            elif self.Model[gates_pos[i]] == 'water':
                color = (0, 1, 0)
            else:
                color = (1, 0, 1)
            blocks.append(self.make_block(x, y, color))
        blocks.extend(self.make_basic_blocks())
        return self.render_with_blocks(self.origin_background, blocks)


class FourroomsRTFM(FourroomsGate):
    def __init__(self, featurizer, max_iter=1000, max_placement=1, max_name=8, max_inv=10, max_wiki=80, max_self=40, mode='train', modeuse='default'):
        self.featurizer = featurizer
        self.max_iter = max_iter
        self.max_placement = max_placement
        self.max_name = max_name
        self.max_inv = max_inv
        self.max_wiki = max_wiki
        self.max_self = max_self

        if modeuse == "default":
            self.mode = mode
        else:
            self.mode = modeuse

        self.vocab = Vocab(['pad', 'eos', ''])
        self.vocab.word2index(wordlist, train=True)
        super().__init__(mode=self.mode)
        self.iter = 0
        self.observation_space = self.featurizer.get_observation_space(self)

    def reset(self):
        super().reset()
        self.iter = 0
        return self.featurizer.featurize(self)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.iter += 1
        return self.featurizer.featurize(self), reward, done, info
    
    def read_model(self):
        sentence = [self.Model[pos] for pos in gates_pos]
        return sentence
    
    def read_pos(self, row, col, eos, pad):
        names = []
        lens = []
        pos = self.tostate.get((row, col), -1)
        if pos == self.state.position_n:
            name, length = read_things(self.vocab, 'agent', self.max_name, eos, pad)
            names.append(name)
            lens.append(length)
        if pos == self.state.goal_n:
            name, length = read_things(self.vocab, 'goal', self.max_name, eos, pad)
            names.append(name)
            lens.append(length)
        if self.occupancy[row, col]:
            name, length = read_things(self.vocab, 'wall', self.max_name, eos, pad)
            names.append(name)
            lens.append(length)
        if pos in gates_pos and self.gate_list[gates_pos.index(pos)] == 1:
            name, length = read_things(self.vocab, 'gate', self.max_name, eos, pad)
            names.append(name)
            lens.append(length)
        return names, lens
