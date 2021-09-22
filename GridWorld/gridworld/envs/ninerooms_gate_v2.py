"""
Ninerooms Game with gates -- v2
All gates are in one color, whose type is given by the description.

gate_class:
    'all': whole 12 gates,
    '8gates': 8 gates
    '6gates': 6 gates
    '4gates': 4 gates
    'no_gate': no gate
diag:
    True: init_pos and goal are in diagonal rooms
    False: init_pos and goal are in different rooms
easy_env: affect train_model_gen()
    'no_easy': default config where true model can be learned
    'one_env': only one env in training set, gate_class must be 'all'
    'cross_type': each gate_class has different types
    'no_pair': each pair gates are different in >=1 env
"""

from .ninerooms_gate import *
from ..utils.env_utils.ninerooms_util import *

class NineroomsGateV2State(NineroomsGateState):
    def __init__(self, position_n, current_steps, goal_n, done, num_pos, gatedict, cum_reward: list, descr=None):
        # gatedict: gate_pos -> bool (exist or not)
        super(NineroomsGateV2State, self).__init__(position_n, current_steps, goal_n, done, num_pos, gatedict, cum_reward, descr)

    def to_obs(self):
        if self.gatedict is None or len(self.gatedict) == 0:
            return 0
        gate_list = [(k, v) for k, v in self.gatedict.items()]
        gate_list.sort(key=lambda ele: ele[0])
        mult_list = [1 if v else 0 for (_, v) in gate_list]
        multiplier = np.dot(mult_list, [2 ** i for i in range(len(mult_list))])
        return multiplier * self.num_pos + self.position_n

    @classmethod
    def frombase(cls, base:NineroomsGateState):
        return cls(base.position_n, base.current_steps, base.goal_n, base.done, base.num_pos, base.gatedict, base.cum_reward, base.descr)


class NineroomsGateV2(NineroomsGate):
    def __init__(self, Model=None, max_epilen=100, init_pos=None, goal=None, gate_class='all', easy_env='no_easy',
                 diag=False, seed=None, mode='train'):
        super(NineroomsGateV2, self).__init__(Model=Model, max_epilen=max_epilen, init_pos=init_pos, goal=goal,
                                              seed=seed, mode=mode, easy_dy=False, easy_env=False, fix_pos=False)
        self.state = NineroomsGateV2State.frombase(self.state)
        self.gate_class = gate_class
        self.diag = diag
        self.easy_env = easy_env
        self.init_gatedict = self.init_gate()
        
    def init_gate(self):
        init_gatedict = dict()
        for gate_index in gate_group_index[self.gate_class]:
            init_gatedict[gates_pos[gate_index]] = True
        for gate in gates_pos:
            if not init_gatedict.get(gate, False):
                init_gatedict[gate] = False
        return init_gatedict

    def to_descr(self):
        if self.Model is None or len(self.Model) == 0:
            return [0]*12
        gate_list = [(k, v) for k, v in self.Model.items()]
        gate_list.sort(key=lambda ele: ele[0])
        descr = [v for (k, v) in gate_list]
        return descr
    
    def model_gen(self):
        if self.mode != 'train':
            return test_model_gen(self.gate_class)
        elif self.easy_env == 'one_env':
            return easy_model_gen(self.gate_class)
        elif self.easy_env == 'cross_type':
            return easy_model_gen_type(self.gate_class)
        elif self.easy_env == 'no_pair':
            return easy_model_gen_group(self.gate_class)
        else:
            return train_model_gen(self.gate_class)
    
    def _one_random_pos(self, pos_n):
        room = -1
        rooms = list(range(9))
        rooms.remove(4)
        for i in rooms:
            if pos_n in rooms_pos[i]:
                room = i
                break
        if room == -1:
            raise Exception(f'Position {pos_n} cannot be initial position or goal.')
        if self.diag:
            if room not in [0, 2, 6, 8]:
                raise Exception(f'Position {pos_n} cannot be initial position or goal in diagonal-init env.')
            _room = 8 - room
            return np.random.choice(rooms_pos[_room])
        rooms.remove(room)
        _room = np.random.choice(rooms)
        return np.random.choice(rooms_pos[_room])
        
    def _init_pos_goal(self):
        if self.init_random and self.goal_random:
            if self.diag:
                init_rooms = [0, 2, 6, 8]
            else:
                init_rooms = [0, 1, 2, 3, 5, 6, 7]
            init_room = np.random.choice(init_rooms)
            init_pos_n = np.random.choice(rooms_pos[init_room])
            return init_pos_n, self._one_random_pos(init_pos_n)
        elif self.init_random:
            return self._one_random_pos(self.goal), self.goal
        elif self.goal_random:
            return self.init_pos, self._one_random_pos(self.init_pos)
        else:
            return self.init_pos, self.goal
        
    def reset(self):
        self.open = True

        if self.model_random:
            self.Model = self.model_gen()
        init_pos_n, goal_n = self._init_pos_goal()

        self.state = NineroomsGateV2State(init_pos_n, 0, goal_n, False, self.num_pos, deepcopy(self.init_gatedict), [],  self.to_descr())
        return self.state.to_obs()

    def step(self, action):
        if self.state.done:
            raise Exception("Environment should be reseted")

        info = {}
        self.state.current_steps += 1

        next_pos = self.basic_step(self.state.position_n, self.directions[action])
        if next_pos in gates_pos and self.state.gatedict[next_pos]:
            if self.Model[next_pos] == 'coin':
                reward = 1
                self.state.gatedict[next_pos] = False
            elif self.Model[next_pos] == 'water':
                reward = -1
            else:
                next_pos = self.state.position_n
                reward = -0.1
        elif next_pos == self.state.goal_n:
            reward = 10
        else:
            reward = -0.1

        if next_pos == self.state.goal_n or self.state.current_steps >= self.max_epilen:
            done = True
        else:
            done = False

        self.state.position_n = next_pos
        self.state.cum_reward.append(reward)
        self.state.done = done
        if done:
            info = {'episode': {'r': sum(self.state.cum_reward), 'l': self.state.current_steps}}

        return self.state.to_obs(), reward, done, info

    def render(self, mode=0):
        blocks = []
        for gate in gates_pos:
            if self.state.gatedict.get(gate, None):
                x, y = self.tocell[gate]
                blocks.append(self.make_block(x, y, (0, 1, 0)))
        blocks.extend(self.make_basic_blocks())
        arr = self.render_with_blocks(self.origin_background, blocks)
        return arr
    
    def color_render(self):
        blocks = []
        for gate in gates_pos:
            if self.state.gatedict.get(gate, None):
                x, y = self.tocell[gate]
                if self.Model[gate] == 'coin':
                    color = (1, 1, 0)
                elif self.Model[gate] == 'water':
                    color = (0, 1, 0)
                else:
                    color = (1, 0, 1)
                blocks.append(self.make_block(x, y, color))
        blocks.extend(self.make_basic_blocks())
        arr = self.render_with_blocks(self.origin_background, blocks)
        return arr
        
        