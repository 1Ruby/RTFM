"""
Ninerooms Game with gates
One gate between two adjacent rooms.
Gates may be a coin(rew+), a wall, or a water(rew-).

The transition is a combination of 'action' and 'extra step', as in FourroomsWater.
3 types of gates, each acts as a coin, wall, or water in a game.
A description is allocated to a game, telling the transition model and the role of each type of gates.

Super parameters include the model(transition, gate types), init_pos, goal.
Agents can observerse a description and the image.
Model: {
    'action': ..., 'extra step': ...,
    'gate A': one of ['coin', 'wall', 'water'], 'gate B': ..., 'gate C': ,,,
}

reward:
    penalty: -0.1
    coin gate: 10 (would be eaten)
    water gate: -1 (-0.1 if stay at water, and -1 if hit next time)
    goal: 10

new settings:
    model: {'train', 'test"} use train model or test model, models in file 'train_model_nine' and 'test_model_nine'
    easy_env: whether to fix gates (easy_gate)
    fix_pos: whether to fix initial position and goal (initial in the left-up room, goal in the right-low room)
"""

from .ninerooms import *
from ..utils.env_utils.ninerooms_util import *
import numpy as np
from ..utils.test_util import *
import pickle
import bz2
import os

dirpath = os.path.dirname(__file__)
train_file_path = os.path.join(dirpath, '../utils/env_utils/train_model_nine')
test_file_path = os.path.join(dirpath, '../utils/env_utils/test_model_nine')
train_file = bz2.BZ2File(train_file_path)
test_file = bz2.BZ2File(test_file_path)
train_list = pickle.load(train_file)
test_list = pickle.load(test_file)

easy_gate = {42: 'gate A', 63: 'gate B', 84: 'gate B', 85: 'gate A', 86: 'gate B', 126: 'gate C', 138: 'gate B',
             149: 'gate A', 150: 'gate C', 151: 'gate A', 158: 'gate A', 207: 'gate B'}
fix_init = 57
fix_goal = 195

class NineroomsGateState(NineroomsBaseState):
    def __init__(self, position_n, current_steps, goal_n, done, num_pos, gatedict, cum_reward: list, descr=None):
        super().__init__(position_n, current_steps, goal_n, done, num_pos)
        self.gatedict = gatedict  # dict(pos_n: gate_type), gate_type is one of 'gate A', 'gate B', 'gate C', 'empty'
        self.descr = descr
        self.cum_reward = cum_reward  # list of reward

    def to_obs(self):
        gate_obs = 0
        for i in range(len(gates_pos)):
            gatetype = self.gatedict[gates_pos[i]]
            if gatetype == 'gate A':
                gateindex = 1
            elif gatetype == 'gate B':
                gateindex = 2
            elif gatetype == 'gate C':
                gateindex = 3
            else:
                gateindex = 0
            gate_obs += gateindex * (4 ** i)
        return gate_obs * self.num_pos + self.position_n

class NineroomsGate(NineroomsBase):
    def __init__(self, Model=None, max_epilen=100, init_pos=None, goal=None, gatedict=None, seed=0, mode='train',
                 easy_dy=True, easy_env=True, fix_pos=True):
        super().__init__(max_epilen, init_pos, goal, seed)
        self.Model = Model or dict()
        self.init_pos = init_pos
        self.goal = goal
        self.gatedict = gatedict or dict()
        self.state = NineroomsGateState(0, 0, 0, False, 0, None, [])
        self.open = False
        self.mode = mode
        self.easy_dy = easy_dy
        self.easy_env = easy_env
        self.fix_pos = fix_pos

        if Model is None:
            self.model_random = True
        else:
            self.model_random = False

        if init_pos is None:
            self.init_random = True
        else:
            self.init_random = False

        if goal is None:
            self.goal_random = True
        else:
            self.goal_random = False

        if gatedict is None:
            self.gate_random = True
        else:
            self.gate_random = False

    def to_descr(self):
        descr = ''
        for k in ['action', 'extra step']:
            descr += (k + ': ' + self.Model[k] + ', ')
        descr += ('yello' + ': ' + self.Model['gate A'] + ', ')
        descr += ('green' + ': ' + self.Model['gate B'] + ', ')
        descr += ('purple' + ': ' + self.Model['gate C'] + '.')
        return descr

    def rand_model(self):
        if self.mode == 'train':
            self.Model = np.random.choice(train_list)
        else:
            self.Model = np.random.choice(test_list)
        if self.easy_dy:
            self.Model['action'] = 'normal'
            self.Model['extra step'] = 'stay'
        else:
            self.Model['action'] = np.random.choice(['normal', 'left', 'right', 'inverse'])
            self.Model['extra step'] = np.random.choice(['stay', 'left', 'right', 'forward'])

    def reset(self):
        # If init_random and goal_random, init pos and goal is in diff & random outer rooms.
        # If gate_random, no empty gate will assigned.
        self.open = True

        if self.easy_env:
            self.gatedict = deepcopy(easy_gate)
        elif self.gate_random:
            for pos_n in gates_pos:
                self.gatedict[pos_n] = np.random.choice(['gate A', 'gate B', 'gate C'])

        if self.model_random:
            self.rand_model()

        init_rooms = [0, 1, 2, 3, 5, 6, 7, 8]
        if self.fix_pos:
            init_pos_n = fix_init
        elif self.init_random:
            init_room = np.random.choice(init_rooms)
            init_rooms.remove(init_room)
            init_pos_n = np.random.choice(rooms_pos[init_room])
        else:
            init_pos_n = self.init_pos

        if self.fix_pos:
            goal_pos_n = fix_goal
        elif self.goal_random:
            goal_room = np.random.choice(init_rooms)
            goal_pos_n = np.random.choice(rooms_pos[goal_room])
        else:
            goal_pos_n = self.goal

        self.state = NineroomsGateState(position_n=init_pos_n, current_steps=0, goal_n=goal_pos_n, done=False,
                                        num_pos=self.num_pos, gatedict=self.gatedict, cum_reward=[],
                                        descr=self.to_descr())
        return self.state.to_obs()

    def basic_step(self, pos_n, transfer):
        nextcell = tuple(self.tocell[pos_n] + transfer)
        if not self.occupancy[nextcell]:
            return self.tostate[nextcell]
        return pos_n

    @staticmethod
    def turn(transfer, direction):
        # Help the agent to turn direction
        if direction == 'right':
            if transfer[0]:
                transfer *= -1
            transfer[[0, 1]] = transfer[[1, 0]]
        elif direction == 'left':
            if transfer[1]:
                transfer *= -1
            transfer[[0, 1]] = transfer[[1, 0]]
        elif direction == 'inverse':
            transfer *= -1
        elif direction == 'stay':
            transfer = np.array((0, 0))
        elif direction == 'normal' or direction == 'forward':
            pass
        return transfer

    @staticmethod
    def penalty(pos_n):
        if pos_n in corr_pos:
            return -0.1
        else:
            return -0.1

    def step(self, action):
        if self.state.done:
            raise Exception("Environment should be reseted")
        info = {}
        self.state.current_steps += 1
        if self.state.current_steps >= self.max_epilen:
            done = True
        else:
            done = False
        transfer = self.turn(deepcopy(self.directions[action]), self.Model['action'])
        nextpos = self.basic_step(self.state.position_n, transfer)
        get_penalty = True  # if get a coin or water, not do penalty
        finish = False  # after finish, update state.cum_reward, state.done and info
        reward = 0

        if nextpos == self.state.position_n or (nextpos in gates_pos and self.Model.get(self.gatedict[nextpos], 'empty') == 'wall'):
            # hit a wall or a wall-gate
            reward += self.penalty(self.state.position_n)
            finish = True
        elif nextpos in gates_pos and self.Model.get(self.gatedict[nextpos], 'empty') == 'coin':
            # hit a coin-gate
            self.gatedict[nextpos] = 'empty'
            self.state.gatedict[nextpos] = 'empty'
            reward += 1
            get_penalty = False
        elif nextpos in gates_pos and self.Model.get(self.gatedict[nextpos], 'empty') == 'water':
            # hit a water
            reward -= 1
            get_penalty = False
        # else get an empty gird or the goal gird, not viewed reach the goal

        if not finish and self.Model['extra step'] == 'stay':
            if nextpos == self.state.goal_n:
                done = True
                reward += 10
            elif get_penalty:
                reward += self.penalty(nextpos)
            self.state.position_n = nextpos
            finish = True

        if not finish:
            extra_transfer = self.turn(transfer, self.Model['extra step'])
            extrapos = self.basic_step(nextpos, extra_transfer)
            if extrapos == nextpos or (extrapos in gates_pos and self.Model.get(self.gatedict[extrapos], 'empty') == 'wall'):
                if nextpos == self.state.goal_n:
                    done = True
                    reward += 10
                elif get_penalty:
                    reward += self.penalty(nextpos)
                self.state.position_n = nextpos
                finish = True
            elif extrapos in gates_pos and self.Model.get(self.gatedict[extrapos], 'empty') == 'coin':
                self.gatedict[extrapos] = 'empty'
                self.state.gatedict[extrapos] = 'empty'
                reward += 1
                get_penalty = False
            elif extrapos in gates_pos and self.Model.get(self.gatedict[extrapos], 'empty') == 'water':
                reward -= 1
                get_penalty = False
            elif extrapos == self.state.goal_n:
                reward += 10
                get_penalty = False
                done = True

            if not finish:
                self.state.position_n = extrapos
                if get_penalty:
                    reward += self.penalty(extrapos)

        # finished
        self.state.cum_reward.append(reward)
        self.state.done = done
        if done:
            info = {'episode': {'r': sum(self.state.cum_reward), 'l': self.state.current_steps}}

        return self.state.to_obs(), reward, done, info

    def render(self, mode=0):
        blocks = []
        for pos_n, gatetype in self.gatedict.items():
            if pos_n == self.state.position_n:
                continue
            x, y = self.tocell[pos_n]
            if gatetype == 'empty':
                continue
            elif gatetype == 'gate A':
                blocks.append(self.make_block(x, y, (1, 1, 0)))
            elif gatetype == 'gate B':
                blocks.append(self.make_block(x, y, (0, 1, 0)))
            elif gatetype == 'gate C':
                blocks.append(self.make_block(x, y, (1, 0, 1)))
        blocks.extend(self.make_basic_blocks())

        arr = self.render_with_blocks(self.origin_background, blocks)
        return arr

if __name__ == '__main__':
    env = ImageInputWrapper(NineroomsGate())
    check_render(env)
    check_run(env)
    print('check finished.')

