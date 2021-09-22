"""
Fourrooms Game with water

class:
    + FourroomsWaterState
    + FourroomsWater
    + FourroomsWaterNorender

Properties:
Random waters garantee the rest space to be connected.
Resetting resets start position, goal position, coins and waters.
Each game has fixed number of coins and waters.

Update:
Now the agent walks two steps within a time step and the model can be random.
    A model is a dict containing:
        'water': one of 'pass', 'block', 'left', 'right', 'forward'
        'coin': one of 'pass', 'left', 'right', 'forward'
        'action': one of 'normal', 'left', 'right', 'inverse'
        'extra step': one of 'stay', 'left', 'right', 'forward'
    Basic: {'water': 'block', 'coin': 'pass', 'action': normal', 'extra step': 'stay'}
    One model itself is deterministic.
    A description is seen to the agent about the model.
Method 'play' is to play the game by hand.

Update:
New settings:
    mode: {'train', 'test'}, use train model or test model, models in file 'train_model' and 'test_model'
    easy_env: whether to use easy env, 3-5 coins and waters in train mode, 6-8 coins and waters in test mode
    fix_pos: whether to fix initial position and goal

Possible extensions:
Each game has random number of coins and waters.
Length-variable discription
Variable action space
"""

from .fourrooms_coin import *
from ..utils.wrapper.wrappers import ImageInputWrapper
from copy import deepcopy
from ..utils.test_util import *
import os
import pickle
import bz2

dirpath = os.path.dirname(__file__)
train_file_path = os.path.join(dirpath, '../utils/env_utils/train_model')
test_file_path = os.path.join(dirpath, '../utils/env_utils/test_model')
train_file = bz2.BZ2File(train_file_path, 'r')
test_file = bz2.BZ2File(test_file_path, 'r')
train_list = pickle.load(train_file)
test_list = pickle.load(test_file)
fix_init = 11
fix_goal = 92


class FourroomsWaterState(FourroomsCoinState):
    def __init__(self, position_n: int, current_steps: int, goal_n: int, done: bool, num_pos: int, coin_dict: dict,
                 num_coins, water_list: list, num_waters, cum_reward: list, description):
        self.position_n = position_n
        self.current_steps = current_steps
        self.goal_n = goal_n
        self.done = done
        self.num_pos = num_pos
        self.coin_dict = coin_dict
        self.num_coins = num_coins
        self.water_list = water_list
        self.num_waters = num_waters
        self.cum_reward = cum_reward
        self.description = description

    def __init__(self, base: FourroomsCoinState, water_list, num_waters, description):
        self.position_n = base.position_n
        self.current_steps = base.current_steps
        self.goal_n = base.goal_n
        self.done = base.done
        self.num_pos = base.num_pos
        self.coin_dict = base.coin_dict
        self.num_coins = base.num_coins
        self.water_list = water_list
        self.num_waters = num_waters
        self.cum_reward = base.cum_reward
        self.description = description

    def watered_state(self):
        num_coins = len(self.coin_dict)
        value_list = [(v[0] if v[1] else 0) for v in self.coin_dict.values()]
        multiplier = np.dot(value_list, [2 ** i for i in range(num_coins)])
        return multiplier * self.num_pos + self.position_n

    def to_obs(self) -> np.array:
        return np.array(self.watered_state())


class FourroomsWater(FourroomsCoin):
    def __init__(self, Model=None, max_epilen=100, goal=None, num_coins=3, num_waters=3, seed=0, mode='train',
                 easy_env=True, fix_pos=True):
        super(FourroomsCoin, self).__init__(max_epilen, goal, seed)
        self.num_waters = num_waters
        assert self.num_pos > (self.num_waters + 10), "too many waters."
        self.num_coins = num_coins
        assert (self.num_pos - self.num_waters) > (self.num_coins + 5), "too many coins."
        self.init_states = list(range(self.observation_space.n))
        self.init_states_ori = deepcopy(self.init_states)
        self.observation_space = spaces.Discrete((self.num_pos - self.num_waters) * (2 ** self.num_coins))
        self.occupancy_ori = deepcopy(self.occupancy)
        if Model is None:
            self.model_random = 1
        else:
            self.model_random = 0
            self.Model = Model
        self.mode = mode  # train or test
        self.easy_env = easy_env  # if easy_env, #coins in train set is 3-5, in test set is 6-8
        self.fix_pos = fix_pos  # if fix_init, the start position and the goal is fixed
        self.reset()

    def basic_step(self, cell, action):
        nextcell = tuple(cell + self.directions[action])
        if not self.occupancy[nextcell]:
            return nextcell
        return cell

    def not_block(self, water):
        # Check whether random water will block the rest space.
        water_cell = self.tocell[water]
        self.occupancy[self.tocell[water]] = 1
        around = self.empty_around(water_cell)
        if len(around) == 0:
            return True
        spread = []
        remain = [around[0]]
        while len(around) != 0 and len(remain) != 0:
            currentcell = remain[0]
            remain.remove(currentcell)
            spread.append(currentcell)
            if currentcell in around:
                around.remove(currentcell)
            for action in range(self.action_space.n):
                nextcell = self.basic_step(currentcell, action)
                if nextcell not in spread and nextcell not in remain:
                    remain.append(nextcell)
        if len(around) == 0:
            return True
        self.occupancy[self.tocell[water]] = 0
        return False

    def reset(self):
        # reset water_list, init_states, occupancy
        if self.easy_env:
            if self.mode == 'train':
                self.num_coins = np.random.choice([3, 4, 5])
                self.num_waters = np.random.choice([3, 4, 5])
            else:
                self.num_coins = np.random.choice([6, 7, 8])
                self.num_waters = np.random.choice([6, 7, 8])
        self.observation_space = spaces.Discrete((self.num_pos - self.num_waters) * (2 ** self.num_coins))
        self.init_states = deepcopy(self.init_states_ori)
        self.occupancy = deepcopy(self.occupancy_ori)
        water_list = np.array([], dtype=int)
        state_list = deepcopy(self.init_states)
        for _ in range(self.num_waters):
            while True:
                if len(state_list) == 0:
                    water = None
                    break
                water = np.random.choice(state_list)
                state_list.remove(water)
                if self.not_block(water):
                    break
            if water is not None:
                water_list = np.append(water_list, water)
                self.init_states.remove(water)
                state_list = deepcopy(self.init_states)
            else:
                raise NotImplementedError("Building waters error.")

        # reset goal, position_n, coin_dict, state
        super(FourroomsCoin, self).reset()
        if self.fix_pos:
            self.state.position_n = fix_init
            self.state.goal_n = fix_goal

        self.state = FourroomsCoinState(self.state, {}, self.num_coins, [])
        init_states = deepcopy(self.init_states)
        if self.state.goal_n in init_states:
            init_states.remove(self.state.goal_n)
        if self.state.position_n in init_states:
            init_states.remove(self.state.position_n)
        coin_list = np.random.choice(init_states, self.num_coins, replace=False)
        coin_dict = {coin: (1, True) for coin in coin_list}
        self.state.coin_dict = coin_dict

        # reset Model, description
        if self.model_random:
            if self.mode == 'train':
                self.Model = np.random.choice(train_list)
            else:
                self.Model = np.random.choice(test_list)
        self.state = FourroomsWaterState(self.state, water_list, self.num_waters, None)
        descr = self.todescr(self.Model)
        self.state.descr = descr
        return self.state.to_obs()

    @staticmethod
    def todescr(Model: dict):
        # Turn the Model to discription
        descr = []
        for s, t in Model.items():
            descr.append(s + ': ' + t)
        return tuple(descr)

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

    def model_step(self, cell, transfer, extra=1, push=0, coin_get=None, push_num=0):
        # Help for self.step and debugging, return next_cell, coin_git
        if coin_get is None:
            coin_get = []
        if not push:
            if extra:
                transfer = self.turn(transfer, self.Model['action'])
            else:
                if self.Model['extra step'] == 'stay':
                    return cell, coin_get
                transfer = self.turn(transfer, self.Model['extra step'])
        next_cell = tuple(cell + transfer)
        if self.occupancy[next_cell]:
            if self.tostate.get(next_cell, -1) in self.state.water_list:
                if self.Model['water'] == 'pass':
                    pass
                elif self.Model['water'] == 'block':
                    next_cell = cell
                    return next_cell, coin_get
                else:
                    if push_num < 4:
                        transfer = self.turn(transfer, self.Model['water'])
                        return self.model_step(next_cell, transfer, extra, 1, coin_get, push_num+1)
                    else:
                        pass
            else:
                next_cell = cell
                return next_cell, coin_get
        elif self.state.coin_dict.get(self.tostate[next_cell], (0, False))[1] and\
                self.tostate[next_cell] not in coin_get:
            coin_get.append(self.tostate[next_cell])
            if self.Model['coin'] == 'pass':
                pass
            else:
                if push_num < 4:
                    transfer = self.turn(transfer, self.Model['coin'])
                    return self.model_step(next_cell, transfer, extra, 1, coin_get, push_num+1)
        if extra:
            return self.model_step(next_cell, transfer, 0, 0, coin_get)
        return next_cell, coin_get

    def step(self, action):
        transfer = deepcopy(self.directions[action])
        position_cell = self.tocell[self.state.position_n]
        next_cell, coin_get = self.model_step(position_cell, transfer, coin_get=[])
        reward = 0
        done = 0
        for coin in coin_get:
            self.state.coin_dict[coin] = (self.state.coin_dict[coin][0], False)
            reward += self.state.coin_dict[coin][0] * 10
        next_state = self.tostate[next_cell]
        if self.state.coin_dict.get(next_state, (0, False))[1]:
            reward += self.state.coin_dict.get(next_state, 0)[0] * 10
            self.state.coin_dict[next_state] = (self.state.coin_dict[next_state][0], False)
        elif next_state == self.state.goal_n:
            reward += 10
            done = 1
        if reward == 0:
            reward = -0.1

        self.state.cum_reward.append(reward)
        self.state.position_n = next_state
        self.state.current_steps += 1
        self.state.done = done or (self.state.current_steps >= self.max_epilen)
        info = {}
        if self.state.done:
            info = {'episode': {'r': np.sum(self.state.cum_reward), 'l': self.state.current_steps}}

        return self.state.to_obs(), reward, self.state.done, info

    def render(self, mode=0):
        blocks = []
        return self.render_water_blocks(blocks)

    def render_water_blocks(self, blocks=None):
        if blocks is None:
            blocks = []
        for water in self.state.water_list:
            x, y = self.tocell[water]
            blocks.append(self.make_block(x, y, (0, 1, 0)))
        for coin, count in self.state.coin_dict.items():
            x, y = self.tocell[coin]
            if count[1]:  # exist
                blocks.append(self.make_block(x, y, (1, 1, 0)))
        blocks.extend(self.make_basic_blocks())

        arr = self.render_with_blocks(self.origin_background, blocks)
        return arr


if __name__ == '__main__':
    env = ImageInputWrapper(FourroomsWater())
    check_render(env)
    check_run(env)
    print("Basic check finished.")
