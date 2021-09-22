from .fourrooms_coin import FourroomsCoinWhiteBackground
import numpy as np
from copy import deepcopy
from ..utils.wrapper.wrappers import ImageInputWrapper
import cv2
from ..utils.test_util import check_render, check_run

class PoisonousCoin(FourroomsCoinWhiteBackground):
    def __init__(self, *args, obs_size=64,**kwargs):
        super(PoisonousCoin, self).__init__( *args, obs_size=obs_size, \
        **kwargs)
        pink = np.array([221,160,221], dtype=int)
        pink = pink.reshape((1,1,3))
        self.background = np.repeat(np.repeat(pink, obs_size, axis=0), obs_size, axis=1)
        

    def step(self, action):
        if self.state.done:
            raise Exception("Environment should be reseted")
        currentcell = self.tocell[self.state.position_n]
        possible_actions = list(range(self.action_space.n))
        try:
            nextcell = tuple(currentcell + self.directions[action])
            possible_actions.remove(action)
        except TypeError:
            nextcell = tuple(currentcell + self.directions[action[0]])
            possible_actions.remove(action[0])

        if not self.occupancy[nextcell]:
            currentcell = nextcell

        state = self.tostate[currentcell]
        self.state.position_n = state
        if state == self.state.goal_n:
            reward = 10
        elif self.state.coin_dict.get(state, (0, False))[1]:  # if find coin
            reward = self.state.coin_dict.get(state, 0)[0] * -10
            self.state.coin_dict[state] = (self.state.coin_dict[state][0], False)
        else:
            reward = -0.1
        self.state.cum_reward.append(reward)

        self.state.current_steps += 1
        self.state.done = (state == self.state.goal_n) or self.state.current_steps >= self.max_epilen

        info = {}
        if self.state.done:
            info = {'episode': {'r': np.sum(self.state.cum_reward), 'l': self.state.current_steps}}
            # print(np.sum(self.state.cum_reward))
            self.state.cum_reward = []

        return self.state.to_obs(), reward, self.state.done, info

    def render_coin_blocks(self, blocks=None):
        """
        render posionous coins to green
        """
        if blocks is None:
            blocks = []
        for coin, count in self.state.coin_dict.items():
            x, y = self.tocell[coin]
            if count[1]:  # exist
                blocks.append(self.make_block(x, y, (0, 1, 0)))
        blocks.extend(self.make_basic_blocks())

        arr = self.render_with_blocks(self.origin_background, blocks)
        return arr



if __name__ == '__main__':
    # basic test)
    env = ImageInputWrapper(PoisonousCoin())
    check_render(env)
    check_run(env)
    print("basic check finished")