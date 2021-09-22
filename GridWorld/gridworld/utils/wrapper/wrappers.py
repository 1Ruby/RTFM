import gym
from gym import core, spaces
import numpy as np
from copy import deepcopy
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
class ImageInputWrapper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        screen_height = self.env.obs_height
        screen_width = self.env.obs_width
        self.observation_space = spaces.Box(low=0, high=255, shape=\
        (screen_height, screen_width,3), dtype=np.uint8)
        self.mean_obs = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        obs = self.env.render()
        info['s_tp1'] = state
        return obs.astype(np.uint8), reward, done, info

    def reset(self):
        self.env.reset()
        obs = self.env.render()
        return obs.astype(np.uint8)

class GateDescWrapper:
    # wrap up image imput envs
    # descr is converted to onehot
    def __init__(self, env):
        self.env = env
        self.observation_space = spaces.Dict({
            'image': self.env.observation_space,
            'desc': spaces.Box(low=0, high=1, shape=(12, 3), dtype=np.uint8)
        })

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observation(self, obs):
        # support some wrappers
        return obs

    @staticmethod
    def desc2onehot(desc):
        # desc: 1*N list
        onehot_desc = []
        for gate in desc:
            if gate == 'coin':
                onehot = [1, 0, 0]
            elif gate == 'water':
                onehot = [0, 1, 0]
            elif gate == 'wall':
                onehot = [0, 0, 1]
            else:
                raise Exception(f"Gate type {gate} is not defined in the used wrapper.")
            onehot_desc.append(onehot)
        return onehot_desc

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if hasattr(self.env.state, 'description'):
            descr = self.env.state.description
        elif hasattr(self.env.state, 'descr'):
            descr = self.env.state.descr
        else:
            descr = None
        desc_obs = {'image':obs, 'desc':self.desc2onehot(descr)}
        return self.observation(desc_obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        if hasattr(self.env.state, 'description'):
            descr = self.env.state.description
        elif hasattr(self.env.state, 'descr'):
            descr = self.env.state.descr
        else:
            descr = None
        desc_obs = {'image':obs, 'desc':self.desc2onehot(descr)}
        return self.observation(desc_obs)


class WhiteBackgroundWrapper:
    # wrap up gridworld envs
    # render return square img w. white background, dynamic block_size 1/2/4/8 if not fixed
    def __init__(self, env, size=64, block_size=None):
        self.render_size = size
        if env.Row > self.render_size or env.Col > self.render_size:
            raise Exception(f'The layout is {env.Row}*{env.Col}, which cannot be converted to {size}*{size}')
        if block_size is not None and (block_size * env.Row > size or block_size * env.Col > size):
            raise Exception('Block render_size is too large.')
        self.env = env
        self.block_size = block_size
        
        if block_size is None:
            bs_tmp = 8
            while bs_tmp * env.Row > size or bs_tmp * env.Col > size:
                bs_tmp /= 2
            self.block_size = int(bs_tmp)
        
        self.env.block_size = self.block_size
        self.env.obs_height = self.block_size * self.env.Row
        self.env.obs_width = self.block_size * self.env.Col
        self.env.origin_background = self.env.init_background()
        self.obs_height = size
        self.obs_width = size
        self.observation_space = spaces.Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8)
        self.background = 255 * np.ones((size, size, 3), dtype=np.int)
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def render(self, mode=0):
        obs = deepcopy(self.background)
        arr = self.env.render()
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs