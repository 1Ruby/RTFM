"""
Basic Ninerooms Game
3 * 3 rooms.
A corridor around the nine rooms, connected to the outer 8 rooms.
"""


from .fourrooms import *
import numpy as np

class NineroomsBaseState(FourroomsBaseState):
    def __init__(self, position_n, current_steps, goal_n, done, num_pos):
        super().__init__(position_n, current_steps, goal_n, done, num_pos)

    def to_obs(self):
        return np.array(self.position_n)

    def to_tuple(self):
        return self.position_n, self.current_steps, self.goal_n, self.done

class NineroomsBase(FourroomsBase):
    def __init__(self, max_epilen=100, init_pos=None, goal=None, seed=0):
        super().__init__(max_epilen=max_epilen, goal=goal, seed=seed)
        self.init_pos = init_pos

    def init_layout(self):
        layout = """\
11111111111111111111
1                  1
1 111 1111 111 111 1
1 1    1    1    1 1
1 1         1      1
1      1         1 1
1 1    1    1    1 1
1 1 111111 11111 1 1
1 1    1    1    1 1
1      1    1    1 1
1 1    1           1
1 1         1    1 1
1 111 111 1111 111 1
1 1         1    1 1
1 1    1    1      1
1      1    1    1 1
1 1    1         1 1
1 11 111111 111 11 1
1                  1
11111111111111111111
"""
        return layout

    def reset(self):
        super().reset()
        self.state.position_n = self.init_pos or self.state.position_n
        return self.state.to_obs

if __name__ == '__main__':
    # basic test
    env_origin = ImageInputWrapper(NineroomsBase(seed=None))
    check_render(env_origin)
    check_run(env_origin)