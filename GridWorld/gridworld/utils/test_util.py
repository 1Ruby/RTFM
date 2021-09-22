'''
Test scripts:
- check_render(env):check rendering
- check_run(env):random run
'''
import cv2
import os
import numpy as np
def check_render(env):
    #check if it renders correctly
    path = os.path.dirname(__file__)
    render_path = os.path.join(path, 'test_render')
    if not os.path.exists(render_path):
        os.mkdir(render_path)

    env.reset()
    cv2.imwrite(os.path.join(render_path,'test0.jpg'),np.flip(env.render(), -1))
    env.step(0)
    cv2.imwrite(os.path.join(render_path,'test1.jpg'),np.flip(env.render(), -1))
    env.step(1)
    cv2.imwrite(os.path.join(render_path,'test2.jpg'),np.flip(env.render(), -1))
    env.step(2)
    cv2.imwrite(os.path.join(render_path,'test3.jpg'),np.flip(env.render(), -1))
    env.step(3)
    cv2.imwrite(os.path.join(render_path,'test4.jpg'),np.flip(env.render(), -1))

def check_run(env):
    #Run a few episodes
    reward_list=[]
    for i in range(1000):
        obs,reward,done,_=env.step(env.action_space.sample())
        reward_list.append(reward)
        if done:
            env.reset()
            reward_list=[]

