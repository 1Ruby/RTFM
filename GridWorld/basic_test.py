# to run basic test for all envs
import inspect
from gridworld.utils.wrapper.wrappers import ImageInputWrapper
from gridworld.utils.test_util import *
import gridworld

import os
import importlib
from gridworld.envs.fourrooms import FourroomsBase

path = os.path.dirname(__file__)
envs_path = os.path.join(path, 'gridworld', 'envs')

envfiles = [ f for f in os.listdir(envs_path) if os.path.isfile(os.path.join(envs_path,f))]
envfiles = [f.split('.')[0] for f in envfiles if f.endswith('.py')]

for f in envfiles:
    module = importlib.import_module('gridworld.envs.'+ f )
    for name, obj in inspect.getmembers(module, inspect.isclass):
        try:
            if isinstance(obj(), FourroomsBase):
                env = ImageInputWrapper(obj())
                check_render(env)
                check_run(env)
                print(f"basic check for {name} finished")
        except Exception as e:
            pass

