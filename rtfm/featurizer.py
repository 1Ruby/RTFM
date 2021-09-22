# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import revtok
import random
import numpy as np
from pprint import pprint
from rtfm.dynamics import monster as M, item as I, world_object as O, event as E
from GridWorld.gridworld.utils.env_utils.rtfm_util import read_things


class Featurizer:

    def get_observation_space(self, task):
        raise NotImplementedError()

    def featurize(self, task):
        raise NotImplementedError()


class Concat(Featurizer, list):

    def get_observation_space(self, task):
        feat = {}
        for f in self:
            feat.update(f.get_observation_space(task))
        return feat

    def featurize(self, task):
        feat = {}
        for f in self:
            feat.update(f.featurize(task))
        return feat


class ValidMoves(Featurizer):
    #  valid actions: [1, 1, 1, 1] in our gridworld
    def get_observation_space(self, task):
        return {'valid': (task.action_space.n,)}

    def featurize(self, task):
        return {'valid': torch.tensor([1, 1, 1, 1], dtype=torch.float32)}

class Position(Featurizer):
    #  position: [x, y]
    def get_observation_space(self, task):
        return {'position': (2,)}

    def featurize(self, task):
        pos = task.tocell[task.state.position_n]
        return {'position': torch.tensor(pos, dtype=torch.float32)}


class RelativePosition(Featurizer):
    #  relative position to the map, size=(Row, Col, 2)
    #  rel_pos[:, i, 0] = (i - x) / Col
    #  rel_pos[i, :, 1] = (i - y) / Row
    def get_observation_space(self, task):
        return {'rel_pos': (task.Row, task.Col, 2)}

    def featurize(self, task):
        x_offset = torch.Tensor(task.Row, task.Col).zero_()
        y_offset = torch.Tensor(task.Row, task.Col).zero_()
        # x, y = task.tocell[task.state.position_n]
        # for i in range(task.Row):
        #     x_offset[i, :] = i - x
        # for i in range(task.Col):
        #     y_offset[:, i] = i - y
        # We don't offer the position infomation.
        return {'rel_pos': torch.stack([x_offset / task.Col, y_offset / task.Row], dim=2)}


class WikiExtract(Featurizer):

    def get_observation_space(self, task):
        return {
            'wiki_extract': (task.max_wiki, ),
        }

    def featurize(self, task):
        return {'wiki_extract': task.get_wiki_extract()}


class Progress(Featurizer):

    def get_observation_space(self, task):
        return {'progress': (1, )}

    def featurize(self, task):
        return {'progress': torch.tensor([task.iter / task.max_iter], dtype=torch.float)}
    
    
def clear():
    if os.name == 'posix':
        _ = os.system('cls')
    else:
        _ = os.system('clear')


class Terminal(Featurizer):
    #  print terminal info
    def get_observation_space(self, task):
        return {}

    def featurize(self, task):
        clear()
        reward = np.sum(task.state.cum_reward)
        iteration = task.iter
        print('total reward: ' + str(reward) + ', iteration: ' + str(iteration))


class Text(Featurizer):
    def __init__(self, max_cache=1e6):
        super().__init__()
        self._cache = {}
        self.max_cache = max_cache

    def get_observation_space(self, task):
        return {
            'name': (task.Row, task.Col, task.max_placement, task.max_name),
            'name_len': (task.Row, task.Col, task.max_placement),
            'inv': (task.max_inv,),
            'inv_len': (1,),
            'wiki': (task.max_wiki,),
            'wiki_len': (1,),
            'task': (task.max_task,),
            'task_len': (1,),
        }

    def featurize(self, task, eos='pad', pad='pad'):
        # use words to describe the map
        smat = []
        lmat = []
        for x in range(0, task.Row):
            srow = []
            lrow = []
            for y in range(0, task.Col):
                names, lens = task.read_pos(x, y, eos, pad)
                empty_name, empty_length = read_things(task.vocab, 'empty', task.max_name, eos, pad)
                names = names[:task.max_placement]
                lens = lens[:task.max_placement]
                names += [empty_name] * (task.max_placement - len(names))
                lens += [empty_length] * (task.max_placement - len(lens))
                srow.append(names)
                lrow.append(lens)
            smat.append(srow)
            lmat.append(lrow)
        wiki, wiki_len = read_things(task.vocab, task.read_model(), task.max_wiki, eos, pad)
        ins, ins_len = read_things(task.vocab, [], task.max_task, eos, pad)
        inv, inv_len = read_things(task.vocab, [], task.max_inv, eos, pad)
        ret = {
            'name': smat,
            'name_len': lmat,
            'inv': inv,
            'inv_len': [inv_len],
            'wiki': wiki,
            'wiki_len': [wiki_len],
            'task': ins,
            'task_len': [ins_len]
        }
        ret = {k: torch.tensor(v, dtype=torch.long) for k, v in ret.items()}
        return ret
