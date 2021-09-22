"""
Some utils for fourrooms
"""
import itertools
import numpy as np
from copy import deepcopy

LU_pos = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45]
RU_pos = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 26, 27, 28, 29, 30, 36, 37, 38, 39, 40, 46, 47, 48, 49, 50, 52, 53, 54, 55,
          56]
LD_pos = [57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 83, 84, 85, 86, 87, 94, 95, 96, 97, 98]
RD_pos = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]
rooms_pos = [LU_pos, RU_pos, LD_pos, RD_pos]
gates_pos = [25, 51, 62, 88]

train_models_value = [
    # 0: coin, 1: water, 2: wall
    (2, 0, 1, 1),
    (1, 1, 2, 0),
    (0, 1, 0, 2),
    (1, 2, 0, 0),
    (0, 0, 1, 1),
    (1, 0, 1, 0),
    (0, 1, 0, 0),
    (1, 1, 0, 1)
]

test_models_value = []
gate_value = (0, 1, 2)
for model_value in itertools.product(gate_value, gate_value, gate_value, gate_value):
    if model_value.count(2) > 1 or model_value in train_models_value:
        continue
    test_models_value.append(model_value)


def value2model(model_value):
    Model = dict()
    for i in range(4):
        if model_value[i] == 0:
            Model[gates_pos[i]] = 'coin'
        elif model_value[i] == 1:
            Model[gates_pos[i]] = 'water'
        else:
            Model[gates_pos[i]] = 'wall'
    return Model

train_models = [value2model(model_value) for model_value in train_models_value]
test_models = [value2model(model_value) for model_value in test_models_value]

gates_of_room = {
    0: [25, 51],
    1: [25, 62],
    2: [51, 88],
    3: [62, 88]
}

rooms_of_gates = {
    25: [0, 1],
    51: [0, 2],
    62: [1, 3],
    88: [2, 3]
}

def best_return(Model, init_pos, goal):
    init_room = np.where([init_pos in room for room in rooms_pos])[0][0]
    goal_room = np.where([goal in room for room in rooms_pos])[0][0]
    gate_list = [Model[k] for k in gates_pos]
    
    if init_room == goal_room:
        init_coin_gates = []
        for gate in gates_of_room[init_room]:
            if Model[gate] == 'coin':
                init_coin_gates.append(gate)
        if len(init_coin_gates) == 0:
            num_coins = 0
        elif len(init_coin_gates) == 1:
            adj_room = (set(rooms_of_gates[init_coin_gates[0]]) - {init_room}).pop()
            adj_gate = (set(gates_of_room[adj_room]) - {init_coin_gates[0]}).pop()
            if Model[adj_gate] == 'coin':
                num_coins = gate_list.count('coin')
            else:
                num_coins = 1
        num_waters = 0
    elif init_room + goal_room == 3:  # diag case
        num_coins = gate_list.count('coin')
        leftrooms = list(set(range(4)) - {init_room, goal_room})
        gates_on_way = [[Model[k] for k in gates_of_room[leftrooms[i]]] for i in range(2)]
        num_waters_list = []
        for i in range(2):
            if 'wall' in gates_on_way[i]:
                num_waters_list.append(2)
            else:
                num_waters_list.append(gates_on_way[i].count('water'))
        num_waters = min(num_waters_list)
    else:  # adjacent case
        init_gates_pos = set(gates_of_room[init_room])
        goal_gates_pos = set(gates_of_room[goal_room])
        common_gate = Model[(init_gates_pos & goal_gates_pos).pop()]
        nearby_gates = [Model[k] for k in iter(init_gates_pos ^ goal_gates_pos)]
        if not 'coin' in nearby_gates and common_gate != 'wall':  # wont reach the far gate
            num_coins = 1 if common_gate == 'coin' else 0
        else:
            num_coins = gate_list.count('coin')
        _gate_list = deepcopy(gate_list)
        _gate_list.remove(common_gate)
        gates_on_way = [[common_gate,], _gate_list]
        num_water_list = []
        for i in range(2):
            if 'wall' in gates_on_way[i]:
                num_water_list.append(3)
            else:
                num_water_list.append(gates_on_way[i].count('water'))
        num_waters = min(num_water_list)
    
    return 10 * (1 + num_coins - num_waters)
        