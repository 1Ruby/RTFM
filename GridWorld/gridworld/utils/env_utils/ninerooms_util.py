"""
Some utils for ninerooms
"""
import numpy as np

# position_n
LU_pos = [24, 25, 26, 27, 38, 39, 40, 41, 55, 56, 57, 58, 70, 71, 72, 73]
MU_pos = [28, 29, 30, 31, 43, 44, 45, 46, 59, 60, 61, 62, 74, 75, 76, 77]
RU_pos = [32, 33, 34, 35, 47, 48, 49, 50, 64, 65, 66, 67, 78, 79, 80, 81]
LM_pos = [89, 90, 91, 92, 104, 105, 106, 107, 118, 119, 120, 121, 134, 135, 136, 137]
MM_pos = [93, 94, 95, 96, 108, 109, 110, 111, 122, 123, 124, 125, 139, 140, 141, 142]
RM_pos = [97, 98, 99, 100, 112, 113, 114, 115, 127, 128, 129, 130, 143, 144, 145, 146]
LD_pos = [154, 155, 156, 157, 169, 170, 171, 172, 185, 186, 187, 188, 199, 200, 201, 202]
MD_pos = [159, 160, 161, 162, 173, 174, 175, 176, 189, 190, 191, 192, 203, 204, 205, 206]
RD_pos = [163, 164, 165, 166, 177, 178, 179, 180, 193, 194, 195, 196, 208, 209, 210, 211]
rooms_pos = [LU_pos, MU_pos, RU_pos, LM_pos, MM_pos, RM_pos, LD_pos, MD_pos, RD_pos]
gates_pos = [42, 63, 84, 85, 86, 126, 138, 149, 150, 151, 158, 207]
corr_pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 147, 148, 22, 23, 152, 132, 153, 133,
            36, 37, 167, 168, 52, 53, 182, 183, 68, 197, 198, 69, 82, 83, 212, 213, 87, 88, 217, 218, 219, 220, 221,
            222, 223, 224, 225, 226, 227, 228, 101, 230, 231, 232, 233, 234, 235, 229, 102, 116, 117]

gate_groups_all = [
    [
        [0, 4, 5, 11],
        [1, 2, 8, 10],
        [3, 6, 7, 9]
    ],
    [
        [0, 5, 7, 9],
        [1, 3, 6, 11],
        [2, 4, 8, 10]
    ],
    [
        [0, 3, 8, 10],
        [1, 6, 7, 11],
        [2, 4, 5, 9]
    ]
]

easy_group = [
    [0, 3, 7, 9, 10],
    [1, 2, 4, 6, 11],
    [5, 8]
]

gate8_index = [0, 1, 3, 5, 6, 8, 10, 11]
gate6_index = [0, 1, 5, 6, 10, 11]
gate4_index = [3, 5, 6, 8]

gate_group_index = {
    'all': list(range(12)),
    '8gates': gate8_index,
    '6gates': gate6_index,
    '4gates': gate4_index,
    'no_gate': [],
}

gate_groups_8 = [
    [
        [0, 3, 5],
        [6, 8, 11],
        [1, 10]
    ],
    [
        [1, 3, 6],
        [5, 8, 10],
        [0, 11]
    ]
]

gate_groups_6 = [
    [[0, 5], [6, 11], [1, 10]],
    [[1, 6], [5, 10], [0, 11]]
]

gate_group_4 = [
    [[3, 5], [6,], [8,]],
    [[3,], [5,], [6, 8]]
]

gate_groups = {
    'all': gate_groups_all,
    '8gates': gate_groups_8,
    '6gates': gate_groups_6,
    '4gates': gate_group_4,
    'no_gate': [[]],
}

def gen_model(group, group_types):
    Model = dict()
    for gates, gate_type in zip(group, group_types):
        for gate_index in gates:
            Model[gates_pos[gate_index]] = gate_type
    for gate in gates_pos:
        if not Model.get(gate, False):
            Model[gate] = 'coin'
    return Model

def test_model_gen(gate_class='all'):
    Model = dict()
    for gate_index in gate_group_index[gate_class]:
        Model[gates_pos[gate_index]] = np.random.choice(['coin', 'water', 'wall'])
    for gate in gates_pos:
        if not Model.get(gate, False):
            Model[gate] = 'coin'
    return Model

def train_model_gen(gate_class='all'):
    group_types = np.random.permutation(['coin', 'water', 'wall'])
    group = gate_groups[gate_class][np.random.randint(len(gate_groups[gate_class]))]
    return gen_model(group, group_types)

def easy_model_gen(gate_class='all'):
    group_types = ['coin', 'water', 'wall']
    if gate_class == 'all':
        group = easy_group
    else:
        group = gate_groups[gate_class][0]
    return gen_model(group, group_types)

def easy_model_gen_type(gate_class='all'):
    group_types = np.roll(['coin', 'water', 'wall'], np.random.randint(3))
    group = gate_groups[gate_class][0]
    return gen_model(group, group_types)

def easy_model_gen_group(gate_class='all'):
    group_types = ['coin', 'water', 'wall']
    if gate_class == 'all':
        i = np.random.randint(3)
        group = np.roll(gate_groups_all[i], i)
    else:
        group = gate_groups[gate_class][np.random.randint(len(gate_groups[gate_class]))]
    return gen_model(group, group_types)

def model2index(Model: dict):
    gate_list = [(k, v) for k, v in Model.items()]
    gate_list.sort(key=lambda ele: ele[0])
    type_list = [0 if v=='coin' else 1 if v=='water' else 2 for (_, v) in gate_list]
    index = 0
    for order, _type in enumerate(type_list):
        index += _type * (3 ** order)
    return index

def index2model(index):
    Model = dict()
    for gate in gates_pos:
        _type = index % 3
        Model[gate] = 'coin' if _type == 0 else 'water' if _type == 1 else 'wall'
        index //= 3
    return Model

train_model_index = {
    'all': [105670, 109928, 110874, 151292, 153514, 153870, 220098, 222134, 222724, 308716, 309306, 311342, 377570,
            377926, 380148, 420566, 421512, 425770],
    '8gates': [59594, 67371, 118375, 132465, 178666, 184979, 302541, 308854, 355055, 369145, 420149, 427926],
    '6gates': [59540, 60756, 118348, 119316, 178364, 178612, 295732, 295980, 355028, 355996, 413588, 414804],
    '4gates': [297, 513, 1269, 1728, 7101, 7344, 7776, 8019, 13392, 13851, 14607, 14823],
    'no_gate': [0,]
}

easy_model_index = {
    'all': [191577,], '8gates': [302541,], '6gates': [295980,], '4gates': [13851,], 'no_gate': [0,]
}

easy_type_index = {
    'all': [110874, 308716, 377570],
    '8gates': [59594, 302541, 369145],
    '6gates': [59540, 295980, 355996],
    '4gates': [1728, 7101, 13851],
    'no_gate': [0,],
}

easy_group_index = {
    'all': [102364, 152638, 308716],
    '8gates': [132465, 369145],
    '6gates': [119316, 355996],
    '4gates': [513, 1728],
    'no_gates': [0,]
}

train_index_group = {
    'no_easy': train_model_index,
    'one_env': easy_model_index,
    'cross_type': easy_type_index,
    'no_pair': easy_group_index
}
    
