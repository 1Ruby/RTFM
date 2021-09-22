from .fourrooms import FourroomsBase
from .fourrooms_coin import FourroomsCoin, FourroomsCoinBackgroundNoise, FourroomsCoinRandomNoise,\
    FourroomsCoinRandomNoiseV2, FourroomsCoinWhiteBackground, FourroomsKidNoise
from .fourrooms_water import FourroomsWater
from .fourrooms_gate import FourroomsGate
from .ninerooms import NineroomsBase
from .ninerooms_gate import NineroomsGate
from .ninerooms_gate_v2 import NineroomsGateV2
from gym.envs.registration import register

register(
    id='fourrooms_base-v0',
    entry_point='gridworld.envs:FourroomsBase'
)

register(
    id='fourrooms_coin-v0',
    entry_point='girdworld.envs:FourroomsCoin'
)

register(
    id='fourrooms_water-v0',
    entry_point='girdworld.envs:FourroomsWater'
)

register(
    id='fourrooms_gate-v0',
    entry_point='gridworld.envs:FourroomsGate'
)

register(
    id='ninerooms_base-v0',
    entry_point='gridworld.envs:NineroomsBase'
)

register(
    id='ninerooms_gate-v1',
    entry_point='gridworld.envs:NineroomsGate'
)

register(
    id='ninerooms_gate-v2',
    entry_point='gridworld.envs:NineroomsGateV2'
)
