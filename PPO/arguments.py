# cuda
import numpy as np
import pandas as pd
CUDA_VISIBLE_DEVICES = 3


############################### atari game #########################
# learning rate
learning_rate = 2.5e-4
# learning_rate = 2.5e-4/2
# learning_rate = 2.5e-4
# learning_rate = 2.5e-4*2
# learning_rate = 2.5e-4*4
# learning_rate = 2.5e-4*8
# learning_rate = 2.5e-4*16

# learning_rate = 2.5e-4*32

reward_scale = 1
record_time = 0
reward_num = pd.DataFrame(columns={'run', 'record_time', '0', '10', 'others'})
# SEED_LIST = [1,2,3]
#
#SEED_LIST = [4, 5, 6]
#SEED_LIST = [7,8,9]
#SEED_LIST = [10,11,12]
# SEED_LIST = [13,14,15]
# SEED_LIST = [16,17,18]
# SEED_LIST = [19,20]
SEED_LIST = [21, 22,23,24]
SEED_LIST = [25,26,27]
SEED_LIST = [28,29,30]

#SEED_LIST = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# SEED_LIST = [1,2,3]
# SEE_LIST
zero_reward_num = 0
ten_reward_num = 0
other_reward = 0

ATARI_NAME = 'AlienNoFrameskip-v4'
# ATARI_NAME = 'BoxingNoFrameskip-v4'
# ATARI_NAME = 'MsPacmanNoFrameskip-v4'
# ATARI_NAME = 'StarGunnerNoFrameskip-v4'

# ATARI_NAME = 'BreakoutNoFrameskip-v4'
# ATARI_NAME = 'QbertNoFrameskip-v4'
# ATARI_NAME = 'ZaxxonNoFrameskip-v4'

# env
# ATARI_NAME = 'BeamRiderNoFrameskip-v4'
# ATARI_NAME = 'BowlingNoFrameskip-v4'
# ATARI_NAME = 'BoxingNoFrameskip-v4'
# ATARI_NAME = 'BreakoutNoFrameskip-v4'
# ATARI_NAME = 'CentipedeNoFrameskip-v4'
# ATARI_NAME = 'ChopperCommandNoFrameskip-v4'
# ATARI_NAME = 'CrazyClimberNoFrameskip-v4'
# ATARI_NAME = 'DemonAttackNoFrameskip-v4'
# ATARI_NAME = 'DoubleDunkNoFrameskip-v4'
# ATARI_NAME = 'EnduroNoFrameskip-v4'
# ATARI_NAME = 'FishingDerbyNoFrameskip-v4'
# ATARI_NAME = 'FreewayNoFrameskip-v4'
# ATARI_NAME = 'FrostbiteNoFrameskip-v4'
# ATARI_NAME = 'GopherNoFrameskip-v4'
# ATARI_NAME = 'GravitarNoFrameskip-v4'
# ATARI_NAME = 'IceHockeyNoFrameskip-v4'
# ATARI_NAME = 'JamesbondNoFrameskip-v4'
# ATARI_NAME = 'KangarooNoFrameskip-v4'
# ATARI_NAME = 'KrullNoFrameskip-v4'
# ATARI_NAME = 'KungFuMasterNoFrameskip-v4'
# ATARI_NAME = 'MontezumaRevengeNoFrameskip-v4'
# ATARI_NAME = 'MsPacmanNoFrameskip-v4'
# ATARI_NAME = 'NameThisGameNoFrameskip-v4'
# ATARI_NAME = 'PitfallNoFrameskip-v4'

# ATARI_NAME = 'PongNoFrameskip-v4'
# ATARI_NAME = 'PrivateEyeNoFrameskip-v4'
# ATARI_NAME = 'QbertNoFrameskip-v4'
# ATARI_NAME = 'RiverraidNoFrameskip-v4'
# ATARI_NAME = 'RoadRunnerNoFrameskip-v4'
# ATARI_NAME = 'RobotankNoFrameskip-v4'
# ATARI_NAME = 'SeaquestNoFrameskip-v4'
# ATARI_NAME = 'SpaceInvadersNoFrameskip-v4'
# ATARI_NAME = 'StarGunnerNoFrameskip-v4'
# ATARI_NAME = 'TennisNoFrameskip-v4'
# ATARI_NAME = 'TimePilotNoFrameskip-v4'
# ATARI_NAME = 'TutankhamNoFrameskip-v4'
# ATARI_NAME = 'UpNDownNoFrameskip-v4'
# ATARI_NAME = 'VentureNoFrameskip-v4'
# ATARI_NAME = 'VideoPinballNoFrameskip-v4'
# ATARI_NAME = 'WizardOfWorNoFrameskip-v4'
# ATARI_NAME = 'ZaxxonNoFrameskip-v4'

# ATARI_NAME = 'AlienNoFrameskip-v4'
# ATARI_NAME = 'AmidarNoFrameskip-v4'
# ATARI_NAME = 'AssaultNoFrameskip-v4'
# ATARI_NAME = 'AsterixNoFrameskip-v4'
# ATARI_NAME = 'AsteroidsNoFrameskip-v4'
# ATARI_NAME = 'AtlantisNoFrameskip-v4'
# ATARI_NAME = 'BankHeistNoFrameskip-v4'
# ATARI_NAME = 'BattleZoneNoFrameskip-v4'

# MAX_RUNS = 3
NUMBER_ENV = 8
FINAL_STEP = 1e7
tempM = [] # temp. mean score
Mean_Score = []
tempC = [] # temp. mean clip frac
Clip_Fraction = []
clip_fraction = pd.DataFrame(columns=['run', 'update_time', 'clip_fraction'])
tempMeanLength = []
Mean_Length = []
batch_env = None
run = 0
update_time = 0
