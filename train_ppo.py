''' this module executes the PPO algorithm '''
import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import shutil
import json
import logging
import numpy as np
from silence_tensorflow import silence_tensorflow
import tensorflow as tf
from AIOP.sim_maker import Simulator
from AIOP.ppo import ReplayBuffer, PPOAgent
# from AIOP.ppomodified import ReplayBuffer, PPOAgent
from AIOP.reward import RewardFunction


#--------------------Parameters for Simulator-------------------------------
DATA_DIR='data/norm_data/'
DT1 = ['FI101_AiPV8969']
DT2 = None
DT3 = None
DT4 = None
DT5 = None

#what variables do you want the agent to see?
MV_INDEX = 3
PV_INDEX = 1
SV_INDEX = 2

AGENT_INDEX = [1,2,3,5,6,7]
AGENT_LOOKBACK = 5

#about 3-5x as long as the system needs to respond to SP change
EPISODE_LENGTH = 600
#add some noise to the SV to help with simulating responses.
SV_NOISE = 0.05

#--------------------Parameters for Agent-----------------------
CONTROLLER_NAME = 'LIC01_AiMV'
# GAMMA = 0.96

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
# CLIP_EPSILON = 0.18
ENTROPY_COEF = 0.01
# ENTROPY_COEF = 0.0
CRITIC_COEF = 0.5
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
EPOCHS = 10
# EPOCHS = 100

MAX_STEP = 0.05       	#how much can the agent move each timestep
TRAINING_SCANRATE = 5   #scanrate that the dt was trained on and agent trains on.
EXECUTION_SCANRATE = 5  #rate that the model is to be executed

reward_function = RewardFunction(AGENT_INDEX, MV_INDEX, SV_INDEX, PV_INDEX)

######################################################################
#------------Initalize Simulator from trained Environment-------------
######################################################################
sim = Simulator(dt1=DT1, dt2=DT2, dt3=DT3, dt4=DT4, dt5=DT5, data_dir=DATA_DIR, agentIndex=AGENT_INDEX,
                MVindex=MV_INDEX, SVindex=SV_INDEX, agent_lookback=AGENT_LOOKBACK, training_scanrate=TRAINING_SCANRATE,
                episode_length=EPISODE_LENGTH, SVnoise=SV_NOISE
                )

#######################################################################
#--------------Initalize PPO Agent/ReplayBuffer-----------------------
#######################################################################
buff = ReplayBuffer(AGENT_INDEX, agent_lookback=sim.agent_lookback,
                    capacity=1000000, batch_size=BATCH_SIZE)

# Initialize Agent
agent = PPOAgent(agentIndex=AGENT_INDEX, MVindex=MV_INDEX, agent_lookback=AGENT_LOOKBACK,
                 gamma=GAMMA, lambda_=LAMBDA, clip_epsilon=CLIP_EPSILON,
                 entropy_coef=ENTROPY_COEF, critic_coef=CRITIC_COEF,
                 max_step=MAX_STEP, training_scanrate=TRAINING_SCANRATE,
                 execution_scanrate=EXECUTION_SCANRATE, learning_rate=LEARNING_RATE)

# Build actor and critic networks
agent.build_actor(input_dims=len(AGENT_INDEX), lr=LEARNING_RATE)
agent.build_critic(input_dims=len(AGENT_INDEX), lr=LEARNING_RATE)

################################################################################
#---------------------config file--------------------------------------------
################################################################################

#make a directory to save stuff in
# MODEL_DIR = CONTROLLER_NAME + str(int(round(np.random.rand()*10000, 0))) + '/'
MODEL_DIR = CONTROLLER_NAME + 'temp' + '/'
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
os.mkdir(MODEL_DIR)
print(MODEL_DIR)

#import tag_dictionary
with open('norm_vals.json', 'r', encoding='utf-8') as norm_file:
    tag_dict = json.load(norm_file)

#create a config file
config = {}
config['variables'] = {
    'dependantVar': MV_INDEX,
    'independantVars': AGENT_INDEX,
    'svIndex': SV_INDEX,
    'pvIndex': PV_INDEX,
}
config['agent_lookback'] = AGENT_LOOKBACK
config['training_scanrate'] = TRAINING_SCANRATE
config['execution_scanrate'] = EXECUTION_SCANRATE
config['data_sample_rate'] = sim.timestep
config['gamma'] = GAMMA
config['lambda'] = LAMBDA
config['clip_epsilon'] = CLIP_EPSILON
config['entropy_coef'] = ENTROPY_COEF
config['critic_coef'] = CRITIC_COEF
config['svNoise'] = SV_NOISE
config['episode_length'] = EPISODE_LENGTH
config['max_step'] = MAX_STEP
config['rewards'] = {'general': reward_function.general,
                     'stability': reward_function.stability,
                     'stability_tolerance': reward_function.stability_tolerance,
                     'response': reward_function.response,
                     'response_tolerance': reward_function.response_tolerance
                     }
config['dt'] = {
    'dt1': DT1,
    'dt2': DT2,
    'dt3': DT3,
    'dt4': DT4,
    'dt5': DT5
}

config['physics'] = sim.physics
config['tag_normalize_dict'] = tag_dict

with open(MODEL_DIR + 'config.json', 'w', encoding='utf-8') as outfile:
    json.dump(config, outfile, indent=4)

##############################___PPO___#######################################
# ---------------------For Episode 1, M do------------------------------------
##############################################################################
NUM_EPISODES = 500
scores = []
print("Training Stated!")
for episode in range(NUM_EPISODES):
    score = 0
    state, done = sim.reset()
    control = state[AGENT_LOOKBACK - 1, AGENT_INDEX.index(MV_INDEX)]
    states, actions, rewards, dones, next_states, log_probs = [], [], [], [], [], []

    while not done:
        action, log_prob = agent.select_action(state)
        print("action selected: ",action, "prob: ", log_prob)
        control = action
        # control = np.clip(control, sim.MV_min, sim.MV_max)
        state_, done = sim.step(control)
        reward = reward_function.calculate_reward(state_,control)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        next_states.append(state_)
        log_probs.append(log_prob)
        # print("score: ",score)
        state = state_
        score += reward

    buff.store_episode(states, actions, rewards, dones, next_states, log_probs)
    # agent.learn(buff)
    print("log_prob: ",log_prob, log_prob.shape)
    agent.learn(states, actions, rewards, dones, next_states, log_probs)
    scores.append(score)

    if episode > 25:
        moving_average = np.mean(scores[-25:])
    else:
        moving_average = 0

    if episode % 10 == 0:
        buff.saveEpisodes(MODEL_DIR + 'replaybuffer.csv')
        agent.save_policy(MODEL_DIR + CONTROLLER_NAME)

    print(f'Episode {episode + 1}/{NUM_EPISODES} - Score: {score}, Moving Average: {moving_average:.2f}')
