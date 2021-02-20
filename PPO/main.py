import os
from env_wrappers import *
import arguments as args
from agents import PPO_Agent
from networks import nature_cnn
import pandas as pd
import torch
import random
import json
def run_atari(train = True, render = False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA_VISIBLE_DEVICES)
    args.batch_env = Baselines_DummyVecEnv(env_id=args.ATARI_NAME,num_env=args.NUMBER_ENV)
    agent = PPO_Agent(args.batch_env.action_space,args.batch_env.observation_space,nature_cnn)

    states = args.batch_env.reset()
    rewards, dones, info = None,None,None
    current_step = 0
    tstart = time.time()
    while current_step < args.FINAL_STEP:
        actions = agent.act(states,rewards, dones, info, train=train,current_step=current_step)
        next_states, rewards, dones, info = args.batch_env.step(actions)
        if render:
            args.batch_env.render()
        states = next_states
        current_step += args.batch_env.get_num_of_envs()
        if current_step % 10000 == 0:
            tnow = time.time()
            fps = current_step / (tnow - tstart)
            print('game: {}, run: {}, lr:{}, reward_scale:{}, current_step: {:.2e}, time: {:.2f}, fps: {:.2f}, mean reward: {}, mean length: {}'.format(
                args.ATARI_NAME, [args.run,args.SEED_LIST],args.learning_rate,args.reward_scale, current_step, tnow - tstart, fps,args.batch_env.get_episode_rewmean(),args.batch_env.get_episode_lenmean()))
    
    args.Mean_Score.append(args.tempM)
    args.Clip_Fraction.append(args.tempC)
    args.Mean_Length.append(args.tempMeanLength)

    args.tempC = []
    args.tempM = []
    args.tempMeanLength = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('Starting...')
    print('lr: ', args.learning_rate)
    # while args.run < args.MAX_RUNS:
    #     seed = 1000*(args.run+1)
    #     setup_seed(seed)
    #
    #     run_atari()
    #     args.run += 1
    #     args.update_time = 0
    print(torch.cuda.is_available())

    print(args.SEED_LIST)
    for item in args.SEED_LIST:
        seed = 1000 * item
        setup_seed(seed)
        args.other_reward, args.zero_reward_num, args.ten_reward_num = 0, 0, 0
        run_atari()

        args.run += 1
        args.update_time = 0
        args.record_time = 0

        # path = "loss_data" + str(args.run) +'-'+str(args.learning_rate[args.lr])+'-'+str(args.reward_scale)+ ".json"
        # with open(path, mode='w') as loss_data_file:
        #     json.dump(args.returns, loss_data_file)
        # args.returns = {}


        # print(len(args.Mean_Score[0]))


    print('Saving data...')
    mean_score = pd.DataFrame(args.Mean_Score).melt(var_name='iteration', value_name='mean_score') # thiskind of dict is easy to draw line plots with std
    clip_fraction = pd.DataFrame(args.Clip_Fraction).melt(var_name='update_time', value_name='clip_fraction')
    mean_length = pd.DataFrame(args.Mean_Length).melt(var_name='iteration', value_name='mean_length')


    args.reward_num.to_csv(args.ATARI_NAME + '_' + str(args.learning_rate)+'_' + str(args.SEED_LIST)+'_' +str(args.reward_scale) + "_reward_num.csv", index=False)
    print("Save reward_num successfully.")
    # save mean score
    mean_score.to_csv(args.ATARI_NAME+'_'+str(args.learning_rate)+'_' + str(args.SEED_LIST)+'_'+str(args.reward_scale) +"_mean_score_std.csv", index=False)
    print("Save mean_score successfully.")

    # save mean length
    mean_length.to_csv(args.ATARI_NAME+'_'+str(args.learning_rate)+'_' + str(args.SEED_LIST)+'_'+str(args.reward_scale) +"_mean_length_std.csv", index=False)
    print("Save mean_length successfully.")

    # save clip fraction
    clip_fraction.to_csv(args.ATARI_NAME+'_'+str(args.learning_rate)+'_' + str(args.SEED_LIST)+'_'+str(args.reward_scale) +"_clip_fraction_std.csv", index=False)

    args.clip_fraction.to_csv(args.ATARI_NAME+'_'+str(args.learning_rate)+'_' + str(args.SEED_LIST)+'_'+str(args.reward_scale)+"_clip_fraction.csv", index=False) # easy to distinguish runs
    print("Save clip_fraction successfully.")
    print('Done.')