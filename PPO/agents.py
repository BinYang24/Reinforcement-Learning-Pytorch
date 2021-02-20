import numpy as np
import torch
from rl_algorithms import PPO_clip
from schedules import LinearSchedule
import arguments as args
from buffers import BatchBuffer
import seaborn as sns
import matplotlib.pyplot as plt
import arguments as args
import json
import pandas as pd
class PPO_Agent():
    def __init__(self,action_space, state_space,net):
        self.action_space = action_space
        self.state_space = state_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        self.net = net(self.action_space,self.state_space).to(self.device)
        self.decay = LinearSchedule(schedule_timesteps=args.FINAL_STEP, final_p=0.)
        self.update = PPO_clip(self.net, self.decay,self.device)
        self.batch_buffer = BatchBuffer(buffer_num=args.NUMBER_ENV,gamma=0.99,lam=0.95)

    def act(self,states,rewards,dones,info,train,current_step):
        states = torch.from_numpy(np.array(states)).to(self.device).float()
        policy_head, values = self.net(states)
        # print('policy head: ',policy_head)
        # print('values: ',values)
        actions = policy_head.sample()
        log_probs = policy_head.log_prob(actions)

        if train:
            self.train(states.detach().cpu().numpy(),
                       actions.detach().cpu().numpy(),
                       rewards,
                       dones,
                       values.detach().cpu().numpy(),
                       log_probs.detach().cpu().numpy(),
                       current_step)

        return actions.detach().cpu().numpy()

    def train(self,states,actions,rewards,dones,values,log_probs,current_step):
        values = np.reshape(values, [-1])
        if rewards is None and dones is None:
            for i in range(self.batch_buffer.buffer_num):
                self.batch_buffer.buffer_list[i].add_data(
                    state_t=states[i],
                    action_t=actions[i],
                    value_t=values[i],
                    log_prob_t=log_probs[i])
        else:
            for i in range(self.batch_buffer.buffer_num):
                self.batch_buffer.buffer_list[i].add_data(
                    state_t=states[i],
                    action_t=actions[i],
                    reward_t=rewards[i],
                    terminal_t=dones[i],
                    value_t=values[i],
                    log_prob_t=log_probs[i])

        if current_step > 0 and current_step / self.batch_buffer.buffer_num % self.update.time_horizon == 0:
            #print(np.shape(self.batch_buffer.buffer_list))

            args.tempM.append(args.batch_env.get_episode_rewmean())
            args.tempMeanLength.append(args.batch_env.get_episode_lenmean())
            args.reward_num = args.reward_num.append(pd.DataFrame({'run': [args.run], 'record_time': [args.record_time], '0': [args.zero_reward_num], '10': [args.ten_reward_num], 'others': [args.other_reward]}))

            s, a, ret, v, logp, adv = self.batch_buffer.get_data()
            args.other_reward, args.ten_reward_num, args.zero_reward_num = 0, 0, 0

            # miu = np.mean(ret, axis=1).reshape(-1, 1)
            # std = np.std(ret, axis=1).reshape(-1, 1)
            # # print(miu.shape, std.shape)
            # ret = (ret - miu) / (std + 1e-8)
            for epoch in range(self.update.training_epoch):
                s, a, ret, v, logp, adv = self.batch_buffer.shuffle_data(s, self.state_space, a, ret, v, logp, adv)
                
                num_batch = self.update.time_horizon*self.batch_buffer.buffer_num // self.update.batch_size
                for i in range(num_batch):
                    
                    batch_s, batch_a, batch_ret, batch_v, batch_logp, batch_adv = self.batch_buffer.get_minibatch(i*self.update.batch_size,
                                                                                                                  self.update.batch_size,
                                                                                                                  s, a,
                                                                                                                  ret,
                                                                                                                  v,
                                                                                                                  logp,
                                                                                                                  adv)
                    # print('minibatch data shape:', s.shape, a.shape, ret.shape, v.shape,logp.shape, adv.shape)
                    # print('minibatch data shape:', batch_s.shape, batch_a.shape, batch_ret.shape, batch_v.shape,batch_logp.shape, batch_adv.shape)

                    _, _ = self.update.learn(current_step, batch_s, batch_a, batch_ret, batch_v, batch_logp, batch_adv)

            args.record_time += 1
            #reward
            self.batch_buffer.initialize_buffer_list()

            states = torch.from_numpy(states).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            policy_head, values = self.net(states)
            log_probs = policy_head.log_prob(actions)
            log_probs = log_probs.detach().cpu().numpy()
            states = states.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            values = values.detach().cpu().numpy()
            values = np.reshape(values, [-1])

            for i in range(self.batch_buffer.buffer_num):
                self.batch_buffer.buffer_list[i].add_data(
                    state_t=states[i],
                    action_t=actions[i],
                    value_t=values[i],
                    log_prob_t=log_probs[i]) # here may be the problem
