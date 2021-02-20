import torch
import arguments as args
import numpy as np
import pandas as pd
class PPO_clip():
    def __init__(self, net, decay,device):
        self.net = net
        self.decay = decay
        self.device = device
        # parameters
        self.value_factor = 1. # paper value
        self.entropy_factor = 0.01
        self.clip_epsilon = 0.1
        self.learning_rate = args.learning_rate
        # self.adjustlr = args.adjustlr
        self.training_epoch = 3
        self.time_horizon = 128  # ENV_NUMBER x TIME_HORIZON should be lager than BATCH_SIZE
        self.batch_size = 256  # can be bigger

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        # print('self.net.parameters():',self.net.parameters())

    def learn(self,current_step,state_batch,action_batch,return_batch,old_value_batch,old_log_prob_batch,adv_batch):
        state_batch = torch.from_numpy(state_batch).to(self.device)
        action_batch = torch.from_numpy(action_batch).to(self.device)
        return_batch = torch.from_numpy(return_batch).to(self.device)

        old_value_batch = torch.from_numpy(old_value_batch).to(self.device)
        old_log_prob_batch = torch.from_numpy(old_log_prob_batch).to(self.device)
        adv_batch = torch.from_numpy(adv_batch).to(self.device)
        # print('state_batch:',state_batch)

        self.alpha = self.decay.value(current_step)
        lr = self.learning_rate * self.alpha

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        policy_head, value_batch = self.net(state_batch)
        # print('value_batch:',value_batch)
        log_prob_batch = policy_head.log_prob(action_batch)
        # print('log_prob_batch : ',log_prob_batch.shape)

        self.v_loss,self.v_others = self.value_loss_clip(value_batch, return_batch,old_value_batch) # todo: not mentioned in paper, but used in openai baselines
        self.v_loss_no_clip,_ = self.value_loss(value_batch, return_batch)
        self.pi_loss,self.pi_others = self.policy_loss(log_prob_batch,old_log_prob_batch,adv_batch)
        # print('entropy :',policy_head.entropy())
        self.entropy = torch.mean(policy_head.entropy())

        loss = self.v_loss_no_clip * self.value_factor - self.pi_loss - self.entropy * self.entropy_factor
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        return self.v_loss.cpu().item(), self.v_loss_no_clip.cpu().item()


    def value_loss(self,value_batch,return_batch):
        value_loss = torch.mean((value_batch - return_batch) ** 2)
        # others = {'r_square':r_square}
        others = None
        return value_loss,others

    def value_loss_clip(self,value_batch, return_batch,old_value_batch):#value clip code level skill 1
        value_clipped = old_value_batch + torch.clamp(value_batch-old_value_batch,-self.clip_epsilon, self.clip_epsilon)
        value_loss_1 = (value_batch - return_batch) ** 2
        value_loss_2 = (return_batch - value_clipped) ** 2
        # print('value loss: ',value_loss_1.shape)
        # print('max: ',torch.max(value_loss_1, value_loss_2).shape)
        value_loss = .5 * torch.mean(torch.max(value_loss_1, value_loss_2))

        # others = {'r_square':r_square}
        others = None
        return value_loss, others

    def policy_loss(self,log_prob_batch,old_log_prob_batch,adv_batch):
        ratio = torch.exp(log_prob_batch - old_log_prob_batch)
        ratio_average = ratio.cpu().detach().numpy().mean()
        #args.Ratio['run_time'].append(args.index)
        #args.Ratio['ratio'].append(ratio_average)

        ratio = ratio.view(-1,1) # take care the dimension here!!!

        # print('log_prob_batch:',log_prob_batch.shape)
        # print('old_log_prob_batch:',old_log_prob_batch.shape)
        # print('adv_batch:',adv_batch.shape)
        # print('ratio:',ratio.shape)

        surrogate_1 = ratio * adv_batch
        surrogate_2 = torch.clamp(ratio, 1 - self.clip_epsilon*self.alpha, 1 + self.clip_epsilon*self.alpha) * adv_batch
        surrogate = torch.min(surrogate_1, surrogate_2)
        policy_loss = torch.mean(surrogate)

        approxkl = .5 * torch.mean((old_log_prob_batch - log_prob_batch)**2)
        # print('ratio : ', torch.gt(torch.abs(ratio-1.),self.clip_epsilon*self.alpha).float())
        clipfrac = torch.mean(torch.gt(torch.abs(ratio-1.),self.clip_epsilon*self.alpha).float())
        # print('clipfrac :',clipfrac)

        args.clip_fraction = args.clip_fraction.append(pd.DataFrame({'run': [args.run], 'update_time':[args.update_time], 'clip_fraction':clipfrac.item()}),ignore_index=True)

        args.tempC.append(clipfrac.item())
        #print(args.clip_fraction)
        args.update_time +=1
        others = {'approxkl':approxkl,'clipfrac':clipfrac}
        return policy_loss, others
