import numpy as np

# need to speed up, such as put them on numpy directly or on Tensor
class Buffer():
    def __init__(self):
        self.initialize_buffer()

    def initialize_buffer(self):
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.terminal_list = []
        self.value_list = []
        self.log_prob_list = []

    def add_data(self,state_t=None,action_t=None,reward_t=None,terminal_t=None,value_t=None,log_prob_t=None):
        if state_t is not None:
            self.state_list.append(state_t)
        if action_t is not None:
            self.action_list.append(action_t)
        if reward_t is not None:
            self.reward_list.append(reward_t)
        if terminal_t is not None:
            self.terminal_list.append(terminal_t)
        if value_t is not None:
            self.value_list.append(value_t)
        if log_prob_t is not None:
            self.log_prob_list.append(log_prob_t)

class BatchBuffer():
    def __init__(self,buffer_num,gamma,lam):
        self.buffer_num = buffer_num
        self.buffer_list = [Buffer() for _ in range(self.buffer_num)]
        self.gamma = gamma
        self.lam = lam
    def initialize_buffer_list(self):
        for buffer in self.buffer_list:
            buffer.initialize_buffer()

    def add_batch_data(self,states_t=None,actions_t=None,rewards_t=None,terminals_t=None,values_t=None,log_probs_t=None):
        for i in range(self.buffer_num):
            self.buffer_list[i].add_data(states_t[i],actions_t[i],rewards_t[i],terminals_t[i],values_t[i],log_probs_t[i])

    def buffer_list_to_array(self):
        states = []
        actions = []
        rewards = []
        terminals = []
        values = []
        log_probs = []
        for buffer in self.buffer_list:
            states.append(buffer.state_list)
            actions.append(buffer.action_list)
            rewards.append(buffer.reward_list)
            terminals.append(buffer.terminal_list)
            values.append(buffer.value_list)
            log_probs.append(buffer.log_prob_list)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        values = np.array(values)
        log_probs = np.array(log_probs)

        return states,actions,rewards,terminals,values,log_probs

    def compute_reward_to_go_returns(self,rewards,values,terminals):
        '''
        the env will reset directly once it ends and return a new state
        st is only one more than at and rt at the end of the episode
        state:    s1 s2 s3 ... st-1 -
        action:   a1 a2 a3 ... at-1 -
        reward:   r1 r2 r3 ... rt-1 -
        terminal: t1 t2 t3 ... tt-1 -
        value:    v1 v2 v3 ... vt-1 vt
        '''
        # (N,T) -> (T,N)   N:n_envs   T:traj_length
        rewards = np.transpose(rewards,[1,0])
        values = np.transpose(values, [1, 0])
        terminals = np.transpose(terminals,[1,0])
        R = values[-1]
        returns = []

        for i in reversed(range(rewards.shape[0])):
            R = rewards[i] + (1. - terminals[i]) * self.gamma * R
            returns.append(R)
        returns = list(reversed(returns))
        # (T,N) -> (N,T)
        returns = np.transpose(returns,[1,0])
        return returns

    def compute_GAE(self,rewards,values,terminals):
        # (N,T) -> (T,N)
        rewards = np.transpose(rewards,[1,0])
        values = np.transpose(values,[1,0])
        terminals = np.transpose(terminals,[1,0])
        length = rewards.shape[0]
        # print('reward:{},value:{},terminal{}'.format(rewards.shape,values.shape,terminals.shape))
        deltas = []
        for i in reversed(range(length)):
            v = rewards[i] + (1. - terminals[i]) * self.gamma * values[i+1]
            delta = v - values[i]
            deltas.append(delta)
        deltas = np.array(list(reversed(deltas)))

        A = deltas[-1,:]
        advantages = [A]
        for i in reversed(range(length-1)):
            A = deltas[i] + (1. - terminals[i]) * self.gamma * self.lam * A
            advantages.append(A)
        advantages = reversed(advantages)
        # (T,N) -> (N,T)
        advantages = np.transpose(list(advantages),[1,0])
        # print(advantages)
        return advantages

    def get_data(self):
        states, actions, rewards, terminals, values, log_probs = self.buffer_list_to_array()
        advs = self.compute_GAE(rewards,values,terminals)
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
        returns = self.compute_reward_to_go_returns(rewards,values,terminals)

        return states[:,:len(advs[0])],actions[:,:len(advs[0])],returns,values[:,:len(advs[0])],log_probs[:,:len(advs[0])],advs

    def shuffle_data(self,states, state_space, actions,returns,values,log_probs,advs):
        states = np.reshape(states, [-1] + state_space)
        actions = np.reshape(actions, [-1])
        returns = np.reshape(returns, [-1, 1])
        values = np.reshape(values, [-1, 1])
        log_probs = np.reshape(log_probs, [-1])
        advs = np.reshape(advs, [-1, 1])

        indices = np.random.permutation(range(len(advs))).tolist()

        states = states[indices]
        actions = actions[indices]
        returns = returns[indices]
        values = values[indices]
        log_probs = log_probs[indices]
        advs = advs[indices]

        return states,actions,returns,values,log_probs,advs

    def get_minibatch(self,startingIndex, batch_size,states,actions,returns,values,log_probs,advs):
        batch_states = states[startingIndex : startingIndex+batch_size]
        batch_actions = actions[startingIndex : startingIndex+batch_size]

        batch_returns = returns[startingIndex : startingIndex+batch_size]
        batch_values = values[startingIndex : startingIndex + batch_size]
        batch_log_probs = log_probs[startingIndex : startingIndex + batch_size]
        batch_advs = advs[startingIndex : startingIndex + batch_size]

        return batch_states, batch_actions, batch_returns, batch_values, batch_log_probs, batch_advs