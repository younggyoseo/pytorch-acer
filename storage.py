import torch
import numpy as np


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class Buffer(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size, buffer_size):
        self.size = buffer_size // num_steps
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.next_idx = 0
        self.num_in_buffer = 0
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
            action_size = action_space.n
        else:
            action_shape = action_space.shape[0]
            action_size = action_shape
        
        # Memory
        self.obs = torch.zeros(self.size, num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(self.size, num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(self.size, num_steps, num_processes, 1)
        self.policies = torch.zeros(self.size, num_steps, num_processes, action_size)
        self.q_values = torch.zeros(self.size, num_steps, num_processes, action_size)
        self.action_log_probs = torch.zeros(self.size, num_steps, num_processes, 1)
        self.actions = torch.zeros(self.size, num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(self.size, num_steps + 1, num_processes, 1)
    
    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.policies = self.policies.to(device)
        self.q_values = self.q_values.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def has_atleast(self, frames):
        return self.num_in_buffer >= (frames // self.num_steps)
    
    def can_sample(self):
        return self.num_in_buffer > 0

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, policies, q_values,
               rewards, masks):
        self.obs[self.next_idx].copy_(obs)
        self.recurrent_hidden_states[self.next_idx].copy_(recurrent_hidden_states)
        self.actions[self.next_idx].copy_(actions)
        self.action_log_probs[self.next_idx].copy_(action_log_probs)
        self.policies[self.next_idx].copy_(policies)
        self.q_values[self.next_idx].copy_(q_values)
        self.rewards[self.next_idx].copy_(rewards)
        self.masks[self.next_idx].copy_(masks)

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
    
    def take(self, x, idx, envx):
        num_processes = self.num_processes
        out = torch.zeros(*x.shape[1:])
        for i in range(num_processes):
            out[:, i, :] = x[idx[i], :, envx[i]]
        return out

    def get(self):
        num_processes = self.num_processes
        assert self.can_sample()

        # Sample exactly one id per env. If you sample across envs, then higher correlation in samples from same env.
        idx = np.random.randint(0, self.num_in_buffer, num_processes)
        envx = np.arange(num_processes)

        take = lambda x: self.take(x, idx, envx)
        obs = take(self.obs)
        recurrent_hidden_states = take(self.recurrent_hidden_states)
        rewards = take(self.rewards)
        policies = take(self.policies)
        q_values = take(self.q_values)
        action_log_probs = take(self.action_log_probs)
        actions = take(self.actions).long()
        masks = take(self.masks)
        return obs, recurrent_hidden_states, rewards, policies, q_values, action_log_probs, actions, masks


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
            action_size = action_space.n
        else:
            action_shape = action_space.shape[0]
            action_size = action_shape
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.policies = torch.zeros(num_steps, num_processes, action_size)
        self.q_values = torch.zeros(num_steps, num_processes, action_size)
        self.q_retraces = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.policies = self.policies.to(device)
        self.q_values = self.q_values.to(device)
        self.q_retraces = self.q_retraces.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, policies, 
               q_values, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.policies[self.step].copy_(policies)
        self.q_values[self.step].copy_(q_values)

        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
