import torch
import torch.nn as nn
import torch.optim as optim

class ACER():
    def __init__(self, actor_critic, on_rollouts, off_rollouts, buffer, episode_rewards, agent, envs):
        self.actor_critic = actor_critic
        self.on_rollouts = on_rollouts
        self.off_rollouts = off_rollouts
        self.buffer = buffer
        self.episode_rewards = episode_rewards
        self.agent = agent
        self.envs = envs
        self.num_steps = on_rollouts.num_steps
        self.device = self.on_rollouts.obs.device
    
    def call(self, on_policy):
        actor_critic, on_rollouts, off_rollouts, buffer, episode_rewards, agent, envs =\
            self.actor_critic, self.on_rollouts, self.off_rollouts, self.buffer, self.episode_rewards, self.agent, self.envs

        rollouts = on_rollouts if on_policy else off_rollouts

        if on_policy:
            for step in range(self.num_steps):
                with torch.no_grad():
                    policy, _, q_value, action, _, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                
                obs, reward, done, infos = envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                
                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                            for done_ in done])
                rollouts.insert(obs, recurrent_hidden_states, action, policy, 
                                q_value, reward, masks)

            buffer.insert(rollouts.obs, rollouts.recurrent_hidden_states, rollouts.actions,
                        rollouts.policies, rollouts.q_values, rollouts.rewards, rollouts.masks)

        else:
            # Off Policy
            rollouts.obs, rollouts.recurrent_hidden_states, rollouts.rewards,\
                rollouts.policies, rollouts.q_values, rollouts.actions,\
                    rollouts.masks = buffer.get()
            rollouts.to(self.device)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, on_policy)
        rollouts.after_update()
        return value_loss, action_loss, dist_entropy

class ACER_AGENT():
    def __init__(self,
                 actor_critic,
                 average_actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 gamma,
                 clip,
                 no_trust_region,
                 alpha,
                 delta,
                 lr=None,
                 eps=None,
                 rms_alpha=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic
        self.average_actor_critic = average_actor_critic

        self.gamma = gamma
        self.clip = clip
        self.no_trust_region = no_trust_region
        self.delta = delta
        self.alpha = alpha
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=rms_alpha)

    def update(self, rollouts, on_policy):
        if on_policy:
            return self._update_on_policy(rollouts)
        else:
            return self._update_off_policy(rollouts)
    
    def _update_on_policy(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        action_size = rollouts.policies.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        policies, values, q_values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        policies = policies.view(-1, action_size)
        values = values.view(-1, 1)
        q_values = q_values.view(-1, action_size)
        action_log_probs = action_log_probs.view(-1, 1)
        actions = rollouts.actions.view(-1, action_shape)

        q_i = q_values.gather(1, actions)
        rho_i = torch.ones_like(q_i)

        with torch.no_grad():
            next_value = self.actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        q_retraces = compute_q_retraces(rollouts.rewards, rollouts.masks,
                                        values.view(num_steps, num_processes, 1),
                                        q_i.view(num_steps, num_processes, 1),
                                        rho_i.view(num_steps, num_processes, 1),
                                        next_value, self.gamma)

        q_retraces = q_retraces.view(-1, 1)
        advantages = q_retraces - values

        action_loss = -(rho_i.clamp(max=self.clip) * action_log_probs * advantages.detach()).mean()
        value_loss = (q_i - q_retraces.detach()).pow(2).mean()

        loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * dist_entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()

        for param, avg_param in zip(self.actor_critic.parameters(),
                                    self.average_actor_critic.parameters()):
            avg_param = avg_param * self.alpha + param * (1-self.alpha)

        return value_loss.item(), action_loss.item(), dist_entropy.item()
    
    def _update_off_policy(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        action_size = rollouts.policies.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        policies, values, q_values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        average_policies, _, _, _, _, _ = self.average_actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.average_actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        old_policies = rollouts.policies.view(-1, action_size)
        policies = policies.view(-1, action_size)
        average_policies = average_policies.view(-1, action_size)

        values = values.view(-1, 1)
        q_values = q_values.view(-1, action_size)
        action_log_probs = action_log_probs.view(-1, 1)
        actions = rollouts.actions.view(-1, action_shape)

        q_i = q_values.gather(1, actions)
        rho = policies / (old_policies + 1e-10)
        rho_i = rho.gather(1, actions)

        with torch.no_grad():
            next_value = self.actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        q_retraces = compute_q_retraces(rollouts.rewards, rollouts.masks,
                                        values.view(num_steps, num_processes, 1),
                                        q_i.view(num_steps, num_processes, 1),
                                        rho_i.view(num_steps, num_processes, 1),
                                        next_value, self.gamma)

        q_retraces = q_retraces.view(-1, 1)
        advantages = q_retraces - values

        # Truncated importance sampling
        loss_f = -((rho_i.clamp(max=self.clip) * advantages).detach() * action_log_probs).mean()

        # Bias correction for the truncation
        adv_bc = (q_values - values)
        logf_bc = (policies + 1e-10).log()
        gain_bc = logf_bc * ((adv_bc * (1.0 - (self.clip / (rho + 1e-10))).clamp(min=0) * policies).detach())
        loss_bc = -gain_bc.sum(-1).mean()

        action_loss = loss_f + loss_bc
        value_loss = (q_i - q_retraces.detach()).pow(2).mean() * 0.5

        self.optimizer.zero_grad()
        if not self.no_trust_region:
            # Trust Region Policy Optimization
            g = torch.autograd.grad(-(action_loss - self.entropy_coef * dist_entropy) * num_steps * num_processes, policies)[0]
            k = -average_policies / (policies + 1e-10)
            k_dot_g = (k * g).sum(-1)
            k_dot_k = (k * k).sum(-1)
            adj = ((k_dot_g - self.delta) / (k_dot_k + 1e-10)).clamp(min=0)

            g = g - adj.unsqueeze(1) * k
            grads_f = -g / (num_steps * num_processes)
            torch.autograd.backward(policies, grad_tensors=(grads_f,), retain_graph=True)
            (self.value_loss_coef * value_loss).backward()
        else:
            loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * dist_entropy
            loss.backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()

        for param, avg_param in zip(self.actor_critic.parameters(),
                                    self.average_actor_critic.parameters()):
            avg_param = avg_param * self.alpha + param * (1-self.alpha)

        return value_loss.item(), action_loss.item(), dist_entropy.item()

def compute_q_retraces(rewards, masks, values, q_i, rho_i, next_value, gamma):
    num_steps, num_processes = rewards.shape[:2]
    q_retraces = rewards.new(num_steps + 1, num_processes, 1).zero_()
    q_retraces[-1] = next_value
    for step in reversed(range(rewards.size(0))):
        q_ret = rewards[step] + gamma * q_retraces[step + 1] * masks[step + 1]
        q_retraces[step] = q_ret
        q_ret = (rho_i[step] * (q_retraces[step] - q_i[step])) + values[step]
    return q_retraces[:-1]

def compute_avg_norm(arr):
    return arr.pow(2).sum(-1).sqrt().mean()