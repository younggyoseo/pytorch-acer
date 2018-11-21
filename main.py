import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import Buffer, RolloutStorage
from utils import get_vec_normalize

args = get_args()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    average_actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    average_actor_critic.load_state_dict(actor_critic.state_dict())
    actor_critic.to(device)
    average_actor_critic.to(device)

    agent = algo.ACER_AGENT(actor_critic, average_actor_critic,
                    args.value_loss_coef, args.entropy_coef,
                    args.gamma, args.clip, args.no_trust_region, args.alpha, args.delta,
                    lr=args.lr, eps=args.eps, rms_alpha=args.rms_alpha,
                    max_grad_norm=args.max_grad_norm)

    buffer = Buffer(args.num_steps, args.num_processes,
                    envs.observation_space.shape, envs.action_space,
                    actor_critic.recurrent_hidden_state_size,
                    args.buffer_size)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)
    
    off_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    off_rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    acer = algo.ACER(actor_critic, rollouts, off_rollouts, buffer, episode_rewards, agent, envs)

    start = time.time()
    for j in range(num_updates):
        # On-policy ACER
        value_loss, action_loss, dist_entropy = acer.call(on_policy=True)
        if args.replay_ratio > 0 and buffer.has_atleast(args.replay_start):
            # Off-policy ACER
            n = np.random.poisson(args.replay_ratio)
            for _ in range(n):
                acer.call(on_policy=False)

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \nLast {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\ndist_entropy {:.1f}, value/action loss {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards),
                       dist_entropy,
                       value_loss,
                       action_loss))

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            eval_episode_rewards = []

            obs = eval_envs.reset().to(device)
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, _, _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                            obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, _, done, infos = eval_envs.step(action)

                obs = obs.to(device)
                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done]).to(device)
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))


if __name__ == "__main__":
    main()
