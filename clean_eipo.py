# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import pickle


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MontezumaRevenge-v5",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=2000000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=128,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--sticky-action", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, sticky action will be used")

    # EIPO arguments
    parser.add_argument("--alpha-lr", type=float, default=0.01,
                        help="learning rate of lagrangian multiplier")
    parser.add_argument("--alpha-g-clip", type=float, default=0.05,
                        help="clip on alpha update")
    parser.add_argument("--alpha-clip", type=float, default=10,
                        help="clip on alpha value")

    # RND arguments
    parser.add_argument("--update-proportion", type=float, default=0.25,
                        help="proportion of exp used for predictor update")
    parser.add_argument("--int-coef", type=float, default=1.0,
                        help="coefficient of extrinsic reward")
    parser.add_argument("--ext-coef", type=float, default=2.0,
                        help="coefficient of intrinsic reward")
    parser.add_argument("--int-gamma", type=float, default=0.99,
                        help="Intrinsic reward discount rate")
    parser.add_argument("--num-iterations-obs-norm-init", type=int, default=50,
                        help="number of iterations to initialize the observations normalization parameters")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, init_alpha=0.5):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor_ei = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.actor_e = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ei_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_ei_int = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_e_ext = layer_init(nn.Linear(448, 1), std=0.01)

        self.rollout_by_pi_e = True
        self.alpha = init_alpha

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)

        # EI
        logits_ei = self.actor_ei(hidden)
        probs_ei = Categorical(logits=logits_ei)

        # E
        logits_e = self.actor_e(hidden)
        probs_e = Categorical(logits=logits_e)

        if action is None:
            action_ei = probs_ei.sample()
            action_e = probs_e.sample()
        else:
            action_ei = action
            action_e = action

        return (
            action_ei,
            action_e,
            probs_ei.log_prob(action_ei),
            probs_e.log_prob(action_e),
            probs_ei.entropy(),
            probs_e.entropy(),
            self.critic_ei_ext(features + hidden),
            self.critic_ei_int(features + hidden),
            self.critic_e_ext(features + hidden)
        )

    def get_value(self, x):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ei_ext(features + hidden), self.critic_ei_int(features + hidden), self.critic_e_ext(
            features + hidden)

    def is_rollout_by_pi_e(self):
        return self.rollout_by_pi_e

    def maybe_switch_rollout_pi(self,
                                old_max_objective_value, new_max_objective_value,
                                old_min_objective_value, new_min_objective_value):
        if self.is_rollout_by_pi_e() and old_max_objective_value is not None:
            if (new_max_objective_value - old_max_objective_value) < 0:
                self.rollout_by_pi_e = False
                print("Switch to pi_EI")
        elif not self.is_rollout_by_pi_e() and old_min_objective_value is not None:
            if (new_min_objective_value - old_min_objective_value) < 0:
                self.rollout_by_pi_e = True
                print("Switch to pi_E")


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        repeat_action_probability=0.25,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    rnd_model = RNDModel(4, envs.single_action_space.n).to(device)
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_ei = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    actions_e = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_ei = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_e = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ei_ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ei_int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    e_ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    if not os.path.exists(f".atari_norm/{args.env_id}.pkl"):
        print("Start to initialize observation normalization parameter.....")
        next_ob = []
        for step in tqdm(range(args.num_steps * args.num_iterations_obs_norm_init)):
            acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
            s, r, d, _ = envs.step(acs)
            next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()

            if len(next_ob) % (args.num_steps * args.num_envs) == 0:
                next_ob = np.stack(next_ob)
                obs_rms.update(next_ob)
                next_ob = []
        os.makedirs(".atari_norm", exist_ok=True)
        with open(f".atari_norm/{args.env_id}.pkl", "wb") as f:
            pickle.dump(obs_rms, f)
    else:
        print(f"Load obs_rms from .atari_norm/{args.env_id}.pkl")
        with open(f".atari_norm/{args.env_id}.pkl", "rb") as f:
            obs_rms = pickle.load(f)
    print("End to initialize...")

    old_max_objective_value = None
    new_max_objective_value = None
    old_min_objective_value = None
    new_min_objective_value = None

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value_ei_ext, value_ei_int, value_e_ext = agent.get_value(obs[step])
                ei_ext_values[step], ei_int_values[step], e_ext_values[step] = (
                    value_ei_ext.flatten(),
                    value_ei_int.flatten(),
                    value_e_ext.flatten(),
                )
                action_ei, action_e, logprob_ei, logprob_e, _, _, _, _, _ = agent.get_action_and_value(obs[step])

            actions_ei[step] = action_ei
            actions_e[step] = action_e
            logprobs_ei[step] = logprob_ei
            logprobs_e[step] = logprob_e

            # TRY NOT TO MODIFY: execute the game and log data.
            rollout_action = action_e if agent.is_rollout_by_pi_e() else action_ei
            next_obs, reward, done, info = envs.step(rollout_action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            rnd_next_obs = (
                (
                        (next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(
                            device))
                        / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                ).clip(-5, 5)
            ).float()
            target_next_feature = rnd_model.target(rnd_next_obs)
            predict_next_feature = rnd_model.predictor(rnd_next_obs)
            curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    avg_returns.append(info["r"][idx])
                    epi_ret = np.average(avg_returns)
                    print(
                        f"global_step={global_step}, episodic_return={info['r'][idx]}, curiosity_reward={np.mean(curiosity_rewards[step].cpu().numpy())}"
                    )
                    writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar(
                        "charts/episode_curiosity_reward",
                        curiosity_rewards[step][idx],
                        global_step,
                    )
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        reward_rms.update_from_moments(mean, std ** 2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ei_ext, next_value_ei_int, next_value_e_ext = agent.get_value(next_obs)
            next_value_ei_ext, next_value_ei_int, next_value_e_ext = next_value_ei_ext.reshape(1,
                                                                                               -1), next_value_ei_int.reshape(
                1, -1), next_value_e_ext.reshape(1, -1)
            ei_ext_advantages = torch.zeros_like(rewards, device=device)
            ei_int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ei_advantages = torch.zeros_like(rewards, device=device)  # U_{max}
            ei_ext_lastgaelam = 0
            ei_int_lastgaelam = 0
            e_ext_advantages = torch.zeros_like(rewards, device=device)
            e_advantages = torch.zeros_like(rewards, device=device)  # U_min
            e_ext_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ei_ext_nextnonterminal = 1.0 - next_done
                    ei_int_nextnonterminal = 1.0
                    ei_ext_nextvalues = next_value_ei_ext
                    ei_int_nextvalues = next_value_ei_int
                    e_ext_nextnonterminal = 1.0 - next_done
                    e_ext_nextvalues = next_value_e_ext
                else:
                    ei_ext_nextnonterminal = 1.0 - dones[t + 1]
                    ei_int_nextnonterminal = 1.0
                    ei_ext_nextvalues = ei_ext_values[t + 1]
                    ei_int_nextvalues = ei_int_values[t + 1]
                    e_ext_nextnonterminal = 1.0 - dones[t + 1]
                    e_ext_nextvalues = e_ext_values[t + 1]
                ei_ext_delta = rewards[t] + args.gamma * ei_ext_nextvalues * ei_ext_nextnonterminal - ei_ext_values[t]
                ei_int_delta = curiosity_rewards[t] + args.int_gamma * ei_int_nextvalues * ei_int_nextnonterminal - \
                               ei_int_values[t]
                e_ext_delta = rewards[t] + args.gamma * e_ext_nextvalues * e_ext_nextnonterminal - e_ext_values[t]
                ei_ext_advantages[t] = ei_ext_lastgaelam = (
                        ei_ext_delta + args.gamma * args.gae_lambda * ei_ext_nextnonterminal * ei_ext_lastgaelam
                )
                ei_int_advantages[t] = ei_int_lastgaelam = (
                        ei_int_delta + args.int_gamma * args.gae_lambda * ei_int_nextnonterminal * ei_int_lastgaelam
                )

            # U_max = (1 + alpha) * r_E + r_I + \gamma * \alpha * V^pi_E_E(s') - \alpha * V^\pi_E(s)
            #   = alpha * (r_E + \gamma * V^pi_E_E(s') - V^\pi_E_E(s)) + r_E + r_I
            #   = alpha * A^pi_E(s) + r_E + r_I
            ei_advantages = (rewards + curiosity_rewards) + agent.alpha * e_ext_advantages

            # U_min = (\alpha * rE + gamma * ((1+\alpha) * V_E + V_I)(s') - ((1+\alpha) * V_E + V_I)(s))
            #   = \alpha * (rE + gamma * V_E(s') - V_E(s)) + gamma * (V_E + V_I)(s') - (V_E + V_I)(s)
            #   = \alpha * A_E(s) + A_{E+I}(s) - (r_{E} + r_I)
            #   = (1 + \alpha) * A_E(s) + A_I(s) - (r_E + r_I)
            e_advantages = (1 + agent.alpha) * ei_ext_advantages + \
                           ei_int_advantages - \
                           (rewards + curiosity_rewards)

            ei_ext_returns = ei_ext_advantages + ei_ext_values
            ei_int_returns = ei_int_advantages + ei_int_values
            e_ext_returns = e_ext_advantages + e_ext_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs_ei = logprobs_ei.reshape(-1)
        b_actions_ei = actions_ei.reshape(-1)
        b_ei_ext_advantages = ei_ext_advantages.reshape(-1)
        b_ei_int_advantages = ei_int_advantages.reshape(-1)
        b_ei_advantages = ei_advantages.reshape(-1)
        b_ei_ext_returns = ei_ext_returns.reshape(-1)
        b_ei_int_returns = ei_int_returns.reshape(-1)
        b_ei_ext_values = ei_ext_values.reshape(-1)

        b_logprobs_e = logprobs_e.reshape(-1)
        b_actions_e = actions_e.reshape(-1)
        b_e_ext_advantages = e_ext_advantages.reshape(-1)
        b_e_advantages = e_advantages.reshape(-1)
        b_e_ext_returns = e_ext_returns.reshape(-1)
        b_e_ext_values = e_ext_values.reshape(-1)

        b_ei_eipo_advantages = b_ei_advantages
        b_ei_ppo_advantages = b_ei_int_advantages * args.int_coef + b_ei_ext_advantages * args.ext_coef

        b_e_eipo_advantages = b_e_advantages
        b_e_ppo_advantages = b_e_ext_advantages

        obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        rnd_next_obs = (
            (
                    (b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
            ).clip(-5, 5)
        ).float()

        clipfracs_ei_ei = []
        clipfracs_e_e = []
        clipfracs_ei_e = []
        clipfracs_e_ei = []
        alpha_derivatives = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)

                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )

                # Depends on who collect the data
                b_actions = b_actions_e if agent.is_rollout_by_pi_e() else b_actions_ei

                _, _, newlogprob_ei, newlogprob_e, entropy_ei, entropy_e, new_ei_ext_values, new_ei_int_values, new_e_ext_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                # EI_new / EI_old
                logratio_ei_ei = newlogprob_ei - b_logprobs_ei[mb_inds]
                ratio_ei_ei = logratio_ei_ei.exp()
                # EI_new / E_old
                logratio_ei_e = newlogprob_ei - b_logprobs_e[mb_inds]
                ratio_ei_e = logratio_ei_e.exp()
                # E_new / E_old
                logratio_e_e = newlogprob_e - b_logprobs_e[mb_inds]
                ratio_e_e = logratio_e_e.exp()
                # E_new / EI_old
                logratio_e_ei = newlogprob_e - b_logprobs_ei[mb_inds]
                ratio_e_ei = logratio_e_ei.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl_ei_ei = (-logratio_ei_ei).mean()
                    approx_kl_ei_ei = ((ratio_ei_ei - 1) - logratio_ei_ei).mean()
                    clipfracs_ei_ei += [((ratio_ei_ei - 1.0).abs() > args.clip_coef).float().mean().item()]

                    old_approx_kl_ei_e = (-logratio_ei_e).mean()
                    approx_kl_ei_e = ((ratio_ei_e - 1) - logratio_ei_e).mean()
                    clipfracs_ei_e += [((ratio_ei_e - 1.0).abs() > args.clip_coef).float().mean().item()]

                    old_approx_kl_e_e = (-logratio_e_e).mean()
                    approx_kl_e_e = ((ratio_e_e - 1) - logratio_e_e).mean()
                    clipfracs_e_e += [((ratio_e_e - 1.0).abs() > args.clip_coef).float().mean().item()]

                    old_approx_kl_e_ei = (-logratio_e_ei).mean()
                    approx_kl_e_ei = ((ratio_e_ei - 1) - logratio_e_ei).mean()
                    clipfracs_e_ei += [((ratio_e_ei - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_ei_eipo_advantages = b_ei_eipo_advantages[mb_inds]
                mb_ei_ppo_advantages = b_ei_ppo_advantages[mb_inds]
                mb_e_eipo_advantages = b_e_eipo_advantages[mb_inds]
                mb_e_ppo_advantages = b_e_ppo_advantages[mb_inds]
                if args.norm_adv:
                    mb_ei_eipo_advantages = (mb_ei_eipo_advantages - mb_ei_eipo_advantages.mean()) / (
                                mb_ei_eipo_advantages.std() + 1e-8)
                    mb_ei_ppo_advantages = (mb_ei_ppo_advantages - mb_ei_ppo_advantages.mean()) / (
                                mb_ei_ppo_advantages.std() + 1e-8)
                    mb_e_eipo_advantages = (mb_e_eipo_advantages - mb_e_eipo_advantages.mean()) / (
                                mb_e_eipo_advantages.std() + 1e-8)
                    mb_e_ppo_advantages = (mb_e_ppo_advantages - mb_e_ppo_advantages.mean()) / (
                                mb_e_ppo_advantages.std() + 1e-8)

                # Policy loss
                if agent.is_rollout_by_pi_e():  # Max stage
                    pg_loss1_ei_eipo = -mb_ei_eipo_advantages * ratio_ei_e
                    pg_loss2_ei_eipo = -mb_ei_eipo_advantages * torch.clamp(ratio_ei_e, 1 - args.clip_coef,
                                                                            1 + args.clip_coef)
                    pg_loss_ei_eipo = torch.max(pg_loss1_ei_eipo, pg_loss2_ei_eipo).mean()

                    pg_loss1_e_ppo = -mb_e_ppo_advantages * ratio_e_e
                    pg_loss2_e_ppo = -mb_e_ppo_advantages * torch.clamp(ratio_e_e, 1 - args.clip_coef,
                                                                        1 + args.clip_coef)
                    pg_loss_e_ppo = torch.max(pg_loss1_e_ppo, pg_loss2_e_ppo).mean()
                    pg_loss = pg_loss_ei_eipo + pg_loss_e_ppo
                    alpha_derivative = mb_e_ppo_advantages.mean().detach().cpu().item()
                    alpha_derivatives.append(alpha_derivative)
                    # For logging
                    pg_ei_loss = pg_loss_ei_eipo
                    pg_e_loss = pg_loss_e_ppo
                else:  # Min stage
                    pg_loss1_e_eipo = -mb_e_eipo_advantages * ratio_e_ei
                    pg_loss2_e_eipo = -mb_e_eipo_advantages * torch.clamp(ratio_e_ei, 1 - args.clip_coef,
                                                                          1 + args.clip_coef)
                    pg_loss_e_eipo = torch.max(pg_loss1_e_eipo, pg_loss2_e_eipo).mean()

                    pg_loss1_ei_ppo = -mb_ei_ppo_advantages * ratio_ei_ei
                    pg_loss2_ei_ppo = -mb_ei_ppo_advantages * torch.clamp(ratio_ei_ei, 1 - args.clip_coef,
                                                                          1 + args.clip_coef)
                    pg_loss_ei_ppo = torch.max(pg_loss1_ei_ppo, pg_loss2_ei_ppo).mean()
                    pg_loss = pg_loss_e_eipo + pg_loss_ei_ppo
                    # For logging
                    pg_ei_loss = pg_loss_ei_ppo
                    pg_e_loss = pg_loss_e_eipo

                # Value loss
                new_ei_ext_values, new_ei_int_values = new_ei_ext_values.view(-1), new_ei_int_values.view(-1)
                new_e_ext_values = new_e_ext_values.view(-1)
                if args.clip_vloss:
                    ei_ext_v_loss_unclipped = (new_ei_ext_values - b_ei_ext_returns[mb_inds]) ** 2
                    ei_ext_v_clipped = b_ei_ext_values[mb_inds] + torch.clamp(
                        new_ei_ext_values - b_ei_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ei_ext_v_loss_clipped = (ei_ext_v_clipped - b_ei_ext_returns[mb_inds]) ** 2
                    ei_ext_v_loss_max = torch.max(ei_ext_v_loss_unclipped, ei_ext_v_loss_clipped)
                    ei_ext_v_loss = 0.5 * ei_ext_v_loss_max.mean()

                    e_ext_v_loss_unclipped = (new_e_ext_values - b_e_ext_returns[mb_inds]) ** 2
                    e_ext_v_clipped = b_e_ext_values[mb_inds] + torch.clamp(
                        new_e_ext_values - b_e_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    e_ext_v_loss_clipped = (e_ext_v_clipped - b_e_ext_returns[mb_inds]) ** 2
                    e_ext_v_loss_max = torch.max(e_ext_v_loss_unclipped, e_ext_v_loss_clipped)
                    e_ext_v_loss = 0.5 * e_ext_v_loss_max.mean()
                else:
                    ei_ext_v_loss = 0.5 * ((new_ei_ext_values - b_ei_ext_returns[mb_inds]) ** 2).mean()
                    e_ext_v_loss = 0.5 * ((new_e_ext_values - b_e_ext_returns[mb_inds]) ** 2).mean()

                ei_int_v_loss = 0.5 * ((new_ei_int_values - b_ei_int_returns[mb_inds]) ** 2).mean()
                ei_v_loss = ei_ext_v_loss + ei_int_v_loss
                e_v_loss = e_ext_v_loss
                v_loss = ei_v_loss + e_v_loss

                entropy_ei = entropy_ei.mean()
                entropy_e = entropy_e.mean()
                entropy_loss = entropy_ei + entropy_e

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()

            if args.target_kl is not None:
                if agent.is_rollout_by_pi_e():
                    if approx_kl_e_e > args.target_kl or approx_kl_ei_e > args.target_kl:
                        break
                else:
                    if approx_kl_ei_ei > args.target_kl or approx_kl_e_ei > args.target_kl:
                        break

        # Check if we need to switch the policy
        old_is_rollout_by_pi_e = agent.is_rollout_by_pi_e()  # True: max stage, False: min stage
        agent.maybe_switch_rollout_pi(old_max_objective_value, ei_advantages.mean(),
                                      old_min_objective_value, e_advantages.mean())
        new_is_rollout_by_pi_e = agent.is_rollout_by_pi_e()  # True: max stage, False: min stage

        # Update alpha (only after max stage)
        if old_is_rollout_by_pi_e and (not new_is_rollout_by_pi_e):
            agent.alpha = agent.alpha - args.alpha_lr * np.clip(np.mean(alpha_derivatives), -args.alpha_g_clip,
                                                                args.alpha_g_clip)
            agent.alpha = np.clip(agent.alpha, -args.alpha_clip, args.alpha_clip)

        old_max_objective_value = ei_advantages.mean().item()
        old_min_objective_value = e_advantages.mean().item()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        writer.add_scalar("charts/alpha", agent.alpha, global_step)
        writer.add_scalar("charts/max_value_diff", (ei_advantages.mean().item() - old_max_objective_value), global_step)
        writer.add_scalar("charts/min_value_diff", (e_advantages.mean().item() - old_min_objective_value), global_step)
        writer.add_scalar("charts/rollout_by_pi_e", old_is_rollout_by_pi_e, global_step)

        writer.add_scalar("losses/ei_value_loss", ei_v_loss.item(), global_step)
        writer.add_scalar("losses/ei_policy_loss", pg_ei_loss.item(), global_step)
        writer.add_scalar("losses/ei_entropy", entropy_ei.item(), global_step)

        writer.add_scalar("losses/e_value_loss", e_v_loss.item(), global_step)
        writer.add_scalar("losses/e_policy_loss", pg_e_loss.item(), global_step)
        writer.add_scalar("losses/e_entropy", entropy_e.item(), global_step)

        writer.add_scalar("losses/ei_ei_old_approx_kl", old_approx_kl_ei_ei.item(), global_step)
        writer.add_scalar("losses/ei_ei_approx_kl", approx_kl_ei_ei.item(), global_step)
        writer.add_scalar("losses/ei_e_old_approx_kl", old_approx_kl_ei_e.item(), global_step)
        writer.add_scalar("losses/ei_e_approx_kl", approx_kl_ei_e.item(), global_step)
        writer.add_scalar("losses/e_ei_old_approx_kl", old_approx_kl_ei_ei.item(), global_step)
        writer.add_scalar("losses/e_ei_approx_kl", approx_kl_ei_ei.item(), global_step)
        writer.add_scalar("losses/e_e_old_approx_kl", old_approx_kl_ei_e.item(), global_step)
        writer.add_scalar("losses/e_e_approx_kl", approx_kl_ei_e.item(), global_step)

        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
