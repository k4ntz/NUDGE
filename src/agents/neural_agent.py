import os
import torch
import torch.nn as nn
import random
import pickle

from torch.distributions import Categorical
from .MLPController.mlpgetout import MLPGetout
from .MLPController.mlpthreefish import MLPThreefish
from .MLPController.mlploot import MLPLoot
from .utils_getout import extract_state, sample_to_model_input, collate, action_map_getout, \
    extract_neural_state_getout
from .utils_threefish import simplify_action_bf, action_map_threefish, extract_neural_state_threefish
from .utils_loot import simplify_action_loot, action_map_loot, extract_neural_state_loot

device = torch.device('cuda:0')


class ActorCritic(nn.Module):
    def __init__(self, args, rng=None):
        super(ActorCritic, self).__init__()

        self.rng = random.Random() if rng is None else rng
        self.args = args
        if self.args.m == 'getout':
            self.num_action = 3
            self.actor = MLPGetout(has_softmax=True)
            self.critic = MLPGetout(has_softmax=False, out_size=1)
        elif self.args.m == 'threefish':
            self.num_action = 5
            self.actor = MLPThreefish(has_softmax=True)
            self.critic = MLPThreefish(has_softmax=False, out_size=1)
        elif self.args.m == "loot":
            self.num_action = 5
            self.actor = MLPLoot(has_softmax=True)
            self.critic = MLPLoot(has_softmax=False, out_size=1)
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_action for _ in range(3)], device=device))

    def forward(self):
        raise NotImplementedError

    def act(self, state, epsilon=0.0):

        action_probs = self.actor(state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
        else:
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class NeuralPPO:
    def __init__(self, lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.args = args
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(self.args).to(device)
        self.optimizer = optimizer([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(self.args).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, epsilon=0.0):

        # extract state info for different games
        if self.args.m == 'getout':
            state = extract_neural_state_getout(state, self.args)
            # model_input = sample_to_model_input((extract_state(state), []))
            # model_input = collate([model_input])
            # state = model_input['state']
            # state = torch.cat([state['base'], state['entities']], dim=1)
        elif self.args.m == 'threefish':
            state = extract_neural_state_threefish(state, self.args)
            # state = state['positions'].reshape(-1)
            # state = torch.tensor(state.tolist()).to(device)
        elif self.args.m == 'loot':
            state = extract_neural_state_loot(state, self.args)
            # state = state['positions'].reshape(-1)
            # state = torch.tensor(state.tolist()).to(device)
        # select random action with epsilon probability and policy probiability with 1-epsilon
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state, epsilon=epsilon)

        self.buffer.states.append(state)
        action = torch.squeeze(action)
        self.buffer.actions.append(action)
        action_logprob = torch.squeeze(action_logprob)
        self.buffer.logprobs.append(action_logprob)

        if self.args.m == 'getout':
            action = action_map_getout(action.item(), self.args)
        elif self.args.m == 'threefish':
            action = action_map_threefish(action.item(), self.args)
        elif self.args.m == 'loot':
            action = action_map_loot(action.item(), self.args)

        return action

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # training does not converge if the entropy term is added ...
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)  # - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # wandb.log({"loss": loss})

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path, directory, step_list, reward_list):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        with open(directory + '/' + "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)

    def load(self, directory):
        # only for recover form crash
        model_name = input('Enter file name: ')
        model_file = os.path.join(directory, model_name)
        self.policy_old.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        with open(directory + '/' + "data.pkl", "rb") as f:
            step_list = pickle.load(f)
            reward_list = pickle.load(f)
        return step_list, reward_list

    def get_weights(self):
        return self.policy.actor.get_params()

class NeuralPlayer:
    def __init__(self, args, model=None):
        self.args = args
        self.model = model
        self.device = torch.device('cuda:0')

    def act(self, state):
        if self.args.m == 'getout':
            action = self.getout_actor(state)
        elif self.args.m == 'threefish':
            action = self.threefish_actor(state)
        elif self.args.m == 'loot':
            action = self.loot_actor(state)
        return action

    def getout_actor(self, getout):
        state = extract_neural_state_getout(getout, self.args)
        # model_input = sample_to_model_input((extract_state(getout), []))
        # model_input = collate([model_input])
        # state = model_input['state']
        # state = torch.cat([state['base'], state['entities']], dim=1)
        prediction = self.model(state)
        # action = coin_jump_actions_from_unified(torch.argmax(prediction).cpu().item() + 1)
        action = torch.argmax(prediction).cpu().item() + 1
        return action

    def threefish_actor(self, state):
        state = extract_neural_state_threefish(state, self.args)
        # state = state.reshape(-1)
        # state = state.tolist()
        predictions = self.model(state)
        action = torch.argmax(predictions)
        action = simplify_action_bf(action)
        return action

    def loot_actor(self, state):
        state = extract_neural_state_loot(state, self.args)
        # state = state.reshape(-1)
        # state = state.tolist()
        predictions = self.model(state)
        action = torch.argmax(predictions)
        action = simplify_action_loot(action)
        return action


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
