import os
import pickle
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical

from nsfr.common import get_nsfr_model
from nsfr.utils.common import load_module


class NSFR_ActorCritic(nn.Module):
    def __init__(self, args, device, rng=None):
        super(NSFR_ActorCritic, self).__init__()
        self.device =device
        self.rng = random.Random() if rng is None else rng
        self.args = args
        self.actor = get_nsfr_model(self.args, device=device, train=True)
        self.prednames = self.get_prednames()

        env_name = self.args.env
        mlp_module_path = f"../envs/{env_name}/mlp.py"
        module = load_module(mlp_module_path)
        self.critic = module.MLP(out_size=1, logic=True)

        self.num_actions = len(self.prednames)
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device=device))
        self.upprior = Categorical(
            torch.tensor([0.9] + [0.1 / (self.num_actions-1) for _ in range(self.num_actions-1)], device=device))

    def forward(self):
        raise NotImplementedError

    def act(self, logic_state, epsilon=0.0):
        action_probs = self.actor(logic_state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
            action = dist.sample()
        else:
            dist = Categorical(action_probs)
            action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(self.device)
            if torch.numel(action) > 1:
                action = action[0]
        # action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, neural_state, logic_state, action):
        action_probs = self.actor(logic_state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(neural_state)

        return action_logprobs, state_values, dist_entropy

    def get_prednames(self):
        return self.actor.get_prednames()


class LogicPPO:
    def __init__(self, lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args, device=None):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.args = args
        self.policy = NSFR_ActorCritic(self.args, device=device).to(device)
        self.optimizer = optimizer([
            {'params': self.policy.actor.get_params(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = NSFR_ActorCritic(self.args, device=device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.prednames = self.get_prednames()

        # Different games use different action system, need to map it to the correct action.
        # action of logic game means a String, need to map string to the correct action,
        env_name = self.args.env
        action_mapping_module_path = f"../envs/{env_name}/actions.py"
        module = load_module(action_mapping_module_path)
        self.map_action = module.map_action

    def extract_states(self, raw_state):
        """Extracts the logic and the neural state representation of the given raw state.
        The extraction depends on the environment."""
        env_name = self.args.env
        state_extraction_module_path = f"../envs/{env_name}/state_extraction.py"
        module = load_module(state_extraction_module_path)

        logic_state = module.extract_logic_state(raw_state)  # TODO: enable custom args
        logic_state = torch.tensor(logic_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        neural_state = module.extract_neural_state(raw_state)  # TODO: enable custom args
        neural_state = torch.tensor(neural_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        return logic_state, neural_state

    def select_action(self, state, epsilon=0.0):
        logic_state, neural_state = self.extract_states(state)

        # select random action with epsilon probability and policy probiability with 1-epsilon
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            # import ipdb; ipdb.set_trace()
            action, action_logprob = self.policy_old.act(logic_state, epsilon=epsilon)
        
        self.buffer.neural_states.append(neural_state)
        self.buffer.logic_states.append(logic_state)
        action = torch.squeeze(action)
        self.buffer.actions.append(action)
        action_logprob = torch.squeeze(action_logprob)
        self.buffer.logprobs.append(action_logprob)

        action = self.map_action(action.item(), self.args.alg, self.prednames)

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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor

        old_neural_states = torch.squeeze(torch.stack(self.buffer.neural_states, dim=0)).detach().to(self.device)
        old_logic_states = torch.squeeze(torch.stack(self.buffer.logic_states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_neural_states, old_logic_states,
                                                                        old_actions)

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

    def save(self, checkpoint_path, directory, step_list, reward_list, weight_list):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        with open(directory + '/' + "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)

    def load(self, directory):
        # only for recover form crash
        model_name = input('Enter file name: ')
        model_file = os.path.join(directory, model_name)
        self.policy_old.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        with open(directory + '/' + "data.pkl", "rb") as f:
            step_list = pickle.load(f)
            reward_list = pickle.load(f)
            weight_list = pickle.load(f)
        return step_list, reward_list, weight_list

    def get_predictions(self, state):
        self.prediction = state
        return self.prediction

    def get_weights(self):
        return self.policy.actor.get_params()

    def get_prednames(self):
        return self.policy.actor.get_prednames()


class LogicPlayer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.prednames = model.get_prednames()

        env_name = self.args.env

        state_extraction_module_path = f"../envs/{env_name}/state_extraction.py"
        module = load_module(state_extraction_module_path)
        self.extract_logic_state = module.extract_logic_state

        action_mapping_module_path = f"../envs/{env_name}/actions.py"
        module = load_module(action_mapping_module_path)
        self.pred2action = module.pred2action

    def act(self, state):
        extracted_state = self.extract_logic_state(state, variant=self.args.env)  # TODO: rename arg.env
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()
        explaining = self.prednames[prediction]
        action = self.pred2action(prediction, self.prednames)
        return action, explaining

    def get_probs(self):
        probs = self.model.get_probs()
        return probs

    def get_explaining(self):
        explaining = 0
        return explaining

    def get_state(self, raw_state):
        env_name = self.args.env
        state_extraction_module_path = f"../envs/{env_name}/state_extraction.py"
        module = load_module(state_extraction_module_path)

        logic_state = module.extract_logic_state(raw_state)  # TODO: enable custom args
        logic_state = torch.tensor(logic_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        logic_state = logic_state.tolist()
        result = []
        for list in logic_state:
            obj_state = [round(num, 2) for num in list]
            result.append(obj_state)
        return result


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.neural_states = []
        self.logic_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.neural_states[:]
        del self.logic_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.predictions[:]
