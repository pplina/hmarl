import random, math

from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

#import rlearn.attacker as attacker
#import rlearn.defender as defender
#import rlearn.network as network
#from rlearn.config import DDQN_Config


import attacker as attacker
import defender as defender
import network as network
from config import DDQN_Config

from torch.utils.tensorboard import SummaryWriter

"""
Replay Buffer WITHOUT prioritizes experience replay
"""


def Variable(USE_CUDA, *args, **kwargs):
    return (
        autograd.Variable(*args, **kwargs).cuda()
        if USE_CUDA
        else autograd.Variable(*args, **kwargs)
    )


def gamma_by_frame(step, config):
    gamma = (0.99 - 0.01) / (config.num_steps - 1000) * (step - config.num_steps + 1000) + 0.99
    if step > config.num_steps - 1000:
        gamma = 0.99
    elif step <= 1000:
        gamma = 0.1
    return gamma


def epsilon_by_frame(step, config):
    return config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(
        -1.0 * step / config.epsilon_decay
    )


def beta_by_frame(step, config):
    return min(1.0, config.beta_start + step * (1.0 - config.beta_start) / config.beta_frames)


class ReplayBuffer(object):
    # set length of ReplayBuffer
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Push new object into ReplayBuffer
    def push(self, state, action, reward, next_state, terminated):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, terminated))

    # Sample #batchsize elements out of the stored Experience
    def sample(self, batchsize):
        state, action, reward, next_state, terminated = zip(*random.sample(self.buffer, batchsize))
        return np.concatenate(state), action, reward, np.concatenate(next_state), terminated

    # Return length of Buffer
    def __len__(self):
        return len(self.buffer)


# Function that implements the 90% chance of the defender to identify infected nodes
def detectState(state, topology):
    detectedState = state
    for i in range(len(state[: len(topology)])):
        if state[i] == 1 and random.random() > 0.9:
            detectedState[i] = 0
    return detectedState


"""
Replay Buffer WITH prioritizes experience replay
"""


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batchsize, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios**self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batchsize, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


"""
Implementation of the DQN-Network
"""


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        # Specs of the DQN-Network
        self.layers = nn.Sequential(
            #nn.Linear(num_inputs, 128),
            #nn.ReLU(),
            #nn.Linear(128, 128),
            #nn.ReLU(),
            #nn.Linear(128, 128),
            #nn.ReLU(),
            #nn.Linear(128, num_actions),
            nn.Linear(num_inputs, num_inputs),
            nn.ReLU(),
            nn.Linear(num_inputs, num_inputs),
			nn.ReLU(),
            nn.Linear(num_inputs, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    # Depending on Epsilon, return random action or action according to Q-Values
    def act(self, state, epsilon, actionSpaceLen, USE_CUDA):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(USE_CUDA, torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
            #print("Action %s" % action)
        else:
            action = random.randrange(actionSpaceLen)
            action = torch.tensor(action)
            #print("Rand action %s" % action)
        return action


# Function to update the Target-Network while training as DDQN
def update_target(currentmodel, targetmodel):
    targetmodel.load_state_dict(currentmodel.state_dict())


# Function to calculate the Loss of the current Network
#def compute_td_loss(batchsize, currentmodel, targetmodel, optimizer, scheduler, beta, gamma, replay_buffer, USE_CUDA, step):
def compute_td_loss(batchsize, currentmodel, targetmodel, optimizer, beta, gamma, replay_buffer, USE_CUDA, step):

    state, action, reward, nextstate, done, indices, weights = replay_buffer.sample(
        batchsize, beta
    )
    # Extract experience out of the experience replay
    state = Variable(USE_CUDA, torch.FloatTensor(np.float32(state)))
    nextstate = Variable(USE_CUDA, torch.FloatTensor(np.float32(nextstate)))
    action = Variable(USE_CUDA, torch.LongTensor(action))
    reward = Variable(USE_CUDA, torch.FloatTensor(reward))
    done = Variable(USE_CUDA, torch.FloatTensor(done))
    weights = Variable(USE_CUDA, torch.FloatTensor(weights))
    # Get Q-Values of current and targetmodel
    q_values = currentmodel(state)
    next_q_values = currentmodel(nextstate)
    next_q_state_values = targetmodel(nextstate)
    # get q-Value, next q-Value and expected Q-Value
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(
        1, torch.max(next_q_values, 1)[1].unsqueeze(1)
    ).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # Calculate Loss
    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()
    # Optimize the Network
    optimizer.zero_grad()
    loss.backward()
    # Recalculate Priorities of Replay Buffer
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    #print("Step %d: Adam lr=%.4f" % (step, optimizer.param_groups[0]["lr"]))
    #before_lr = optimizer.param_groups[0]["lr"]
    #scheduler.step()
    #after_lr = optimizer.param_groups[0]["lr"]
    #print("Step %d: Adam lr %.4f -> %.4f" % (step, before_lr, after_lr))

    return loss


"""
DQNN-Training-Routine
"""


def trainDDQN(
    currentmodel,
    targetmodel,
    net,
    netgraph,
    critserver,
    optserver,
    actionSpace,
    topology,
    nwstate,
    mode,
    optimizer,
    #scheduler,
    attmode,
    replay_buffer,
):
    USE_CUDA = torch.cuda.is_available()
    # Initialize Training
    StartOptserver = optserver
    StartCritserver = critserver
    losses = []
    allrewards = []
    episodereward = 0
    terminated = 0
    allActions = []
    actionsPerEpisode = []
    config = DDQN_Config.parse_config()
    writer1 = SummaryWriter()

    for step in range(1, config.num_steps + 1):
        #state = network.getVectorFromState(nwstate)
        state = network.getVectorFromState2(nwstate, critserver, netgraph)
        epsilon = epsilon_by_frame(step, config)
        # get Action depending on Q-Values and Epsilon
        # action = currentmodel.act(detectState(state,topology), epsilon)
        action = currentmodel.act(state, epsilon, len(actionSpace), USE_CUDA)
        gamma = gamma_by_frame(step, config)
        # perform chosen action
        nwstate, netgraph, pFlag, critserver, optserver = defender.getAction(
            net, netgraph, critserver, optserver, action, actionSpace, topology, nwstate, mode
        )
        # perform Attack-Move
        nwstate = attacker.attack(net, netgraph, nwstate, critserver, mode, attmode)
        #nextstate = network.getVectorFromState(nwstate)
        nextstate = network.getVectorFromState2(nwstate, critserver, netgraph)
        # calculate Reward
        reward, terminated, reachable_nodes = network.getReward(
            critserver, optserver, topology, nwstate, pFlag, netgraph
        )

        actionsPerEpisode.append([state, action + 1, reward])
        # Push experience into Replay Buffer
        replay_buffer.push(state, action, reward, nextstate, terminated)

        episodereward += reward
        # Check if terminal State was reachd
        if terminated == 1:
            # reset Network and Store generated Values
            nwstate, netgraph = network.resetNetwork(topology, net, netgraph, mode)
            nwstate = attacker.attack(net, netgraph, nwstate, critserver, mode, attmode)
            allActions.append(actionsPerEpisode)
            actionsPerEpisode = []
            allrewards.append(episodereward)
            terminated = 0
            episodereward = 0
            critserver = StartCritserver
            optserver = StartOptserver
        # Start Optimizing after len(batchsize) Steps were performed
        if len(replay_buffer) > config.batchsize:
            beta = beta_by_frame(step, config)
            loss = compute_td_loss(
                #config.batchsize, currentmodel, targetmodel, optimizer, scheduler, beta, gamma, replay_buffer, USE_CUDA, step
                config.batchsize, currentmodel, targetmodel, optimizer, beta, gamma, replay_buffer, USE_CUDA, step
            )
            writer1.add_scalar("Loss/train", loss.data, step)
            losses.append(loss.data)

        #if step % 200 == 0:
        #    print("Current Step:", step)
        # Update the Target model every 100 Steps
        if step % 100 == 0:
            print("Current Step %d: Update targetmodel" % step)
            update_target(currentmodel, targetmodel)
    writer1.flush()
    writer1.close()
    return losses, allrewards, targetmodel


"""
Routine to evaluate the final DQN-Network and generate Plots
"""


def evalDDQN(
    currentmodel,
    net,
    netgraph,
    critserver,
    optserver,
    actionSpace,
    topology,
    nwstate,
    mode,
    pos,
    path2plot,
    attmode,
):
    USE_CUDA = torch.cuda.is_available()
    episodereward = 0
    terminated = 0
    states = []
    rewards = []
    actions = []
    i = 0
    action = -1
    # path = path2plot + str(i) + ".png"
    # Plot initial configuration
    # print(nwstate)
    # network.drawGraph(netgraph,pos,nwstate,path,action,topology,actionSpace)
    i = 1
    # Generate Actions in one Iteration (Until terminated State is reached once)
    while terminated == 0 and i < 25:
        #state = network.getVectorFromState(nwstate)
        state = network.getVectorFromState2(nwstate, critserver, netgraph)
        # Get Action according to Q-Values (no epsilon-Randomness)
        action = currentmodel.act(state, 0, len(actionSpace), USE_CUDA)
        #print(actionSpace)
        #print(len(actionSpace))
        #print(action)

        # perform Action and Attack
        nwstate, netgraph, pFlag, critserver, optserver = defender.getAction(
            net, netgraph, critserver, optserver, action, actionSpace, topology, nwstate, mode
        )
        if i < 9:
            path = path2plot + "0" + str(i) + "a.png"
        else:
            path = path2plot + str(i) + "a.png"
        # Plot current configuration
        network.drawGraph(netgraph, pos, nwstate, path, action, topology, actionSpace)

        nwstate = attacker.attack(net, netgraph, nwstate, critserver, mode, attmode)
        reward, terminated, reachable_nodes = network.getReward(
            critserver, optserver, topology, nwstate, pFlag, netgraph
        )
        states.append(state)
        rewards.append(reward)
        actions.append(action + 1)
        episodereward += reward
        if i < 9:
            path = path2plot + "0" + str(i) + "b.png"
        else:
            path = path2plot + str(i) + "b.png"
        # Plot current configuration
        network.drawGraph(netgraph, pos, nwstate, path, -2, topology, actionSpace)
        i = i + 1

    return states, rewards, actions, episodereward, reachable_nodes


"""
Routine to evaluate the final DQN-Network 
"""


def evalDDQN2(
    currentmodel,
    net,
    netgraph,
    critserver,
    optserver,
    actionSpace,
    topology,
    nwstate,
    mode,
    pos,
    path2plot,
    attmode,
):
    USE_CUDA = torch.cuda.is_available()
    episodereward = 0
    terminated = 0
    states = []
    rewards = []
    actions = []
    i = 0
    action = -1
    i = 1
    # Generate Actions in one Iteration (Until terminated State is reached once)
    while terminated == 0 and i < 25:
        #state = network.getVectorFromState(nwstate)
        state = network.getVectorFromState2(nwstate, critserver, netgraph)
        # Get Action according to Q-Values (no epsilon-Randomness)
        action = currentmodel.act(state, 0, len(actionSpace), USE_CUDA)

        # perform Action and Attack
        nwstate, netgraph, pFlag, critserver, optserver = defender.getAction(
            net, netgraph, critserver, optserver, action, actionSpace, topology, nwstate, mode
        )
        nwstate = attacker.attack(net, netgraph, nwstate, critserver, mode, attmode)
        #nextstate = network.getVectorFromState(nwstate)
        nextstate = network.getVectorFromState2(nwstate, critserver, netgraph)
        reward, terminated, reachable_nodes = network.getReward(
            critserver, optserver, topology, nwstate, pFlag, netgraph
        )
        states.append(state)
        rewards.append(reward)
        actions.append(action + 1)
        episodereward += reward
        # if i < 9:
        #    path = path2plot + "0" + str(i) +".png"
        # else:
        #    path = path2plot + str(i) +".png"
        # Plot current configuration
        # network.drawGraph(netgraph,pos,nwstate,path,action,topology,actionSpace)
        i = i + 1

    return states, rewards, actions, episodereward, reachable_nodes
