#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:38:47 2021

@author: rainer-wittmann
"""


import numpy as np
import matplotlib.pyplot as plt
# =========================
# Methods
# =========================

# assign random probabilities for all bandits
def assign_bandit_probabilities(N_bandits):
    return np.random.beta(2.0, 2.0, N_bandits)

# returns a reward of 1 with the real probability of the respective bandit
def get_success(action, bandit_probs):
    return 1 if (np.random.random() < bandit_probs[action]) else 0

# returns 1 or -1 depending on the success of the action
def get_reward(success):
    return 1 if (success == 1) else -1

# updates the believes of the agend based on the payout of the current episode
def update_Q(k, Q, action, reward):
    k[action] += 1  # update action counter k -> k+1
    Q[action] += (1./(k[action]+1)) * (reward - Q[action]) # calculate new average payoff
    return (k, Q)

# calculates the knowledge as introduced in the P and L paper (the distance from the reality)
def calculate_knowledge(current_beliefs, bandit_probs):
    return 1-sum((current_beliefs - bandit_probs)**2)

# returns a 1 if the current action differs from the previous (exploration) and a 0 if they are the same (exploitation)
def calculate_exploration(action_history, action):
    if (len(action_history) == 0):
        return 1
    else:
        return 0 if (action_history[-1] == action) else 1

# selects one bandit depending on the current payout beliefs with a softmax function
def choose_softmax(Q, tau, N_bandits):
    if tau > 0:
        e = np.exp(np.asarray(Q) / tau)
        dist = e / np.sum(e)
        roulette = np.random.random()
        cum_prob = 0
        for i in range(N_bandits):
            cum_prob = cum_prob + dist[i]
            if cum_prob >= roulette:
                action = i
                break  # stops computing probability of action
    else:  # Means that either first did not work or tau = 0
        action = np.random.choice(np.flatnonzero(Q == Q.max()))
    return action

def update_tau(tau, tau_strategy): # needs additional parameters in the future
    if tau_strategy == "stable":
        return tau
    elif tau_strategy == "performance":
        # not implemented yet
        return tau
    else:
        # defaults to returning tau
        return tau

# =========================
# Define an experiment
# =========================
    
def experiment(N_bandits, N_episodes, tau, shock_prob, tau_strategy):
    # create empty arrays to append elements to
    action_history = np.array([])
    reward_history = np.array([])
    tau_history = np.array([])
    knowledge_history = np.array([])
    exploration_history = np.array([])
    
    k = np.zeros(N_bandits, dtype=np.int)  # number of times an action was performed per bandit
    Q = np.full(N_bandits, 0.5)  # set payout beliefs to initially 0.5
    
    # assign random bandit payout probabilities
    bandit_probs = assign_bandit_probabilities(N_bandits)
    
    for episode in range(N_episodes):
        # check for environmental shock
        if np.random.random() < shock_prob:
            bandit_probs = assign_bandit_probabilities(N_bandits)
        
        # calculate and update tau
        tau = update_tau(tau, tau_strategy)
        
        # Choose action from agent (from current Q estimate)
        action = choose_softmax(Q, tau, N_bandits)
        # calculate success of the action
        success = get_success(action, bandit_probs)
        # Pick up reward from bandit for chosen action
        reward = get_reward(success)
    
        # update parameters
        exploration = calculate_exploration(action_history,action)
        (k, Q) = update_Q(k, Q, action, success)
        
         # calculate the current knowledge
        knowledge = calculate_knowledge(Q, bandit_probs)
        
        # Append values to histories
        action_history = np.append(action_history, action)
        reward_history = np.append(reward_history, reward)
        tau_history= np.append(tau_history, tau)
        knowledge_history = np.append(knowledge_history, knowledge)
        exploration_history = np.append(exploration_history, exploration)
    
    # return all histories
    return (action_history, reward_history, tau_history, knowledge_history, exploration_history)

def perform_experiments(N_bandits, N_experiments, N_episodes, tau, shock_prob=0, tau_strategy="stable"):
    print("Running multi-armed bandits with {} bandits in {} experiments with {} episodes.".format(N_bandits, N_experiments, N_episodes))
    print("Environment shock probability is set to {}, tau to {} and tau strategy is \"{}\".".format(shock_prob, tau, tau_strategy))
    print("")
    
    # create empty matrices to store our history in
    reward_m      = np.zeros((N_experiments, N_episodes))
    tau_m         = np.zeros((N_experiments, N_episodes))
    knowledge_m   = np.zeros((N_experiments, N_episodes))
    exploration_m = np.zeros((N_experiments, N_episodes)) 
    
    for i in range(N_experiments):
        #perform experiment
        (action_h, reward_h, tau_h, knowledge_h, exploration_h) = experiment(N_bandits, N_episodes, tau, shock_prob, tau_strategy)
        
        # print to know at which experiment we currently are
        if (i + 1) % (N_experiments / 10) == 0:
            print("[Experiment {}/{}]".format(i + 1, N_experiments))
            print("")
        
        # append the history arrays to the matrices
        reward_m[i,:] = reward_h
        tau_m[i,:] = tau_h
        knowledge_m[i,:] = knowledge_h
        exploration_m[i,:] = exploration_h
    
    return (reward_m, tau_m, knowledge_m, exploration_m)