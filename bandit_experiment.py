#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:38:47 2021

@author: r-wittmann
@author: Konstantin0694
"""

"""
This script contains the code for a repeted experiments of the multi-armed
bandit model.

First, all methods are defined that are later used in the experiment loop
itself, including softmax choosing method and the method to update ones 
believs.

One experiment is defined as a game of N_periods with certain pre-set 
characteristics (e.g. number of periods, tau and shock probability) 

This script is designed to be called from either the console or other files, 
methods and parameters are commented below.
"""

import numpy as np

# =========================
# Methods
# =========================

# returns a list of payout probabilities for each bandit. Probabilities are 
# drawn from a random beta distribution with a = b = 2
# parameters:
#    N_bandits: number of bandits, int
def assign_bandit_probabilities(N_bandits):
    return np.random.beta(2.0, 2.0, N_bandits)

# returns success (1) or failure (0) based on the real payout probability
# of the respective bandit
# parameters:
#    action: the previously chosen action, int between 0 and len(bandit_probs)
#    bandit_probs: list of payout probabilities for each bandit, list of floats
def get_success(action, bandit_probs):
    return 1 if (np.random.random() < bandit_probs[action]) else 0

# returns 1 or -1 depending on the success of the performed action
# parameters:
#    success: wheter the action was successful or not, 0 or 1
def get_reward(success):
    return 1 if (success == 1) else -1

# updates the believes of the agend based on the payout of the current period
# parameters:
#    k: number of actions previously performed, list of ints
#    Q: current believes of payout probabilities, list of floats
#    action: the previously chosen action, int
#    reward: the payout of the previous action, -1 or 1
def update_Q(k, Q, action, reward):
    k[action] += 1  # update action counter
    Q[action] += (1./(k[action]+1)) * (reward - Q[action]) # calculate new average payoff
    return (k, Q)

# calculates the knowledge as introduced in the Posen and Levinthal paper,
# which can be understood as the distance from the reality (sum of squared 
# errors)
# parameters:
#    Q: current believes of payout probabilities, list of floats
#    bandit_probs: list of payout probabilities for each bandit, list of floats
def calculate_knowledge(Q, bandit_probs):
    return 1-sum((Q - bandit_probs)**2)

# returns a 1 if the current action differs from the previous (exploration) and
# a 0 if they are the same (exploitation)
# this is later used to calculate the exploration probability
# parameters:
#    action_history: list of actions performed, list of ints
#    action: the previously chosen action, int
def calculate_exploration(action_history, action):
    if (len(action_history) == 0): # this is only true in the first period
        return 1
    else:
        return 0 if (action_history[-1] == action) else 1

# selects one bandit to operate, depending on the current payout beliefs and 
# tau with a softmax function
# parameters:
#    Q: current believes of payout probabilities, list of floats
#    tau: the tau parameter, float
#    N_bandits: number of bandits, int
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

# this method calculates the tau to be used in the choose softmax method
# tau-strategies available: "stable" and "accumulated resources"
# if "accumulated resources is selected, tau is recalculated based on the
# current resources
# parameters:
#    tau_0: the initial tau parameter, float
#    tau_strategy: the chosen strategy to update tau, string
#    N_periods: number of periods per experiment, int
#    reward_history: history of payouts from previous periods, list of -1 and 1
def update_tau(tau_0, tau_strategy, N_periods, reward_history):
    if tau_strategy == "stable":
        return tau_0
    elif tau_strategy == "accumulated resources":
        return tau_0+(1/(N_periods)**2)*(np.sum(reward_history)**2)
    else:
        # defaults to returning tau
        return tau_0

# =========================
# Define an experiment
# =========================

# an experiment is a repitition of actions for N_periods
# parameters:
#    N_bandits: the number of bandits, int
#    N_periods: number of periods per experiment, int
#    tau: the tau parameter, float
#    shock_prob: probability of a shock to the environment, float
#    tau_strategy: the chosen strategy to update tau, string
def experiment(N_bandits, N_periods, tau, shock_prob, tau_strategy):
    # create empty arrays to append elements to
    action_history = np.array([])
    reward_history = np.array([])
    tau_history = np.array([])
    knowledge_history = np.array([])
    exploration_history = np.array([])
    tau_0=tau
    
    k = np.zeros(N_bandits, dtype=np.int)  # number of times an action was performed per bandit
    Q = np.full(N_bandits, 0.5)  # set payout beliefs to 0.5 initially
    
    # assign random bandit payout probabilities
    bandit_probs = assign_bandit_probabilities(N_bandits)
    
    for period in range(N_periods):
        # check for environmental shock
        if np.random.random() < shock_prob and np.random.random()<0.5:
            bandit_probs = assign_bandit_probabilities(N_bandits)
        
        # calculate and update tau
        tau = update_tau(tau_0, tau_strategy, N_periods, reward_history)
        
        # Choose action from agent (based on current believes)
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


# an experiment is a repitition of actions for N_periods
# parameters:
#    N_bandits: the number of bandits, int
#    N_experiments: the number of experiments, int
#    N_periods: number of periods per experiment, int
#    tau: the tau parameter, float
#    shock_prob: probability of a shock to the environment, float, default = 0
#    tau_strategy: the chosen strategy to update tau, string, default = "stable"
def perform_experiments(N_bandits, N_experiments, N_periods, tau, shock_prob=0, tau_strategy="stable"):
    print("Running multi-armed bandits with {} bandits in {} experiments with {} periods.".format(N_bandits, N_experiments, N_periods))
    print("Environment shock probability is set to {}, tau to {} and tau strategy is \"{}\".".format(shock_prob, tau, tau_strategy))
    print("")
    
    # create empty matrices to store our history in
    reward_m      = np.zeros((N_experiments, N_periods))
    tau_m         = np.zeros((N_experiments, N_periods))
    knowledge_m   = np.zeros((N_experiments, N_periods))
    exploration_m = np.zeros((N_experiments, N_periods)) 
    
    for i in range(N_experiments):
        #perform experiment
        (action_h, reward_h, tau_h, knowledge_h, exploration_h) = experiment(N_bandits, N_periods, tau, shock_prob, tau_strategy)
        
        # print to know at which experiment we currently are
        if (i + 1) % (N_experiments / 10) == 0:
            print("[Experiment {}/{}]".format(i + 1, N_experiments))
            print("")
        
        # append the history arrays to the matrices
        reward_m[i,:] = reward_h
        tau_m[i,:] = tau_h
        knowledge_m[i,:] = knowledge_h
        exploration_m[i,:] = exploration_h
    
    # return matricies
    return (reward_m, tau_m, knowledge_m, exploration_m)