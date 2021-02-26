#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 2021

@author: r-wittmann
@author: Konstantin0694
"""
# Knowledge berechnet sich noch nicht korrekt!

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Settings
# =========================

N_bandits = 10
N_experiments = 5000 # number of experiments
N_episodes =500 # number of episodes per experiment
tau = 0.5/N_bandits

# =========================
# Methods
# =========================

# assign random probabilities for all bandits
# this happens for each experiment
def assign_bandit_probabilities():
    bandit_probs = np.zeros(N_bandits)
    for i in range(N_bandits):
        bandit_probs[i] = np.random.beta(a=2.0, b=2.0)
    return bandit_probs

# returns a reward of 1 with the real probability of the respective bandit
def get_success(action, bandit_probs):
    rand = np.random.random()  # [0.0,1.0)
    success = 1 if (rand < bandit_probs[action]) else 0
    return success

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
def choose_softmax(Q, tau):
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
                # selection as soon as cumulative probability exceeds roulette
    else:  # Means that either first did not work or tau = 0
        action = np.random.choice(np.flatnonzero(Q == Q.max()))
    return action


# =========================
# Define an experiment
# =========================
    
def experiment_fix_tau(N_episodes): #
    # create empty arrays to append elements to
    action_history = np.array([])
    reward_history = np.array([])
    tau_history = np.array([])
    knowledge_history = np.array([])
    exploration_history = np.array([])
    
    k = np.zeros(N_bandits, dtype=np.int)  # number of times an action was performed per bandit
    Q = np.full(N_bandits, 0.5)  # set payout beliefs to initially 0.5
    
    # assign random bandit payout probabilities
    bandit_probs = assign_bandit_probabilities()
    
    for episode in range(N_episodes):

        # calculate and update tau
        #tau = tau
        
        # Choose action from agent (from current Q estimate)
        action = choose_softmax(Q, tau)
        # calculate success of the action
        success = get_success(action, bandit_probs)
        # Pick up reward from bandit for chosen action
        reward = get_reward(success)
    
        # determin, if the action was exloration or exploitation
        exploration = calculate_exploration(action_history,action)
        # Update Q action-value estimates
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

def experiment_adv(N_episodes): #
    # create empty arrays to append elements to
    action_history = np.array([])
    reward_history = np.array([])
    tau_history = np.array([])
    knowledge_history = np.array([])
    exploration_history = np.array([])
    
    k = np.zeros(N_bandits, dtype=np.int)  # number of times an action was performed per bandit
    Q = np.full(N_bandits, 0.5)  # set payout beliefs to initially 0.5
    
    # assign random bandit payout probabilities
    bandit_probs = assign_bandit_probabilities()
    
    for episode in range(N_episodes):

        # calculate and update tau
        # tau = tau
        
        # Choose action from agent (from current Q estimate)
        action = choose_softmax(Q, tau)
        # calculate success of the action
        success = get_success(action, bandit_probs)
        # Pick up reward from bandit for chosen action
        reward = get_reward(success)
        # calculate the current knowledge
        knowledge = calculate_knowledge(Q, bandit_probs)
        # determin, if the action was exloration or exploitation
        exploration = calculate_exploration(action_history,action)
        # Update Q action-value estimates
        (k, Q) = update_Q(k, Q, action, success)
        
        # Append values to histories
        action_history = np.append(action_history, action)
        reward_history = np.append(reward_history, reward)
        tau_history= np.append(tau_history, tau)
        knowledge_history = np.append(knowledge_history, knowledge)
        exploration_history = np.append(exploration_history, exploration)
    
    # return all histories
    return (action_history, reward_history, tau_history, knowledge_history, exploration_history)

# =========================
# Loop
# =========================

print("Running multi-armed bandits with N_bandits = {} and agent tau = {} in {} experiments with {} episodes".format(N_bandits, tau, N_experiments, N_episodes))
print("")

# create empty matrices to store our history in
reward_history_matrix   = np.zeros((N_experiments, N_episodes))
tau_matrix              = np.zeros((N_experiments, N_episodes))
knowledge_matrix        = np.zeros((N_experiments, N_episodes))
exploration_matrix      = np.zeros((N_experiments, N_episodes)) 

for i in range(N_experiments):
    
    #perform experiment
    (action_history, reward_history, tau_history, knowledge_history, exploration_history) = experiment_adv(N_episodes)
    
    # print to know at which experiment we currently are
    if (i + 1) % (N_experiments / 10) == 0:
        print("[Experiment {}/{}]".format(i + 1, N_experiments))
        print("")
    
    # append the history arrays to the matrices
    reward_history_matrix[i,:] = reward_history
    tau_matrix[i,:] = tau_history
    knowledge_matrix[i,:] = knowledge_history
    exploration_matrix[i,:] = exploration_history
    

# Calculate exploration probability
exploration_prob= np.count_nonzero(exploration_matrix, axis=1)/N_episodes
exploration_percent_avg=np.mean(exploration_prob)

# Calculate accumulated asset stock (rewards_cum)
rewards_cum=np.cumsum(reward_history_matrix,axis=1)
rewards_cum_avg=np.mean(rewards_cum,axis=0)
rewards_end_avg=rewards_cum_avg[-1]

# Calculate average knowledge at the end of the final period
knowledge_avg_over_time=np.mean(knowledge_matrix, axis=0)
knowledge_end_avg=knowledge_avg_over_time[-1]

# print(knowledge_matrix)
# print(knowledge_cum)
# knowledge_cum_avg=np.mean(knowledge_cum,axis=0)
# print(knowledge_cum_avg)
# knowledge_end_avg=knowledge_cum_avg[-1]
# print(knowledge_end_avg)
#Calculate average tau for each period
tau_avg=np.mean(tau_matrix,axis=0)*N_bandits
tau_end_avg=tau_avg[-1]

#Print main insights
print('Exploration probability = '+str(exploration_percent_avg))
print('Performance = '+ str(rewards_end_avg))
print('Knowledge = ' + str(knowledge_end_avg))
print('Tau = ' + str(tau_end_avg))
# =========================
# Plot
# =========================
# Plot Cumulated Performance over time
plt.plot(range(N_episodes), rewards_cum_avg)
plt.xlabel('Periods')
plt.ylabel('Cumulated Performance')
plt.title('Cumulated Performance over time')
plt.figure("""first figure""")

# Plot Knowledge over time
plt.plot(range(N_episodes),knowledge_avg_over_time)
plt.xlabel('Periods')
plt.ylabel('Knowledge')
plt.title('Knowledge over time')
plt.figure("""second figure""")

#Plot tau over time
plt.plot(range(N_episodes),tau_avg)
plt.xlabel('Periods')
plt.ylabel('Tau')
plt.title('Tau over time')
plt.figure("""third figure""")


# =========================
# Replicate figure 1 of Posen, Levinthal 2012
# Calculate results of figure 1 for 5 different strategies

tau_strategy=[0.02,0.25,0.5,0.75,1]
strategies=len(tau_strategy)
N_bandits = 10
N_experiments = 5000 # number of experiments
N_episodes =500 

exploration_percent_avg  = np.array([])
rewards_end_avg          = np.array([])
knowledge_end_avg        = np.array([])

for s in range(strategies):
    
    reward_history_matrix   = np.zeros((N_experiments, N_episodes))
    tau_matrix              = np.zeros((N_experiments, N_episodes))
    knowledge_matrix        = np.zeros((N_experiments, N_episodes))
    exploration_matrix      = np.zeros((N_experiments, N_episodes)) 
    
    for i in range(N_experiments):
        tau=tau_strategy[s]/N_bandits
        #perform experiment
        (action_history, reward_history, tau_history, knowledge_history, exploration_history) = experiment_fix_tau(N_episodes)
        
        # print to know at which experiment we currently are
        if (i + 1) % (N_experiments / 10) == 0:
            print("[Experiment {}/{}]".format(i + 1, N_experiments))
            print("")
        
        # append the history arrays to the matrices
        reward_history_matrix[i,:] = reward_history
        tau_matrix[i,:] = tau_history
        knowledge_matrix[i,:] = knowledge_history
        exploration_matrix[i,:] = exploration_history

    exploration_prob= np.count_nonzero(exploration_matrix, axis=1)/N_episodes
    exploration_percent_avg=np.append(exploration_percent_avg, np.mean(exploration_prob))
    
    rewards_cum=np.cumsum(reward_history_matrix,axis=1)
    rewards_end_avg=np.append(rewards_end_avg, np.mean(rewards_cum[:,-1],axis=0))
    knowledge_avg_over_time=np.mean(knowledge_matrix, axis=0)
    knowledge_end_avg=np.append(knowledge_end_avg,knowledge_avg_over_time[-1])

print('Tau')
print(tau_strategy)
print('---')
print('Exploration probability')
print(exploration_percent_avg)
print('---')
print('Performance')
print(rewards_end_avg)
print('---')
print('Knowledge')
print(knowledge_end_avg)
print('---')
