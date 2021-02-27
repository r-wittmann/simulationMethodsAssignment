#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 2021

@author: r-wittmann
@author: Konstantin0694
"""

import numpy as np
import matplotlib.pyplot as plt
from bandit_experiment import perform_experiments

# =========================
# reproduce Posen and Levinthal Paper
# =========================

print("Performing a replication of Posen and Levinthal 2012\n")


N_bandits = 10
N_experiments = 100
N_episodes = 500

p_l_taus = [0.02, 0.25, 0.5, 0.75, 1]

p_l_results = np.matrix([
        p_l_taus,
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ])

for i in range(len(p_l_taus)):
    (reward_m, tau_m, knowledge_m, exploration_m) = perform_experiments(N_bandits=N_bandits, N_experiments=N_experiments, N_episodes=N_episodes, tau=p_l_taus[i]/N_bandits)
    
    exploration_prob= np.count_nonzero(exploration_m, axis=1)/N_episodes
    p_l_results[1,i] = np.mean(exploration_prob)
    
    rewards_cum=np.cumsum(reward_m,axis=1)
    p_l_results[2,i] = np.mean(rewards_cum[:,-1],axis=0)
    
    knowledge_avg_over_time=np.mean(knowledge_m, axis=0)
    p_l_results[3,i] = knowledge_avg_over_time[-1]

print("Here are the results for the replication of Posen and Levinthal 2012:\n")
print("Exploration: {}".format(p_l_results[1,:].round(5)))
print("Performance: {}".format(p_l_results[2,:].round(5)))
print("Knowledge:   {}".format(p_l_results[3,:].round(5)))






# =========================
# old code, don't run!!!
# =========================


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
