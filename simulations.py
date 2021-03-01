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
# Perform an experiment
# =========================

#Input parameters to perform experiment
N_bandits=10
N_experiments=100
N_episodes=500
tau_fix=1/N_bandits
tau_acc=0.5/N_bandits
shock_prob=0.01
# --------------------------------
#Perform experiment for FIXED TAU
#---------------------------------

(reward_m_fix, tau_m_fix, knowledge_m_fix, exploration_m_fix) =perform_experiments(N_bandits, N_experiments, N_episodes, tau_fix, shock_prob, tau_strategy = "stable")

# Calculate exploration probability
exploration_prob_fix= np.count_nonzero(exploration_m_fix, axis=1)/N_episodes
exploration_percent_avg_fix=np.mean(exploration_prob_fix)

# Calculate accumulated asset stock (rewards_cum)
rewards_cum_fix=np.cumsum(reward_m_fix,axis=1)
rewards_cum_avg_fix=np.mean(rewards_cum_fix,axis=0)
rewards_end_avg_fix=rewards_cum_avg_fix[-1]

# Calculate average knowledge at the end of the final period
knowledge_avg_over_time_fix=np.mean(knowledge_m_fix, axis=0)
knowledge_end_avg_fix=knowledge_avg_over_time_fix[-1]

#Calculate average tau for each period
tau_avg_fix=np.mean(tau_m_fix,axis=0)*N_bandits
tau_end_avg_fix=tau_avg_fix[-1]

#-----------------------------------------
# Perform experiment for ACCUMULATED RESOURCES
#-----------------------------------------

(reward_m_acc, tau_m_acc, knowledge_m_acc, exploration_m_acc) =perform_experiments(N_bandits, N_experiments, N_episodes, tau_acc, shock_prob, tau_strategy= "accumulated resources")

# Calculate exploration probability
exploration_prob_acc= np.count_nonzero(exploration_m_acc, axis=1)/N_episodes
exploration_percent_avg_acc=np.mean(exploration_prob_acc)

# Calculate accumulated asset stock (rewards_cum)
rewards_cum_acc=np.cumsum(reward_m_acc,axis=1)
rewards_cum_avg_acc=np.mean(rewards_cum_acc,axis=0)
rewards_end_avg_acc=rewards_cum_avg_acc[-1]

# Calculate average knowledge at the end of the final period
knowledge_avg_over_time_acc=np.mean(knowledge_m_acc, axis=0)
knowledge_end_avg_acc=knowledge_avg_over_time_acc[-1]

#Calculate average tau for each period
tau_avg_acc=np.mean(tau_m_acc,axis=0)*N_bandits
tau_end_avg_acc=tau_avg_acc[-1]

#--------------------------------------
#Print main insights FIXED TAU
print('FIXED TAU Strategy:')
print('-------------')
print('Exploration probability = '+str(exploration_percent_avg_fix))
print('Performance = '+ str(rewards_end_avg_fix))
print('Knowledge = ' + str(knowledge_end_avg_fix))
print('Tau = ' + str(tau_end_avg_fix))
print('--------------')

#Print main insights ACCUMULATED RESOURCES 
print('ACCUMULATED RESOURCES Strategy:')
print('-------------')
print('Exploration probability = '+str(exploration_percent_avg_acc))
print('Performance = '+ str(rewards_end_avg_acc))
print('Knowledge = ' + str(knowledge_end_avg_acc))
print('Tau = ' + str(tau_end_avg_acc))
print('--------------')

# =========================
# Plot
# =========================
# Plot Cumulated Performance over time
plt.plot(range(N_episodes), rewards_cum_avg_fix)
plt.plot(range(N_episodes), rewards_cum_avg_acc)
plt.xlabel('Periods')
plt.ylabel('Cumulated Performance')
plt.ylim(0,N_episodes)
plt.title('Cumulated Performance over time per strategy')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""first figure""")

# Plot Knowledge over time
plt.plot(range(N_episodes),knowledge_avg_over_time_fix)
plt.plot(range(N_episodes),knowledge_avg_over_time_acc)
plt.xlabel('Periods')
plt.ylabel('Knowledge')
plt.ylim(0,1)
plt.title('Knowledge over time per strategy')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""second figure""")

#Plot tau over time
plt.plot(range(N_episodes),tau_avg_fix)
plt.plot(range(N_episodes),tau_avg_acc)
plt.xlabel('Periods')
plt.ylabel('Tau')
plt.ylim(0,5)
plt.title('Tau over time per strategy')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""third figure""")


plt.plot(rewards_cum_avg_acc,tau_avg_acc)
plt.xlabel('Acc Resources')
plt.ylabel('Tau')
plt.ylim(0,10)
plt.title('Tau over time')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""third figure""")

