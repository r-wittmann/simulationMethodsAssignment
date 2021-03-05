#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 2021

@author: r-wittmann
@author: Konstantin0694
"""

"""
this simulation runs experiments in a stable environment for both the stable
tau-strategy, as well as the accumulated resource strategy. It plots the results
for comparison.
All relevant parameters which are used for later, are suffixed with _2
"""

import numpy as np
import matplotlib.pyplot as plt
from bandit_experiment import perform_experiments

# =========================
# Perform an experiment
# =========================

# Set parameters to perform both experiment
N_bandits=10
N_experiments=10
N_periods=500
tau_fix_2=0.1/N_bandits
tau_acc_2=0.1/N_bandits
shock_prob_2=0.0
# --------------------------------
#Perform experiment for FIXED TAU
#---------------------------------

(reward_m_fix_2, tau_m_fix_2, knowledge_m_fix_2, exploration_m_fix_2) =perform_experiments(N_bandits, N_experiments, N_periods, tau_fix_2, shock_prob_2, tau_strategy = "stable")

# Calculate exploration probability
exploration_prob_fix_2= np.count_nonzero(exploration_m_fix_2, axis=1)/N_periods
exploration_percent_avg_fix_2=np.mean(exploration_prob_fix_2)

# Calculate accumulated asset stock (rewards_cum)
rewards_cum_fix_2=np.cumsum(reward_m_fix_2,axis=1)
rewards_cum_avg_fix_2=np.mean(rewards_cum_fix_2,axis=0)
rewards_end_avg_fix_2=rewards_cum_avg_fix_2[-1]

# Calculate average knowledge at the end of the final period
knowledge_avg_over_time_fix_2=np.mean(knowledge_m_fix_2, axis=0)
knowledge_end_avg_fix_2=knowledge_avg_over_time_fix_2[-1]

#Calculate average tau for each period
tau_avg_fix_2=np.mean(tau_m_fix_2,axis=0)*N_bandits
tau_end_avg_fix_2=tau_avg_fix_2[-1]

#-----------------------------------------
# Perform experiment for ACCUMULATED RESOURCES
#-----------------------------------------

(reward_m_acc_2, tau_m_acc_2, knowledge_m_acc_2, exploration_m_acc_2) =perform_experiments(N_bandits, N_experiments, N_periods, tau_acc_2, shock_prob_2, tau_strategy= "accumulated resources")

# Calculate exploration probability
exploration_prob_acc_2= np.count_nonzero(exploration_m_acc_2, axis=1)/N_periods
exploration_percent_avg_acc_2=np.mean(exploration_prob_acc_2)

# Calculate accumulated asset stock (rewards_cum)
rewards_cum_acc_2=np.cumsum(reward_m_acc_2,axis=1)
rewards_cum_avg_acc_2=np.mean(rewards_cum_acc_2,axis=0)
rewards_end_avg_acc_2=rewards_cum_avg_acc_2[-1]

# Calculate average knowledge at the end of the final period
knowledge_avg_over_time_acc_2=np.mean(knowledge_m_acc_2, axis=0)
knowledge_end_avg_acc_2=knowledge_avg_over_time_acc_2[-1]

#Calculate average tau for each period
tau_avg_acc_2=np.mean(tau_m_acc_2,axis=0)*N_bandits
tau_end_avg_acc_2=tau_avg_acc_2[-1]

#--------------------------------------
#Print main insights FIXED TAU
print('FIXED TAU Strategy:')
print('-------------')
print('Exploration probability = '+str(exploration_percent_avg_fix_2))
print('Performance = '+ str(rewards_end_avg_fix_2))
print('Knowledge = ' + str(knowledge_end_avg_fix_2))
print('Tau = ' + str(tau_end_avg_fix_2))
print('--------------')

#Print main insights ACCUMULATED RESOURCES 
print('ACCUMULATED RESOURCES Strategy:')
print('-------------')
print('Exploration probability = '+str(exploration_percent_avg_acc_2))
print('Performance = '+ str(rewards_end_avg_acc_2))
print('Knowledge = ' + str(knowledge_end_avg_acc_2))
print('Tau = ' + str(tau_end_avg_acc_2))
print('--------------')

# =========================
# Plot
# =========================
# Plot Cumulated Performance over time

plt.plot(range(N_periods), rewards_cum_avg_fix_2)
plt.plot(range(N_periods), rewards_cum_avg_acc_2)
plt.xlabel('Periods')
plt.ylabel('Cumulated Performance')
plt.ylim(0,N_periods)
plt.xlim(0,N_periods)
plt.title('Cumulated Performance over time per strategy')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""first figure""")

# Plot Knowledge over time
plt.plot(range(N_periods),knowledge_avg_over_time_fix_2)
plt.plot(range(N_periods),knowledge_avg_over_time_acc_2)
plt.xlabel('Periods')
plt.ylabel('Knowledge')
plt.ylim(0,1)
plt.xlim(0,N_periods)
plt.title('Knowledge over time per strategy')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""second figure""")

#Plot tau over time
plt.plot(range(N_periods),tau_avg_fix_2)
plt.plot(range(N_periods),tau_avg_acc_2)
plt.xlabel('Periods')
plt.ylabel('Tau')
plt.ylim(0,5)
plt.xlim(0,N_periods)
plt.title('Tau over time per strategy')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""third figure""")


plt.plot(rewards_cum_avg_acc_2,tau_avg_acc_2)
plt.xlabel('Accumulated Resources')
plt.ylabel('Tau')
plt.ylim(0,5)
plt.xlim(0)
plt.title('Tau over time')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""third figure""")

