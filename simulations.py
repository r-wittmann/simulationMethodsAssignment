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
N_experiments=1000
N_episodes=500
tau=0.5/N_bandits
shock_prob=0
tau_strategy="stable"

(reward_m, tau_m, knowledge_m, exploration_m) =perform_experiments(N_bandits, N_experiments, N_episodes, tau, shock_prob, tau_strategy)

# Calculate exploration probability
exploration_prob= np.count_nonzero(exploration_m, axis=1)/N_episodes
exploration_percent_avg=np.mean(exploration_prob)

# Calculate accumulated asset stock (rewards_cum)
rewards_cum=np.cumsum(reward_m,axis=1)
rewards_cum_avg=np.mean(rewards_cum,axis=0)
rewards_end_avg=rewards_cum_avg[-1]

# Calculate average knowledge at the end of the final period
knowledge_avg_over_time=np.mean(knowledge_m, axis=0)
knowledge_end_avg=knowledge_avg_over_time[-1]

#Calculate average tau for each period
tau_avg=np.mean(tau_m,axis=0)*N_bandits
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
