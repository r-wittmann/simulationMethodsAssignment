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
for comparison. This is for the peplication of the results of Posen and 
Levinthal 2012
All relevant parameters which are used for later, are suffixed with _1
"""

import numpy as np
import matplotlib.pyplot as plt
from bandit_experiment import perform_experiments


# =========================
# Perform an experiment
# =========================

# Input parameters to perform experiment
N_bandits=10
N_experiments=25
N_periods=500
shock_prob_1=0.00

p_l_taus_1 = [0.02, 0.25, 0.5, 0.75, 1]
#---------------------------------------------
# Perform FIXED TAU strategy for different taus    
#---------------------------------------------

p_l_results_fix_1 = np.matrix([
            p_l_taus_1,
            np.zeros(len(p_l_taus_1)),
            np.zeros(len(p_l_taus_1)),
            np.zeros(len(p_l_taus_1))
        ])
    
for i in range(len(p_l_taus_1)):
    
    (reward_m_fix_1, tau_m_fix_1, knowledge_m_fix_1, exploration_m_fix_1) = perform_experiments(N_bandits, N_experiments, N_periods, tau=p_l_taus_1[i]/N_bandits, shock_prob=shock_prob_1, tau_strategy="stable")
        
    exploration_prob_fix_1= np.count_nonzero(exploration_m_fix_1, axis=1)/N_periods
    p_l_results_fix_1[1,i] = np.mean(exploration_prob_fix_1)
        
    rewards_cum_fix_1=np.cumsum(reward_m_fix_1,axis=1)
    p_l_results_fix_1[2,i] = np.mean(rewards_cum_fix_1[:,-1],axis=0)
        
    knowledge_avg_over_time_fix_1=np.mean(knowledge_m_fix_1, axis=0)
    p_l_results_fix_1[3,i] = knowledge_avg_over_time_fix_1[-1]
    


#---------------------------------------------
# Perform ACCUMULATED RESOURCES strategy for different taus at t=0  
#---------------------------------------------
p_l_results_acc_1 = np.matrix([
            p_l_taus_1,
            np.zeros(len(p_l_taus_1)),
            np.zeros(len(p_l_taus_1)),
            np.zeros(len(p_l_taus_1))
        ])
    
for i in range(len(p_l_taus_1)):
    
    (reward_m_acc_1, tau_m_acc_1, knowledge_m_acc_1, exploration_m_acc_1) = perform_experiments(N_bandits, N_experiments, N_periods, tau=p_l_taus_1[i]/N_bandits, shock_prob=shock_prob_1, tau_strategy="accumulated resources")
        
    exploration_prob_acc_1= np.count_nonzero(exploration_m_acc_1, axis=1)/N_periods
    p_l_results_acc_1[1,i] = np.mean(exploration_prob_acc_1)
        
    rewards_cum_acc_1=np.cumsum(reward_m_acc_1,axis=1)
    p_l_results_acc_1[2,i] = np.mean(rewards_cum_acc_1[:,-1],axis=0)
        
    knowledge_avg_over_time_acc_1=np.mean(knowledge_m_acc_1, axis=0)
    p_l_results_acc_1[3,i] = knowledge_avg_over_time_acc_1[-1]

print("---------------------------------------------------------------")    
print("Here are the results for FIXED TAU Strategy for different Taus")
print("Tau:         {}".format(p_l_results_fix_1[0,:]))
print("Exploration: {}".format(p_l_results_fix_1[1,:].round(5)))
print("Performance: {}".format(p_l_results_fix_1[2,:]))
print("Knowledge:   {}".format(p_l_results_fix_1[3,:].round(5)))
print("---------------------------------------------------------------")  
print("Here are the results for ACCUMULATED RESOURCES Strategy for different Taus at t=0")
print("Tau:         {}".format(p_l_results_acc_1[0,:]))
print("Exploration: {}".format(p_l_results_acc_1[1,:].round(5)))
print("Performance: {}".format(p_l_results_acc_1[2,:]))
print("Knowledge:   {}".format(p_l_results_acc_1[3,:].round(5)))    
print("---------------------------------------------------------------")  
   
# plot exploration probability
plt.plot(p_l_results_fix_1[0,:].tolist()[0], p_l_results_fix_1[1,:].tolist()[0])
plt.plot(p_l_results_fix_1[0,:].tolist()[0], p_l_results_acc_1[1,:].tolist()[0])
plt.xlabel('Tau')
plt.ylabel('Exploration Probability')
plt.title('Exploration')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""first figure""")
    
# plot performance
plt.plot(p_l_results_fix_1[0,:].tolist()[0], p_l_results_fix_1[2,:].tolist()[0])
plt.plot(p_l_results_fix_1[0,:].tolist()[0], p_l_results_acc_1[2,:].tolist()[0])
plt.xlabel('Tau')
plt.ylabel('Cumulated Performance')
plt.title('Performance')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""second figure""")
    
# plot knowledge
plt.plot(p_l_results_fix_1[0,:].tolist()[0], p_l_results_fix_1[3,:].tolist()[0])
plt.plot(p_l_results_fix_1[0,:].tolist()[0], p_l_results_acc_1[3,:].tolist()[0])
plt.xlabel('Tau')
plt.ylabel('Knowledge')
plt.title('Knowledge')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""third figure""")


