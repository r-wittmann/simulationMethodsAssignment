#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 2021

@author: r-wittmann
@author: Konstantin0694
"""

"""
this simulation runs experiments in an unstable environment for both the stable
tau-strategy, as well as the accumulated resource strategy. It plots the results
for comparison.
All relevant parameters which are used for later, are suffixed with _3
"""

import numpy as np
import matplotlib.pyplot as plt
from bandit_experiment import perform_experiments


# =========================
# Perform an experiment
# =========================

# Input parameters to perform experiment
N_bandits=10
N_experiments=5
N_episodes=500
# shock_prob_3=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]
shock_prob_3=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
tau_3 = 0.5
#---------------------------------------------
# Perform FIXED TAU strategy for different taus    
#---------------------------------------------

p_l_results_fix_3 = np.matrix([
            shock_prob_3,
            np.zeros(len(shock_prob_3)),
            np.zeros(len(shock_prob_3)),
            np.zeros(len(shock_prob_3))
        ])
    
for i in range(len(shock_prob_3)):
    
    (reward_m_fix_3, tau_m_fix_3, knowledge_m_fix_3, exploration_m_fix_3) = perform_experiments(N_bandits, N_experiments, N_episodes, tau=tau_3/N_bandits, shock_prob=shock_prob_3[i], tau_strategy="stable")
        
    exploration_prob_fix_3= np.count_nonzero(exploration_m_fix_3, axis=1)/N_episodes
    p_l_results_fix_3[1,i] = np.mean(exploration_prob_fix_3)
        
    rewards_cum_fix_3=np.cumsum(reward_m_fix_3,axis=1)
    p_l_results_fix_3[2,i] = np.mean(rewards_cum_fix_3[:,-1],axis=0)
        
    knowledge_avg_over_time_fix_3=np.mean(knowledge_m_fix_3, axis=0)
    p_l_results_fix_3[3,i] = knowledge_avg_over_time_fix_3[-1]
    


#---------------------------------------------
# Perform ACCUMULATED RESOURCES strategy for different taus    
#---------------------------------------------
p_l_results_acc_3 = np.matrix([
            shock_prob_3,
            np.zeros(len(shock_prob_3)),
            np.zeros(len(shock_prob_3)),
            np.zeros(len(shock_prob_3))
        ])
    
for i in range(len(shock_prob_3)):
    
    (reward_m_acc_3, tau_m_acc_3, knowledge_m_acc_3, exploration_m_acc_3) = perform_experiments(N_bandits, N_experiments, N_episodes, tau=tau_3/N_bandits, shock_prob=shock_prob_3[i], tau_strategy="accumulated resources")
        
    exploration_prob_acc_3= np.count_nonzero(exploration_m_acc_3, axis=1)/N_episodes
    p_l_results_acc_3[1,i] = np.mean(exploration_prob_acc_3)
        
    rewards_cum_acc_3=np.cumsum(reward_m_acc_3,axis=1)
    p_l_results_acc_3[2,i] = np.mean(rewards_cum_acc_3[:,-1],axis=0)
        
    knowledge_avg_over_time_acc_3=np.mean(knowledge_m_acc_3, axis=0)
    p_l_results_acc_3[3,i] = knowledge_avg_over_time_acc_3[-1]

print("---------------------------------------------------------------")    
print("Here are the results for FIXED TAU Strategy for different turbulence levels")
print("Turbulence:  {}".format(p_l_results_fix_3[0,:]))
print("Exploration: {}".format(p_l_results_fix_3[1,:].round(5)))
print("Performance: {}".format(p_l_results_fix_3[2,:]))
print("Knowledge:   {}".format(p_l_results_fix_3[3,:].round(5)))
print("---------------------------------------------------------------")  
print("Here are the results for ACCUMULATED RESOURCES Strategy for different turbulence levels")
print("Turbulence:  {}".format(p_l_results_acc_3[0,:]))
print("Exploration: {}".format(p_l_results_acc_3[1,:].round(5)))
print("Performance: {}".format(p_l_results_acc_3[2,:]))
print("Knowledge:   {}".format(p_l_results_acc_3[3,:].round(5)))    
print("---------------------------------------------------------------")  
   
# plot exploration probability
plt.plot(p_l_results_fix_3[0,:].tolist()[0], p_l_results_fix_3[1,:].tolist()[0])
plt.plot(p_l_results_fix_3[0,:].tolist()[0], p_l_results_acc_3[1,:].tolist()[0])
plt.xlabel('Turbulence')
plt.ylabel('Exploration Probability')
plt.title('Exploration')
plt.xlim(0)
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""first figure""")
    
# plot performance
plt.plot(p_l_results_fix_3[0,:].tolist()[0], p_l_results_fix_3[2,:].tolist()[0])
plt.plot(p_l_results_fix_3[0,:].tolist()[0], p_l_results_acc_3[2,:].tolist()[0])
plt.xlabel('Turbulence')
plt.ylabel('Cumulated Performance')
plt.title('Performance')
plt.xlim(0)
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""second figure""")
    
# plot knowledge
plt.plot(p_l_results_fix_3[0,:].tolist()[0], p_l_results_fix_3[3,:].tolist()[0])
plt.plot(p_l_results_fix_3[0,:].tolist()[0], p_l_results_acc_3[3,:].tolist()[0])
plt.xlabel('Turbulence')
plt.ylabel('Knowledge')
plt.title('Knowledge')
plt.xlim(0)
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""third figure""")


