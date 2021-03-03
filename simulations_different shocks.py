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
N_experiments=500
N_episodes=500
shock_prob_sim=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
tau_sim = 0.5
#---------------------------------------------
#Perform FIXED TAU strategy for different taus    
#---------------------------------------------

p_l_results_fix = np.matrix([
            shock_prob_sim,
            np.zeros(len(shock_prob_sim)),
            np.zeros(len(shock_prob_sim)),
            np.zeros(len(shock_prob_sim))
        ])
    
for i in range(len(shock_prob_sim)):
    
    (reward_m_fix, tau_m_fix, knowledge_m_fix, exploration_m_fix) = perform_experiments(N_bandits, N_experiments, N_episodes, tau=tau_sim/N_bandits, shock_prob=shock_prob_sim[i], tau_strategy="stable")
        
    exploration_prob_fix= np.count_nonzero(exploration_m_fix, axis=1)/N_episodes
    p_l_results_fix[1,i] = np.mean(exploration_prob_fix)
        
    rewards_cum_fix=np.cumsum(reward_m_fix,axis=1)
    p_l_results_fix[2,i] = np.mean(rewards_cum_fix[:,-1],axis=0)
        
    knowledge_avg_over_time_fix=np.mean(knowledge_m_fix, axis=0)
    p_l_results_fix[3,i] = knowledge_avg_over_time_fix[-1]
    


#---------------------------------------------
#Perform ACCUMULATED RESOURCES strategy for different taus    
#---------------------------------------------
p_l_results_acc = np.matrix([
            shock_prob_sim,
            np.zeros(len(shock_prob_sim)),
            np.zeros(len(shock_prob_sim)),
            np.zeros(len(shock_prob_sim))
        ])
    
for i in range(len(shock_prob_sim)):
    
    (reward_m_acc, tau_m_acc, knowledge_m_acc, exploration_m_acc) = perform_experiments(N_bandits, N_experiments, N_episodes, tau=tau_sim/N_bandits, shock_prob=shock_prob_sim[i], tau_strategy="accumulated resources")
        
    exploration_prob_acc= np.count_nonzero(exploration_m_acc, axis=1)/N_episodes
    p_l_results_acc[1,i] = np.mean(exploration_prob_acc)
        
    rewards_cum_acc=np.cumsum(reward_m_acc,axis=1)
    p_l_results_acc[2,i] = np.mean(rewards_cum_acc[:,-1],axis=0)
        
    knowledge_avg_over_time_acc=np.mean(knowledge_m_acc, axis=0)
    p_l_results_acc[3,i] = knowledge_avg_over_time_acc[-1]

print("---------------------------------------------------------------")    
print("Here are the results for FIXED TAU Strategy for different turbulence levels")
print("Turbulence:  {}".format(p_l_results_fix[0,:]))
print("Exploration: {}".format(p_l_results_fix[1,:].round(5)))
print("Performance: {}".format(p_l_results_fix[2,:]))
print("Knowledge:   {}".format(p_l_results_fix[3,:].round(5)))
print("---------------------------------------------------------------")  
print("Here are the results for ACCUMULATED RESOURCES Strategy for different turbulence levels")
print("Turbulence:  {}".format(p_l_results_acc[0,:]))
print("Exploration: {}".format(p_l_results_acc[1,:].round(5)))
print("Performance: {}".format(p_l_results_acc[2,:]))
print("Knowledge:   {}".format(p_l_results_acc[3,:].round(5)))    
print("---------------------------------------------------------------")  
   
 # plot exploration probability
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_fix[1,:].tolist()[0])
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_acc[1,:].tolist()[0])
plt.xlabel('Turbulence')
plt.ylabel('Exploration Probability')
plt.title('Exploration')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""first figure""")
    
    # plot performance
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_fix[2,:].tolist()[0])
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_acc[2,:].tolist()[0])
plt.xlabel('Turbulence')
plt.ylabel('Cumulated Performance')
plt.title('Performance')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""second figure""")
    
    # plot knowledge
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_fix[3,:].tolist()[0])
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_acc[3,:].tolist()[0])
plt.xlabel('Turbulence')
plt.ylabel('Knowledge')
plt.title('Knowledge')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""third figure""")


