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
shock_prob=0.00

p_l_taus = [0.02, 0.25, 0.5, 0.75, 1]
#---------------------------------------------
#Perform FIXED TAU strategy for different taus    
#---------------------------------------------

p_l_results_fix = np.matrix([
            p_l_taus,
            np.zeros(len(p_l_taus)),
            np.zeros(len(p_l_taus)),
            np.zeros(len(p_l_taus))
        ])
    
for i in range(len(p_l_taus)):
    
    (reward_m_fix, tau_m_fix, knowledge_m_fix, exploration_m_fix) = perform_experiments(N_bandits, N_experiments, N_episodes, tau=p_l_taus[i]/N_bandits, shock_prob=shock_prob, tau_strategy="stable")
        
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
            p_l_taus,
            np.zeros(len(p_l_taus)),
            np.zeros(len(p_l_taus)),
            np.zeros(len(p_l_taus))
        ])
    
for i in range(len(p_l_taus)):
    
    (reward_m_acc, tau_m_acc, knowledge_m_acc, exploration_m_acc) = perform_experiments(N_bandits, N_experiments, N_episodes, tau=p_l_taus[i]/N_bandits, shock_prob=shock_prob, tau_strategy="accumulated resources")
        
    exploration_prob_acc= np.count_nonzero(exploration_m_acc, axis=1)/N_episodes
    p_l_results_acc[1,i] = np.mean(exploration_prob_acc)
        
    rewards_cum_acc=np.cumsum(reward_m_acc,axis=1)
    p_l_results_acc[2,i] = np.mean(rewards_cum_acc[:,-1],axis=0)
        
    knowledge_avg_over_time_acc=np.mean(knowledge_m_acc, axis=0)
    p_l_results_acc[3,i] = knowledge_avg_over_time_acc[-1]

print("---------------------------------------------------------------")    
print("Here are the results for FIXED TAU Strategy for different Taus")
print("Tau:         {}".format(p_l_results_fix[0,:]))
print("Exploration: {}".format(p_l_results_fix[1,:].round(5)))
print("Performance: {}".format(p_l_results_fix[2,:]))
print("Knowledge:   {}".format(p_l_results_fix[3,:].round(5)))
print("---------------------------------------------------------------")  
print("Here are the results for ACCUMULATED RESOURCES Strategy for different Taus")
print("Tau:         {}".format(p_l_results_acc[0,:]))
print("Exploration: {}".format(p_l_results_acc[1,:].round(5)))
print("Performance: {}".format(p_l_results_acc[2,:]))
print("Knowledge:   {}".format(p_l_results_acc[3,:].round(5)))    
print("---------------------------------------------------------------")  
   
 # plot exploration probability
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_fix[1,:].tolist()[0])
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_acc[1,:].tolist()[0])
plt.xlabel('Tau')
plt.ylabel('Exploration Probability')
plt.title('Exporation')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""first figure""")
    
    # plot performance
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_fix[2,:].tolist()[0])
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_acc[2,:].tolist()[0])
plt.xlabel('Tau')
plt.ylabel('Cumulated Performance')
plt.title('Performance')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""second figure""")
    
    # plot knowledge
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_fix[3,:].tolist()[0])
plt.plot(p_l_results_fix[0,:].tolist()[0], p_l_results_acc[3,:].tolist()[0])
plt.xlabel('Tau')
plt.ylabel('Knowledge')
plt.title('Knowledge')
plt.legend(("Fixed", "Accumulated Resources"))
plt.figure("""third figure""")


