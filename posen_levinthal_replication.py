#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 15:34:18 2021

@author: rainer-wittmann
"""

import numpy as np
import matplotlib.pyplot as plt
from bandit_experiment import perform_experiments

# =========================
# reproduce Posen and Levinthal Paper
# =========================

def replicate_posen_levinthal(N_bandits = 10, N_experiments = 500, N_episodes = 500):
    print("Performing a replication of Posen and Levinthal 2012\n")
    
    p_l_taus = [0.02, 0.25, 0.5, 0.75, 1]
    
    p_l_results = np.matrix([
            p_l_taus,
            np.zeros(len(p_l_taus)),
            np.zeros(len(p_l_taus)),
            np.zeros(len(p_l_taus))
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
    print("Tau:         {}".format(p_l_results[0,:]))
    print("Exploration: {}".format(p_l_results[1,:].round(5)))
    print("Performance: {}".format(p_l_results[2,:]))
    print("Knowledge:   {}".format(p_l_results[3,:].round(5)))
    
    # plot exploration probability
    plt.plot(p_l_results[0,:].tolist()[0], p_l_results[1,:].tolist()[0])
    plt.xlabel('Tau')
    plt.ylabel('Exploration Probability')
    plt.title('Exporation')
    plt.figure("""first figure""")
    
    # plot performance
    plt.plot(p_l_results[0,:].tolist()[0], p_l_results[2,:].tolist()[0])
    plt.xlabel('Tau')
    plt.ylabel('Cumulated Performance')
    plt.title('Performance')
    plt.figure("""second figure""")
    
    # plot knowledge
    plt.plot(p_l_results[0,:].tolist()[0], p_l_results[3,:].tolist()[0])
    plt.xlabel('Tau')
    plt.ylabel('Knowledge')
    plt.title('Knowledge')
    plt.figure("""third figure""")
  
# uncomment next line to run the experiment    
# replicate_posen_levinthal(10,5000,500)
