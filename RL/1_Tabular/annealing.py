#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time

from Q_learning import q_learning
from SARSA import sarsa
from MonteCarlo import monte_carlo
from Nstep import n_step_Q
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=51, plot=False, n=5, anneal=False):

    reward_results = np.empty([n_repetitions,n_timesteps]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        if backup == 'q':
            rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, anneal=anneal)
        elif backup == 'sarsa':
            rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        elif backup == 'mc':
            rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
        elif backup == 'nstep':
            rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)

        reward_results[rep] = rewards

    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment():
    ####### Settings
    # Experiment    
    n_repetitions = 50
    smoothing_window = 1001
    plot = False # Plotting is very slow, switch it off when we run repetitions
    
    # MDP    
    n_timesteps = 50000
    max_episode_length = 150
    gamma = 1.0

    # Parameters we will vary in the experiments, set them to some initial values: 
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.05
    temp = 1.0
    # Back-up & updateC
    backup = 'q' # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.25
    n = 5 # only used when backup = 'nstep'
        
    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning',
                  'sarsa': 'SARSA',
                  'mc': 'Monte Carlo',
                  'nstep': 'n-step Q-learning'}
    
    ####### Experiments
    
    optimal_average_reward_per_timestep = 1.3 
    
    #### Assignment 2: Effect of exploration
    epsilons = [1.0, 1.0]
    temps = [1.0, 1.0] 
    learning_rate = 0.25
    backup = 'q'
    Plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration (annealing)')  

    annealing = [True, False]
    label = ['with annealing', 'without annealing']

    for i in range(2):        
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
            gamma, 'egreedy', epsilons[i], temp, smoothing_window, plot, n, anneal=annealing[i])
        Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy {}'.format(label[i]))    

    for i in range(2): 
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
            gamma, 'softmax', epsilon, temps[i], smoothing_window, plot, n, anneal=annealing[i])
        Plot.add_curve(learning_curve,label=r'softmax {}'.format(label[i]))

    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('exploration_annealing.png')

if __name__ == '__main__':
    experiment()
