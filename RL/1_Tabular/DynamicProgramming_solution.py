#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 1: Tabular Reinforcement Learning

Dynamic Programming Solution

Student: 	Peter Breslin
ID: 		s3228282
"""

import numpy as np
from Helper import argmax
import matplotlib.pyplot as plt
from Environment import StochasticWindyGridworld

class QValueIterationAgent:
	''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

	def __init__(self, n_states, n_actions, gamma, threshold=0.01):
		self.n_states = n_states
		self.n_actions = n_actions
		self.gamma = gamma
		self.Q_sa  = np.zeros((n_states, n_actions))
		

	def select_action(self, s):
		''' Returns the greedy best action in state s ''' 
		a = argmax(self.Q_sa[s]) #index of maximum value
		return a
	
	def update(self, env, s, a):
		''' Function updates Q(s,a) using p_sas and r_sas '''

		# Calling the transition function and reward
		p_sas, r_sas = env.model(s, a) 

		# Q-value iteration update
		self.Q_sa[s,a] = np.sum(p_sas * (r_sas + self.gamma*np.max(self.Q_sa, axis=1)))

	
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
	''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

	QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
		
	# TO DO: IMPLEMENT Q-VALUE ITERATION HERE

	delta = threshold + 1 # initialize max error to be some value greater than the threshold
	i = 0

	# Termination criteria
	while delta > threshold:
		Q_prev = np.copy(QIagent.Q_sa)

		# Perform Q-value update for all (s,a) pairs
		for s in range(env.n_states):
			for a in range(env.n_actions):
				QIagent.update(env, s, a)
				delta = np.max(np.abs(Q_prev - QIagent.Q_sa)) 
		i += 1

		# # Keeping track of progression (taking screenshot at these moments)
		# if (i == 1) | (i == 9) | (delta < threshold):
		# 	step_pause=15.0
		# else:
		# 	step_pause=0.2

		# Plot current Q-value estimates & print max error
		env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
		print("Q-value iteration, iteration {}, max error {}".format(i, delta))
		
	return QIagent


def experiment():
	gamma = 1.0
	threshold = 0.001
	env = StochasticWindyGridworld(initialize_model=True)
	env.render()
	QIagent = Q_value_iteration(env, gamma, threshold)
	
	# TO DO: View optimal policy and compute mean reward per timestep under the optimal policy
	done = False
	s = env.reset()
	while not done:
		a = QIagent.select_action(s)
		s_next, r, done = env.step(a)
		env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
		s = s_next

	# Compute V*(s=3) i.e. the converged optimal value at the start
	V_s = np.max(QIagent.Q_sa[3])
	print("V(s=3) = {}".format(V_s))

	# Compute the average reward per timestep under the optimal policy
	final_reward = 40
	reward_per_step = -1

	# Average number of steps
	avg_steps = (V_s - final_reward) / reward_per_step

	# Average reward per timestep
	avg_reward_per_step = V_s / avg_steps

	print("Average reward per timestep under the optimal policy:", avg_reward_per_step)


if __name__ == '__main__':
	experiment()

