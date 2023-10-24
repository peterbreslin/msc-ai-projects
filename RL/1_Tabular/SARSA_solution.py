#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 1: Tabular Reinforcement Learning

Dynamic Programming Solution

Student: 	Peter Breslin
ID: 		s3228282
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class SarsaAgent:

	def __init__(self, n_states, n_actions, learning_rate, gamma):
		self.n_states = n_states
		self.n_actions = n_actions
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.Q_sa = np.zeros((n_states,n_actions))
		

	def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
		
		# The epsilon-greedy policy
		if policy == 'egreedy':
			if epsilon is None:
				raise KeyError("Provide an epsilon")
				
			if np.random.uniform(0, 1) < epsilon:
				a = np.random.choice(self.n_actions) #exploration
			else:
				a = argmax(self.Q_sa[s]) #exploitation
				
		# The softmax policy
		elif policy == 'softmax':
			if temp is None:
				raise KeyError("Provide a temperature")
				
			prob = softmax(self.Q_sa[s], temp)
			a = np.random.choice(self.n_actions, p=prob)
			
		return a
		

	def update(self,s,a,r,s_next,a_next):

		# Implementing the SARSA update
		Gt = r + (self.gamma * self.Q_sa[s_next, a_next])
		self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (Gt - self.Q_sa[s,a]) #standard tabular update
		

def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
	''' runs a single repetition of SARSA
	Return: rewards, a vector with the observed rewards at each timestep ''' 
	
	env = StochasticWindyGridworld(initialize_model=False)
	pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
	rewards = []

	# Implementing the SARSA algorithm
	s = env.reset()                                             #sample initial state
	a = pi.select_action(s, policy, epsilon, temp)              #sample action
	for t in range(n_timesteps):
		s_next, r, done = env.step(a)                           #simulate environment
		a_next = pi.select_action(s, policy, epsilon, temp)     #sample action
		pi.update(s, a, r, s_next, a_next)                      #SARSA
		rewards.append(r)

		# Check if s' is terminal
		if done:
			s = env.reset()
			a = pi.select_action(s, policy, epsilon, temp) 
		else:
			s = s_next
			a = a_next
	
		# if plot:
			# env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1) # Plot the Q-value estimates during SARSA execution

	return rewards 


def test():
	n_timesteps = 5000
	gamma = 1.0
	learning_rate = 0.1

	# Exploration
	policy = 'egreedy' # 'egreedy' or 'softmax' 
	epsilon = 0.1
	temp = 1.0
	
	# Plotting parameters
	plot = True

	rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
	# print("Obtained rewards: {}".format(rewards))
	print(max(rewards))        
	
if __name__ == '__main__':
	test()
