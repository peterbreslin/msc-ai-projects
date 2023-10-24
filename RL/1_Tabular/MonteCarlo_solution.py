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

class MonteCarloAgent:

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
		

	def update(self, states, actions, rewards, done):
		''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
		actions is a list of actions observed in the episode, of length T_ep
		rewards is a list of rewards observed in the episode, of length T_ep
		done indicates whether the final s in states is was a terminal state '''

		# TO DO: Add own code

		Gt = 0 #start accumulating rewards from the first timestep in the episode
		T = len(actions) #maximum episode length

		for i in range(T-1,-1,-1):
			Gt = rewards[i] + (self.gamma * Gt) #discounted sum of rewards starting from t until the end of the episode
			s = states[i]
			a = actions[i]
			self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (Gt - self.Q_sa[s,a]) #standard tabular update


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
				   policy='egreedy', epsilon=None, temp=None, plot=True):
	''' runs a single repetition of an MC rl agent
	Return: rewards, a vector with the observed rewards at each timestep ''' 
	
	env = StochasticWindyGridworld(initialize_model=False)
	pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
	rewards = []
	budget = 0
	

	# TO DO: Write your Monte Carlo RL algorithm here!

	while budget < n_timesteps:
		s = env.reset()
		states = [s]
		actions = []
\
		for t in range(max_episode_length):
			a = pi.select_action(s, policy, epsilon, temp)
			s_next, r, done = env.step(a)
			states.append(s_next)
			actions.append(a)
			rewards.append(r)

			budget += 1
			if done or (budget >= n_timesteps):
				break

		pi.update(states, actions, rewards, done)
	
	# if plot:
	#    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

	return rewards 
	
def test():
	n_timesteps = 10000
	max_episode_length = 100
	gamma = 1.0
	learning_rate = 0.1

	# Exploration
	policy = 'egreedy' # 'egreedy' or 'softmax' 
	epsilon = 0.1
	temp = 1.0
	
	# Plotting parameters
	plot = True

	rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
				   policy, epsilon, temp, plot)
	# print("Obtained rewards: {}".format(rewards))  
	print(max(rewards))


			
if __name__ == '__main__':
	test()
