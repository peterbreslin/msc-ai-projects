#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class NstepQLearningAgent:

	def __init__(self, n_states, n_actions, learning_rate, gamma, n):
		self.n_states = n_states
		self.n_actions = n_actions
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.n = n #n = depth = max_episode_length
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

		T = len(actions) #maximum episode length

		for t in range(T):
			m = min(self.n, T-t) #m won't always be able to go n steps ahead

			for i in range(m):
				if done and (t+m == T): #can finish loop without reaching terminal state
					Gt = np.sum(self.gamma**i * rewards[t+i])
				else:
					Gt = np.sum(self.gamma**i * rewards[t+i] + (self.gamma**m * np.max(self.Q_sa[states[t+m]])))

			s = states[t]
			a = actions[t]
			self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (Gt - self.Q_sa[s,a]) #standard tabular update

		

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
				   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
	''' runs a single repetition of an MC rl agent
	Return: rewards, a vector with the observed rewards at each timestep ''' 
	
	env = StochasticWindyGridworld(initialize_model=False)
	pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
	rewards = []
	budget = 0


	# TO DO: Write your n-step Q-learning algorithm here!

	while budget <= n_timesteps:
		for i in range(n_timesteps):
			s = env.reset()
			states = [s]
			actions = []
			rewards_ep = []

			for t in range(max_episode_length):
				a = pi.select_action(s, policy, epsilon, temp)
				s_next, r, done = env.step(a)
				states.append(s_next)
				actions.append(a)
				rewards_ep.append(r)
				budget += 1

				if done or (budget > n_timesteps):
					break

			rewards.append(np.mean(rewards_ep))
		pi.update(states, actions, rewards, done)

	if plot:
	   env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=5.1) # Plot the Q-value estimates during n-step Q-learning execution

	return rewards 

def test():
	n_timesteps = 10000
	max_episode_length = 100
	gamma = 1.0
	learning_rate = 0.1
	n = 5
	
	# Exploration
	policy = 'egreedy' # 'egreedy' or 'softmax' 
	epsilon = 0.1
	temp = 1.0
	
	# Plotting parameters
	plot = True

	rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
				   policy, epsilon, temp, plot, n=n)
	# print("Obtained rewards: {}".format(rewards)) 
	print(max(rewards))   
	
if __name__ == '__main__':
	test()
