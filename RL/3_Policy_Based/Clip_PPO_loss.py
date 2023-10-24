import os
import sys
import torch
import numpy as np
from catch_alt import Catch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cpu')



# Policy network (actor)
def create_NN(env):
	#function to create the policy neural network

	obs_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2] 
	n_actions = env.action_space.n  

	model = torch.nn.Sequential(
			torch.nn.Flatten(start_dim=-3, end_dim=- 1),
			torch.nn.Linear(obs_size, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 126),
			torch.nn.ReLU(),
			torch.nn.Linear(126, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, n_actions),
			torch.nn.Softmax(dim=-1)
		 )
	return model



# Value network (critic)
def create_V_Network(env):
	#function to create the network for the V-values

	obs_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2] 
	n_actions = env.action_space.n  
	
	model = torch.nn.Sequential(
			torch.nn.Flatten(start_dim=-3, end_dim=- 1),
			torch.nn.Linear(obs_size, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 126),
			torch.nn.ReLU(),
			torch.nn.Linear(126, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 1)
		 )
	return model



# Required to avoid inplace operation errors when performing Clip-PPO
def comp_pi_batch(model, s_batch, a_batch):
	pred_batch = model(s_batch).to(device)
	pi_batch = pred_batch[np.arange(pred_batch.shape[0]), a_batch]
	return torch.log(pi_batch)



def training(env, model, model_V, learning_rate, n, beta, horizon, max_trajectories, gamma, epsilon):
	#main function to train the PPO-Clip algorithms. 
	#env: the catch environment 
	#model: the policy neural network model
	#model_V: the V-value neural network model
	#learning_rate: learning rate for gradient ascent 
	#n: the value for the n-step used during bootstrapping
	#beta: the hyperparameter controlling entropy regularization (called eta in the report)
	#horizon: maximum amount of action that can be taken in a game, i.e maximum amount of sampling that can be done in one episode
	#max_trajectories: number of episodes we want to train the models for 
	#gamma: discount factor
	#epsilon: clipping threshold

	

	#defining the optimizers for the models
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	optimizer_V = torch.optim.Adam(model_V.parameters(), lr=learning_rate)
	n_actions = env.action_space.n

	#traget models are needed, which are basically copies of the original models. These are needed because inplace operation are performed on the networks
	#and this causes errors in Pytorch otherwise
	model_tmp = create_NN(env)   #target model for the policy network
	model_tmp.load_state_dict(model.state_dict())
	optimizer_tmp = torch.optim.Adam(model_tmp.parameters(), lr=learning_rate)
	optimizer_state_dict = optimizer.state_dict()
	optimizer_tmp.load_state_dict(optimizer_state_dict)

	model_V_tmp = create_V_Network(env) #target model for the V-value network
	model_V_tmp.load_state_dict(model_V.state_dict())
	optimizer_V_tmp = torch.optim.Adam(model_V_tmp.parameters(), lr=learning_rate)
	optimizer_state_dict = optimizer_V.state_dict()
	optimizer_V_tmp.load_state_dict(optimizer_state_dict)
	
	score = [] 
	#main loop for epsiodes of playing the game and training the models
	for trajectory in range(max_trajectories):
		s = env.reset()
		done = False
		transitions = [] 
		R = 0
		#loop where we sample the environment (i.e we play the game)
		for t in range(horizon):
			a_prob = model(torch.from_numpy(s.T).float())
			a = np.random.choice(n_actions, p=a_prob.data.numpy())
			s_old = s
			s, r, done, _ = env.step(a) 
			transitions.append((s_old.T, a, r))  #store the important parameters of the MCMC, i,e the state, the action, and the reward
			R += r
			if done: 
				break

		score.append(R)
		r_batch = torch.Tensor([r for (s,a,r) in transitions])	 #array of rewards obtained at each step during the episode
		s_batch = torch.Tensor([s for (s,a,r) in transitions]) 	 #array of states at each step during the episode
		a_batch = torch.Tensor([a for (s,a,r) in transitions]).long()	#array of actions taken at each step during the episode
		

		
		Q_batch =[] #list that will contain the Q-values (the cumulative rewards) needed for the gradient ascent of the models
		V_pred = model_V(s_batch[:]) ##get the V-values for all the states that were visited during the episode

		#calculating the estimated Q-values
		for t in range(len(transitions)):
			Q = 0

			if t + n < len(transitions):
				V = V_pred[t + n]
			else:
				V = 0

			if n > 0:	#appiles to actor-critic with bootstrapping 
				Q += gamma**n * V
				for j in range(0, n):
					if t + j >= len(transitions):
						pass
					else:
						Q = Q + gamma**j * (r_batch[t + j]) 
			else:	#i.e when n = 0, there is no bootstrapping and it's just like in the reinforce algorithm
				for j in range(t, len(transitions)):
					Q = Q + gamma**(j - t) * (r_batch[j]) 


			Q_batch.append(Q)
		
		Q_batch = torch.FloatTensor(Q_batch)
		advantage = Q_batch - V_pred[:, 0]
		


		# We now implement a Clip_PPO policy gradient loss:

		log_pi_batch = comp_pi_batch(model_tmp, s_batch, a_batch) #batch of log(pi(a|s))
		advantage = Q_batch - V_pred[:, 0]
		K_epochs = 5 #number of mini-batch updates (i.e. a mini-batch will be sampled for each update)

		for K in range(K_epochs):
			log_curr_pi_batch = comp_pi_batch(model_tmp, s_batch, a_batch)
			ratio = torch.exp(log_curr_pi_batch - log_pi_batch.detach()) #ratio of the new-to-old policy

			# Now clip this ratio by the clipping threshold epsilon so as to prevent large updates to the policy
			clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon) 

			# Calculate the surrogate losses for estimating the policy gradient loss
			surrogate_term1 = ratio * advantage 
			surrogate_term2 = clipped_ratio * advantage

			# Calculate the clipped surrogate loss
			loss = -torch.min(surrogate_term1, surrogate_term2).mean() 

			# Note: we take the negative of this loss function because the Adam optimizer minimizes the loss but we're 
			# trying to maximize the objective function, so minimizing the negative objective function maximizes it.
			
			# Now compute the gradients of the loss function with respect to the policy network's parameters
			optimizer_tmp.zero_grad() #clears the gradients from the previous step
			loss.backward(retain_graph=True) #gradients are calculated wrt the weights of the policy network
			optimizer_tmp.step() #updates the policy network weights based on the gradients computed by the backward pass

			# Now for updating the value network 
			loss_V = torch.sum(torch.pow(advantage, 2))
			optimizer_V_tmp.zero_grad() 
			loss_V.backward(retain_graph=True) 
			optimizer_V_tmp.step() 
			
		
		model.load_state_dict(model_tmp.state_dict())
		model_V.load_state_dict(model_V_tmp.state_dict())

		#every 200 episodes we check if we want to stop training. We stop training if the optimal policy has been reached in all of the last 50 episodes,
		#or when we see that the model is not learning
		if trajectory % 200 == 0 and trajectory>0:
				print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, np.mean(score[-50:-1])))
				if np.mean(score[-50:-1]) >= 35:
					break
				if (trajectory >= 200000) and ((np.mean(score[-1000:-1]) < -7) or (loss == 0)):
					break


	return np.array(score)



if __name__ == '__main__':
	
	rows = 7
	columns = 7
	speed = 1.0
	max_steps = 250
	max_misses = 10
	observation_type = 'pixel' # 'vector'
	seed = None

	# Initialize environment and Q-array
	env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
				max_misses=max_misses, observation_type=observation_type, seed=seed)
	s = env.reset()

	# Hyperparameters
	n = 10
	beta = 0.075
	gamma = 0.99
	horizon = 250
	learning_rate = 0.001
	max_trajectories = 500000  

	epsilon = 0.2 #clipping threshold
	model = create_NN(env)
	model_V = create_V_Network(env)
	score = training(env, model, model_V, learning_rate, n, beta, horizon, max_trajectories, gamma, epsilon)
	np.save('score_ppo.npy', score)

