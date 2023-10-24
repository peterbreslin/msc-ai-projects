import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from scipy.signal import savgol_filter
import torch
from matplotlib import pyplot as plt
from catch_alt import Catch



def create_NN(env, genre = 'Dense', observation_type = 'pixel'):
	#function to create the policy neural network
	#genre: defines if we built a 'Dense' or 'CNN' architecture
	#observation type: either 'pixel' or 'vector'. Note for CNN only pixel type works! 
	obs_size = env.observation_space.shape[0] 
	n_actions = env.action_space.n  

	if genre == 'CNN':
		obs_size = env.observation_space.shape[0] 
		n_actions = env.action_space.n  

		#Define the input shape
		input_shape = (2, 7, 7)
		

		c = 4 #number of channels for the the convolutional layer

		#Define model architecture
		model = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=input_shape[0], out_channels = c, kernel_size=1),
			torch.nn.ReLU(),
			torch.nn.Flatten(start_dim=-3, end_dim=- 1),
			torch.nn.Linear(input_shape[1] * input_shape[2] * c, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 3),
			torch.nn.Softmax(dim=-1)
		)

		#Print model summary
		print(model)
	elif genre == 'Dense':
		if observation_type=='vector':
			obs_size = env.observation_space.shape[0] 
			n_actions = env.action_space.n  
			
			#Define model architecture
			model = torch.nn.Sequential(
					torch.nn.Linear(obs_size, 256),
					torch.nn.ReLU(),
					torch.nn.Linear(256, 126),
					torch.nn.ReLU(),
					torch.nn.Linear(126, 256),
					torch.nn.ReLU(),
					torch.nn.Linear(256, n_actions),
					torch.nn.Softmax(dim=-1)
					)                
		elif observation_type=='pixel':
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
		else: 
			print('wrong observation type')


	return model


def create_V_Network(env, genre = 'Dense', observation_type = 'pixel'):
	#function to create the network for the V-values
	#genre: defines if we built a 'Dense' or 'CNN' architecture
	#observation type: either 'pixel' or 'vector'. Note for CNN only pixel type works! 


	obs_size = env.observation_space.shape[0] 
	n_actions = env.action_space.n  

	if genre == 'CNN':
		obs_size = env.observation_space.shape[0] 
		n_actions = env.action_space.n  
		
		#Define input shape
		input_shape = (2, 7, 7)

		c = 8 #number of channels for the first convolutional layer

		#Define model architecture
		model = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=input_shape[0], out_channels = c, kernel_size=1),
			torch.nn.ReLU(),
			torch.nn.Flatten(start_dim=-3, end_dim=- 1),
			torch.nn.Linear(input_shape[1] * input_shape[2] * c, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 1),
		)

		#Print model summary
		print(model)
	elif genre == 'Dense':
		if observation_type=='vector':
			obs_size = env.observation_space.shape[0] 
			n_actions = env.action_space.n  
			
			model = torch.nn.Sequential(
					torch.nn.Linear(obs_size, 256),
					torch.nn.ReLU(),
					torch.nn.Linear(256, 126),
					torch.nn.ReLU(),
					torch.nn.Linear(126, 256),
					torch.nn.ReLU(),
					torch.nn.Linear(256, 1)
					)
		elif observation_type=='pixel':
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
		else: 
			print('wrong observation type')


	return model


def training(env, model, model_V, learning_rate, n, beta, horizon, max_trajectories, gamma, i, path, actor_critic = True, baseline_sub = True):
	#main function to train our policy-based algorithms. It is possible to train REINFORCE, and Actor-Critic with this function
	#variables
	#env: the catch environment 
	#model: the policy neural network model
	#model_V: the V-value neural network model
	#learning_rate: learning rate for gradient ascent 
	#n: the value for the n-step used during bootstrapping
	#beta: the hyperparameter controlling entropy regularization (called eta in the report)
	#horizon: maximum amount of action that can be taken in a game, i.e maximum amount of sampling that can be done in one episode
	#max_trajectories: number of episodes we want to train the models for 
	#gamma: discount factor
	#i: index for naming the saved model at the end
	#path: path to save the model
	#actor_critic: boolean which defines if we do the actor critic algorithm of reinforce algorithm
	#baseline_sub: boolean which determines if the actor critic is done with baseline subtraction or not

	#Note: when you want to do Actor-Critic with bootstrap only you define: actor_critic = True, baseline_sub = False, n > 0
	#	   when you want to do Actor-Critic with baseline subtraction only you define: actor_critic = True, baseline_sub = True, n = 0
	#      when you want to do the full Actor-Critic you define: actor_critic = True, baseline_sub = True, n > 0
	#	   when you want to do the REINFORCE algorithm you define: actor_critic = False, n = 0


	#defining the optimizers for the models
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #optimizer for the policy network
	optimizer_V = torch.optim.Adam(model_V.parameters(), lr=learning_rate) #optimizer for the V-value network

	n_actions = env.action_space.n       #number of possible actions (3 for the catch environment)
	
	score = []      #list which records the total score obtained in each episode



	#main loop for epsiodes of playing the game and training the models
	for trajectory in range(max_trajectories):
		s = env.reset()   #every episode the environment needs to be reset 
		done = False
		transitions = [] 
		R = 0             #will keep track of the score in an episode

		#loop where we sample the environment (i.e we play the game)
		for t in range(horizon):
			a_prob = model(torch.from_numpy(s.T).float()) #get the probability distribution of the actions in state s
			a = np.random.choice(n_actions, p=a_prob.data.numpy())   #choose an action randomly based on its probability distribution
			s_old = s
			s, r, done, _ = env.step(a)  #take the action in the environment 
			transitions.append((s_old.T, a, r))   #store the important parameters of the MCMC, i,e the state, the action, and the reward
			R += r
			if done:  #game finished
				break

		score.append(R)
		r_batch = torch.Tensor([r for (s,a,r) in transitions])    #array of rewards obtained at each step during the episode
		s_batch = torch.Tensor([s for (s,a,r) in transitions]) 	  #array of states at each step during the episode
		a_batch = torch.Tensor([a for (s,a,r) in transitions]).long()  #array of actions taken at each step during the episode

		



		Q_batch =[]       #list that will contain the Q-values (the cumulative rewards) needed for the gradient ascent of the models
		V_pred = model_V(s_batch[:])    #get the V-values for all the states that were visited during the episode
		
		#calculating the estimated Q-values
		for t in range(len(transitions)):
			Q = 0

			if t + n < len(transitions):
				V = V_pred[t + n]
			else:
				V = 0

			if n > 0: #appiles to actor-critic with bootstrapping 
				Q += gamma**n * V
				for j in range(0, n):   #I think in the lectures notes it is k instead of j
					if t + j >= len(transitions):
						pass
					else:
						Q = Q + gamma**j * (r_batch[t + j]) 
			else:  #i.e when n = 0, there is no bootstrapping and it's just like in the reinforce algorithm
				for j in range(t, len(transitions)):
					Q = Q + gamma**(j - t) * (r_batch[j]) 





			Q_batch.append(Q)
		
		


		Q_batch = torch.FloatTensor(Q_batch)


		#we now come to the part where we update the policy netork ane the V-value network

		pred_batch = model(s_batch) + 1E-30 #batch of predictions done in the trajectory. I addded 1E-30 in order to prevent 0 probabilities which would result in nan values in the logarithmic function
		pi_batch = pred_batch[np.arange(pred_batch.shape[0]), a_batch]   #batch of pi(a|s)
		entropy = -torch.sum(torch.log(pred_batch) * pred_batch)         #entropy term 

		if actor_critic: #we train an actor critic model
			advantage = Q_batch - V_pred[:, 0]   #advantage value, this is the basline subtraction step
			if baseline_sub:   #we want to do baseline subtraction 
				loss = -(torch.sum(torch.log(pi_batch) * (advantage))  + beta * entropy)   #loss for the policy network, included entropy regularization and baseline subtraction
				
				#performing the gradient ascent on the policy model
				optimizer.zero_grad()    
				loss.backward(retain_graph=True) 
				optimizer.step()


				#update the Value network now
				loss_V = torch.sum(torch.pow(advantage, 2)) #loss for the V-value network
				
				#performing the gradient ascent on the V-value model
				optimizer_V.zero_grad()
				loss_V.backward()
				optimizer_V.step()
			else: #we don't want baseline subtraction
				loss = -(torch.sum(torch.log(pi_batch) * Q_batch)  + beta * entropy)   #loss for the policy network, does not include baseline subtraction
				
				#performing the gradient ascent on the policy model
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()


				#update the Value network now
				loss_V = torch.sum(torch.pow(advantage, 2)) #loss for the V-value network
				
				#performing the gradient ascent on the V-value model
				optimizer_V.zero_grad()
				loss_V.backward()
				optimizer_V.step()
		else:
			#reinforce algorithm
			loss = -(torch.sum(torch.log(pi_batch) * Q_batch)  + beta * entropy)   #loss for the policy network
				
			#performing the gradient ascent on the policy model
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()




		

		#every 200 episodes we check if we want to stop training. We stop training if the optimal policy has been reached in all of the last 50 episodes,
		#or when we see that the model is not learning
		if trajectory % 200 == 0 and trajectory>0:
				print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, np.mean(score[-50:-1])))
				if np.mean(score[-50:-1]) >= 35:#17:#27:#35:
					torch.save(model, path + 'model_{}'.format(i))
					break
				if (trajectory >= 10000) and ((np.mean(score[-1000:-1]) < -7) or (loss == 0)):
					break


		
	return np.array(score)  #return the score obtained in each epsiodes, used to create the learning curves

	
	




if __name__ == '__main__':
	#Note: when you want to do Actor-Critic with bootstrap only you define: actor_critic = True, baseline_sub = False, n > 0
	#	   when you want to do Actor-Critic with baseline subtraction only you define: actor_critic = True, baseline_sub = True, n = 0
	#      when you want to do the full Actor-Critic you define: actor_critic = True, baseline_sub = True, n > 0
	#	   when you want to do the REINFORCE algorithm you define: actor_critic = False, n = 0
	
	# Hyperparameters
	rows = 7
	columns = 7
	speed = 1
	max_steps = 250
	max_misses = 10
	observation_type = 'pixel' # 'vector'
	seed = None 
    
	
	# Initialize environment 
	env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
				max_misses=max_misses, observation_type=observation_type, seed=seed, wind = False)
	s = env.reset()

	
	
	#=======================================
	#hyperparameters
	learning_rate, n, beta = 0.001, 2, 0.075 #beta is the entropy regularization hyperparameter
	gamma = 0.99 #discount factor
	
	horizon = 250 #maximum length of an episode
	max_trajectories = 20000  #for how many episodes we want to train 
	gamma = 0.99 #discount factor

	genre = 'Dense'   #type of network
	path = 'environment_experiment/'
	actor_critic, baseline_sub = True, True
    
	if not os.path.exists(path):
		os.makedirs(path)
    
	for i in range(0, 5):
		
		model = create_NN(env, genre, observation_type)
		model_V = create_V_Network(env, genre, observation_type)
		score = training(env, model, model_V, learning_rate, n, beta, horizon, max_trajectories, gamma, i, path, actor_critic, baseline_sub)
        
		np.save(path + 'score_{}_{}_{}.npy'.format(rows, columns, i), score)
		