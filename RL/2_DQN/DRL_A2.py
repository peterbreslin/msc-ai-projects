import sys
import gym
import heapq
import random
import numpy as np
from collections import deque

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, concatenate, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model, Sequential, clone_model
import tensorflow.keras.backend as K

# get the version of TensorFlow
print("TensorFlow version: {}".format(tf.__version__))
print('What is available:', tf.config.list_physical_devices('GPU'))
# Check that TensorFlow was build with CUDA to use the gpus
print("Device name: {}".format(tf.test.gpu_device_name()))
print("Build with GPU Support? {}".format(tf.test.is_built_with_gpu_support()))
print("Build with CUDA? {} ".format(tf.test.is_built_with_cuda()))




def create_DQN(input_size=4, output_size=2, nhidden_layers=3, neurons=[512, 256, 64], 
	act_hidden='relu', act_output='linear', loss_func='mse', opt='adam'):
	#function to create a general neural network
	#returns a neural network
	#input size: number of neurons in the input layer
	#output_size: number of neurons in the output layer
	#nhidden_layers: number of hidden layers
	#neurons: list of number of neurons for each hidden layer
	#act_hidden: the activation layer of each hidden layer
	#act_output: activation function of the output layer
	#loss_func: the loss function to minimize during training
	#opt: optimizer for training
	
  
	if len(neurons) != nhidden_layers:
		print('Missmatch between number of hidden layers and specified neurons')
		sys.exit()

	model = Sequential()
	model.add(Dense(neurons[0], input_dim=input_size, activation=act_hidden, kernel_initializer='he_uniform'))
	for n in neurons[1:]: 
		model.add(Dense(n, activation=act_hidden, kernel_initializer='he_uniform'))
	model.add(Dense(output_size, activation=act_output, kernel_initializer='he_uniform'))
	model.compile(loss=loss_func, optimizer=opt)
	print(model.summary())

	return model



def select_action(s, env, model, t, N_a, policy='egreedy', epsilon=None, c=None, action_size=2):
	#function to select the action in state s
	#returns an action
	#s: current state
	#env: the environment 
	#t: the number of updates that have already been performed, i.e the timestep
	#N_a: an array counting the number of times each action has been selected already
	#policy: exploration policy to use
	#epsilon: exploration parameter for epsilon-greedy exploration
	#c: confidence parameter for UCB exploration
	#action_size: number of possible actions

	# Epsilon-greedy algorithm
	if policy == 'egreedy':
		if epsilon is None:
			raise KeyError("Provide an epsilon")
		  
		x = np.random.uniform(low=0.0, high=1.0)
		if x < epsilon:
			#exploration
			a = np.random.randint(action_size)
		else:
			#greedy
			Q_sa = model.predict(s.reshape(1, len(s)))[0]
			a = np.argmax(Q_sa)


	# Upper Confidence Bound (UCB) algorithm 
	elif policy == 'ucb':
		if c is None:
			raise KeyError("Provide a c")

		# ucb1(a) = Q(a) + sqrt( 2*log(t) / N ) ... c=1

		# Get Q-values for the current state
		Q_sa = model.predict(s.reshape(1, len(s)))[0]
	
		# Calculate the exploration bonus for each action
		exploration_bonus = c * np.sqrt(np.log(t) / (N_a + 1e-30)) #avoid division by zero by adding a small value to N
		ucb = Q_sa + exploration_bonus

		a = np.argmax(ucb)
		  
		  
	return a




def novelty_exploration(s, state_archive, novelty_threshold = 0.5, k = 50):
	#function which checks if a state is novel. This is needed for novelty-based exploration
	#returns a boolean stating wether state s is novel or not
	#s: current state
	#state_archive: a list of all the states that have been visited so far by the agent
	#novelty_threshold: minimum value needed for a state to be regarded as novel
	#k: number of k-nearest neighbours that are considered for the novelty metric

	dist = np.linalg.norm(s - state_archive, axis = 1)   
	kNN = heapq.nsmallest(k, dist)         #find the distances of the k nearest neighbozrs
	novelty_check = np.mean(kNN) > novelty_threshold 
	
	return novelty_check


#all the functions for curiosity-based exploration follow
#======================================================================================================================================================

def create_icm_model(input_shape=(4,), num_actions=2):
    #function to create the NN needed for curiosity-based exploration
    #returns the icm model neural network
    #input_shape: number of neurons in the input layer
    #num_actions: number of neurons in the output layer
    
    #4 layers to get state feature map
    input_st = Input(shape=input_shape, name='input_st')
    dense_st = Dense(24, activation='relu')(input_st)
    dense_st = Dense(12, activation='relu')(dense_st)
    dense_ft = Dense(4, activation='linear', name='freature_ft')(dense_st)
    
    input_st1 = Input(shape=input_shape, name='input_st1')
    dense_st1 = Dense(24, activation='relu')(input_st1)
    dense_st1 = Dense(12, activation='relu')(dense_st1)
    dense_ft1 = Dense(4, activation='linear', name='feature_ft1')(dense_st1)
    
    merged_f = concatenate([dense_ft, dense_ft1]) #merge 2 feature maps to predict action
    
    #2 layers to get action
    dense_atp = Dense(24, activation='relu')(merged_f)
    dense_atp = Dense(num_actions, activation='softmax')(dense_atp) #change to sigmoid for tunning
    
    input_at = Input(shape=num_actions, name='input_at')
    
    merged_af = concatenate([dense_ft, input_at]) #merge action and current state feature map
    
    #predict the next state feature map
    dense_ftv = Dense(24, activation='relu')(merged_af)
    dense_ftv = Dense(input_shape[0], activation='linear')(dense_ftv)
    
    #calculate loss/intrinsic reward
    int_reward=Lambda(lambda x: 0.5*K.sum(K.square(x[0] - x[1]), axis=-1),
                      output_shape=(1,),
                      name="reward_intrinsic")([dense_ftv, dense_ft1])
    
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy() #loss for action prediction

    optimizer = tf.keras.optimizers.Adam()
    
    model = Model([input_st, input_st1, input_at], [int_reward, dense_atp])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=['mse', cross_entropy_loss])

    return model

def one_hot_encode_action(action):
	#function to convert the action to a one hot encode format
    
    action_encoded=np.zeros((2,), np.float32)
    action_encoded[action]=1
    return action_encoded

def get_intrinsic_reward(model, x):
    #get the intrinsic reward for curiosity exploration
    #x -> [prev_state, state, action]
    
    return K.function([model.get_layer("input_st").input,
                       model.get_layer("input_st1").input,
                       model.get_layer("input_at").input],
                      [model.get_layer("reward_intrinsic").output])(x)[0]

def learn(model, prev_states, states, actions):
    #train the curiosity based ICM network
    s_t=prev_states
    s_t1=states
    actions = actions
    actions = np.array(actions).reshape((2, 1))
    icm_loss=model.fit([np.array([s_t]), 
                        np.array([s_t1]),
                        np.array([actions])],
                       [np.zeros((1,)), np.array([actions])])


#======================================================================================================================================================

    



def simple_DQN(model, target_model, a, s, s_next, target, state_size, action_size, r, gamma):
	#function to perform the DQN algorithm without a replay buffer
	#returns the updated model and new state
	#model: the DQN model
	#target_model: the DQN model to provide the target Q-value for training
	#a: selected actio
	#s: current state
	#s_next: next state
	#target: boolean which indicates wether we use a target network or not for the target Q-values
	#state_size: size of state
	#action_size: number of possible actions in a state
	#r: reward received from action a in state s
	#gamma: discount factor

	#Look up the Q-value estimate of the next state using the NN
	if target:
		Q_sa_next = target_model.predict(s_next.reshape(1, state_size))[0]
	else:
		Q_sa_next = model.predict(s_next.reshape(1, state_size))[0]

	#How the Q-value of the state s and action a should be modified
	Q_sa_target = r + gamma * np.amax(Q_sa_next)

	#What the network thinks the Q-value for state s and action a is
	Q_sa = model.predict(s.reshape(1, state_size))[0]

	#Update the Q_sa for state s and action a
	Q_sa[a] = Q_sa_target

	#update the Network now
	model.fit(s.reshape(1, state_size), Q_sa.reshape(1, action_size), verbose=0)

	s = s_next

	return model, s



def DQN_replay_buffer(model, target_model, memory, batch_size, gamma, target, action_size):
	#function to perform the DQN algorithm without a replay buffer
	#returns the updated model 
	#model: the DQN model
	#target_model: the DQN model to provide the target Q-value for training
	#memory: an array including all the previously visited states, actions taken, rewards obtained, consecutive states, and wether the state is terminal or not
	#batch_size: size of the batch sampled from the replay buffer
	#gamme: discount factor
	#target: boolean which indicates wether we use a target network or not for the target Q-values
	#action_size: number of possible actions in a state


	minibatch = random.sample(memory, batch_size)
	s = np.array(minibatch)[:, 0]
	a = list(np.array(minibatch)[:, 1])   #has to be a list because we will use it as indexing
	r = np.array(minibatch)[:, 2]
	s_next = np.array(minibatch)[:, 3]
	done = list(np.array(minibatch)[:, 4]) #has to be a list because we will use it as indexing

	s, s_next = np.vstack(s), np.vstack(s_next)

	#Look up the Q-value estimate of the next states using the NN
	if target:
		Q_sa_next = target_model.predict(s_next)
	else:
		Q_sa_next = model.predict(s_next)

	Q_sa_next[done] = np.zeros(action_size) #these are Q-values for terminal states, you want them to remain 0

	#How the Q-value of the state s and action a should be modified
	Q_sa_target = r + gamma * np.amax(Q_sa_next, axis = 1)

	#What the network thinks the Q-value for state s and action a is
	Q_sa = model.predict(s)

	Q_sa[np.arange(Q_sa.shape[0]), a] = Q_sa_target


	#Update the model on the minibatch
	model.fit(s, Q_sa, verbose = 0)

	return model



def run(path='DQN_Target/', file_model='cartpole.h5', file_list='reward_list_cp.npy', gamma=0.95, 
		epsilon=1.0, epsilon_min=1E-5, epsilon_decay=0.98, episodes=400, batch_size=64, target_update_frequency=10, c=1, policy = 'egreedy',
		replay_buffer=True, target=True, memory_length=2000, novelty_exp=False, novelty_threshold = 0.5, k = 15, curiosity=True, cartpole_env=True,
		nhidden_layers = 3, neurons=[512, 256, 64], act_hidden = 'relu', act_output='linear', opt='adam'):
	#main function which runs to train the network 
	#path: path to save the list of rewards obtained per epsiodes and final neural network
	#file_model: name of model file to save
	#file_list: name of reward list file to save
	#gamma: discount factor
	#epsilon: initial epsilon in epsilon-greedy exploration
	#epsilon_min: minimum value to which epsilon in epsilon-greedy can decay
	#epsilon_decay: decay constant (tau in report) for epsilon-greedy
	#epsiodes: number of episodes for which we train the agent
	#batch_size: size of the batch sampled from the replay buffer
	#target_update_frequency: number of updates needed (i.e actions taken) to update the taregt neural network
	#c: confidence parameter for UCB exploration
	#policy: exploration policy to use (options are egreedy or ucb)
	#replay_buffer: boolean which determines wether a replay buffer is used for the DQN learning
	#target: boolean which determines wether a target neural network is used for the DQN leaning
	#memory_lenght: maximum size of the replay buffer
	#novelty_exp: boolean to determine wether we do novelty-based exploration
	#novelty_threshold: minimum value needed for a state to be regarded as novel
	#k: number of k-nearest neighbours that are considered for the novelty metric
	#curiosity: boolean which determines if we used curiosity-based exploration
	#cartploe_env: boolean which determines wether we use the cartpole environment or the mountaincar environment 
	#nhidden_layers: number of hidden layers
	#neurons: list of number of neurons for each hidden layer
	#act_hidden: the activation layer of each hidden layer
	#act_output: activation function of the output layer
	#opt: optimizer for training


	if cartpole_env:
		# Create the CartPole Environment
		env = gym.make('CartPole-v1')
		state_size = env.observation_space.shape[0]  #state is defined by 4 parameters
		action_size = env.action_space.n             #2 actions are possible

	else:
		# Create the MountainCar Environment
		env = gym.make('MountainCar-v0')
		state_size = env.observation_space.shape[0]  #state is defined by 2 parameters
		action_size = env.action_space.n             #3 discrete actions are possible



	#create the Q-Network
	model = create_DQN(state_size, action_size, nhidden_layers, neurons, 
						act_hidden, act_output, 'mse', opt)
	#create icm model
	icm_model = create_icm_model()
	# Define a target Network
	target_model = clone_model(model)
	target_model.set_weights(model.get_weights())

	reward_list = []  #stores the total reward per episode
	i = 0   #to keep track how many actions have been taken 
	archive_state = [env.reset()[0]]  #list of states that have been visited (needed for the novelity based exploration)
	memory = deque(maxlen = memory_length)   #the replay buffer
	N_a = np.zeros(action_size, dtype=int) # N(a) = number of times each action has been selected


	# Training
	for episode in range(episodes):
		s = env.reset()[0]
		done = False
		reward = 0
		game_reward = 0
		
		while not done:
			i += 1
			#action selection
			a = select_action(s, env, model, i, N_a, policy, epsilon, c, action_size)

			# Increment the action count
			N_a[a] += 1 

			#Do the action
			s_next, r, done, _, _ = env.step(a)

			reward += 1  #this will be the extrinsic reward only/number of actions taken in an epsiode (r will also include intrinsic reward if it exists)

			#part for novelty exploration based method
			if novelty_exp:
				novelty_check = novelty_exploration(s_next, archive_state, novelty_threshold, k)
				archive_state.append(s_next)
				if novelty_check: #i.e if s_next is a novel state so we add a bonus reward
					r += 1
            
			#part for curiosity exploration based method
			if curiosity:
				act=one_hot_encode_action(a)
                
				learn(icm_model, prev_states=s, states=s_next, actions=act)
            
				int_r_state=np.reshape(s, (1,4))
				int_r_next_state=np.reshape(s_next, (1,4))
				int_r_action=np.reshape(act, (1,2))
                
				int_reward=get_intrinsic_reward(icm_model, [np.array(int_r_state),
                                                 np.array(int_r_next_state),
                                                 np.array(int_r_action)])
                
				tot_r=0.2*int_reward+r
				game_reward+=tot_r


			#network training occurs here
			if replay_buffer:
				#Storing the experience in replay buffer
				if curiosity:
					memory.append((s, a, tot_r, s_next, done))
				else:
					memory.append((s, a, r, s_next, done)) 
				s = s_next
				if len(memory) >= batch_size:
					#update the Network on a random batch of the replay buffer
					model = DQN_replay_buffer(model, target_model, memory, batch_size, gamma, target, action_size)

			else:
				if curiosity:
					model, s = simple_DQN(model, target_model, a, s, s_next, target, state_size, action_size, tot_r, gamma)
				else:
					model, s = simple_DQN(model, target_model, a, s, s_next, target, state_size, action_size, r, gamma)
			
			
			#check if we won the game (we pre terminate an episode a game)
			if reward >= 200:
			    done = True    
			

			if done:
				print("Episode {}: Score = {}, Exploration rate = {:.2f}".format(episode, reward, epsilon))
				reward_list.append(reward)
				if reward >= 200:
					model.save(path + file_model)

		
			#Update target network
			if i % target_update_frequency == 0:
				target_model.set_weights(model.get_weights())
		
		#Decay exploration rate
		epsilon *= epsilon_decay
		epsilon = max(epsilon, epsilon_min)

	np.save(path + file_list, np.array(reward_list))



if __name__ == "__main__":

	path, gamma, epsilon, epsilon_min, epsilon_decay, episodes, batch_size, target_update_frequency, c, policy,\
	replay_buffer, target, memory_length, novelty_exp, novelty_threshold, k, curiosity, cartpole_env, nhidden_layers, \
	neurons, act_hidden, act_output, opt = 'DQN/', 0.95, 1.0, 1E-5, 0.98, 750, 64, 10, 1, 'egreedy', True, True, \
	2000, False, 0.5, 15, False, True, 3, [512, 256, 64], 'relu', 'linear', 'adam'


	for j in range(1):
		file_model = 'cartpole_{}.h5'.format(j)
		file_list = 'reward_list_{}.npy'.format(j)
		run(path, file_model, file_list, gamma, epsilon, epsilon_min, epsilon_decay, episodes, batch_size, target_update_frequency, c, \
			policy, replay_buffer, target, memory_length, novelty_exp, novelty_threshold, k, curiosity, cartpole_env, nhidden_layers, neurons, \
			act_hidden, act_output, opt)
		
