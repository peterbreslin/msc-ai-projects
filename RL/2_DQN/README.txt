Notes on how to run our experiments:

The python script DRL_A2.py needs to be executed to train a DQN. In this there is a function called run() with parameters defining all the properties of the training of the agent. The parameters are the following: 

	gamma: discount factor
	epsilon: initial epsilon in epsilon-greedy exploration
	epsilon_min: minimum value to which epsilon in epsilon-greedy can decay
	epsilon_decay: decay constant (tau in report) for epsilon-greedy
	epsiodes: number of episodes for which we train the agent
	batch_size: size of the batch sampled from the replay buffer
	target_update_frequency: number of updates needed (i.e actions taken) to update the taregt neural network
	c: confidence parameter for UCB exploration
	policy: exploration policy to use (options are egreedy or ucb)
	replay_buffer: boolean which determines wether a replay buffer is used for the DQN learning
	target: boolean which determines wether a target neural network is used for the DQN leaning
	memory_lenght: maximum size of the replay buffer
	novelty_exp: boolean to determine wether we do novelty-based exploration
	novelty_threshold: minimum value needed for a state to be regarded as novel
	k: number of k-nearest neighbours that are considered for the novelty metric
	curiosity: boolean which determines if we used curiosity-based exploration
	cartploe_env: boolean which determines wether we use the cartpole environment or the mountaincar environment 
	nhidden_layers: number of hidden layers
	neurons: list of number of neurons for each hidden layer
	act_hidden: the activation layer of each hidden layer
	act_output: activation function of the output layer
	opt: optimizer for training


Setting these parameters to the appropriate values enables to run all of our experiments. Our baseline model can be executed by setting:

gamma, epsilon, epsilon_min, epsilon_decay, episodes, batch_size, target_update_frequency, c, policy,\
replay_buffer, target, memory_length, novelty_exp, novelty_threshold, k, curiosity, cartpole_env, nhidden_layers, \
neurons, act_hidden, act_output, opt = 0.95, 1.0, 1E-5, 0.995, 750, 64, 10, 1, 'egreedy', True, True, \
	2000, False, 0.5, 15, False, True, 3, [512, 256, 64], 'relu', 'linear', 'adam'


You can define these parameters at the end of the code in if __name__ == "__main__"



The file data_analysis.py gives an example of how we plotted the learning curves.