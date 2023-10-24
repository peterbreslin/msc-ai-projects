import pandas as pd
import numpy as np
from itertools import cycle
import time



def uv_decomposition(set_train, set_test, path, name_end, df, del_col, sets, init, order, d):
	#in this function we perform the UV decomposition algorithm. The meaning of the variables are explained at length in the 
	#in the jupyter notebook. Before performing the algorithm we normalize the utility matrix and initialize U and V.


	#generate first the training and test utility matrix R and R_test. Missing ratings are defined as 0 values
	R, R_test = ut_matrices(set_train, set_test, df, del_col, sets)


	#we now compute the user rating averages, movie rating averages and global average as this is needed to normalize R and R_test
	#================================================================================================================================
	clm_sums = np.sum(R, axis = 0) #we sum together colums
	clm_n0_count =  np.count_nonzero(R, axis = 0) #number of non-zero ratings along each column
	i_miss = np.where(clm_sums == 0) #movies that have completely dissapeard in the training set
	av = np.sum(R) / np.count_nonzero(R)   #the gloabel average (used as a the fallback value) 
	clm_sums[i_miss] = av              #employ fallback on empty columns
	clm_n0_count[i_miss] = 1

	av_item = clm_sums / clm_n0_count  #average rating for each item

	row_sums = np.sum(R, axis = 1) #we sum together rows
	row_n0_count =  np.count_nonzero(R, axis = 1) #number of non-zero ratings along each column
	i_miss = np.where(row_sums == 0) #movies that have completely dissapeard in the training set
	row_sums[i_miss] = av    #employ fallback on empty rows
	row_n0_count[i_miss] = 1

	av_user = row_sums / row_n0_count  #average rating for each item

	i, j = np.where(R != 0)     #find the indices of non-missing ratings in the training data
	i_test, j_test = np.where(R_test != 0)   #find the indices of non-missing ratings in the test data
	N = len(R[i, j])           #number of known ratings in the training set 
	N_test = len(R_test[i_test, j_test])   #number of known ratings in the test set 
	#================================================================================================================================


	#We now initialize the U and V matrix. Also normalize R if we demanded so before
	#================================================================================================================================
	if init == 'normalize':  #we normalize the utility matrix and U and V have initial weights of 1E-3
		R_alg = np.copy(R)
		R_alg[i, j] = R_alg[i, j] - av_user[i] / 2 - av_item[j] / 2  #R has now been normalized
		R_alg_test= np.copy(R_test)
		R_alg_test[i_test, j_test] = R_alg_test[i_test, j_test] - av_user[i_test] / 2 - av_item[j_test] / 2 #R_test has now been normalized

		U = np.zeros((6040, d)) + 1E-3
		V = np.zeros((d, 3706)) + 1E-3
	if init == 'perturbe': #we normalize the utility matrix and U and V are perturbed
		R_alg = np.copy(R)
		R_alg[i, j] = R_alg[i, j] - av_user[i] / 2 - av_item[j] / 2  #R has now been normalized
		R_alg_test= np.copy(R_test)
		R_alg_test[i_test, j_test] = R_alg_test[i_test, j_test] - av_user[i_test] / 2 - av_item[j_test] / 2 #R_test has now been normalized

		np.random.seed(44)
		U = np.zeros((6040, d))  + np.random.normal(loc=0.0, scale=0.1, size=(6040, d))
		V = np.zeros((d, 3706))  + np.random.normal(loc=0.0, scale=0.1, size= (d, 3706))
	elif init == 'standard': #we do not normalize the utility matrix 
		R_alg, R_alg_test = np.copy(R), np.copy(R_test) 

		U = np.zeros((6040, d)) + np.sqrt(av / 2)
		V = np.zeros((d, 3706)) + np.sqrt(av / 2)

	#R_alg and R_alg_test are the matrices used during the algorithm
	#================================================================================================================================


	#We are defining the optimization path now 
	#================================================================================================================================
	idx_u, idx_v = index_array(d) #these arrays will include the coordindates of all the elements of U and V in an ordered way

	np.random.seed(44)
	if order == 'random':  #if we want a random optimization path, we permute the arrays of coordinates
		idx_u, idx_v = np.random.permutation(idx_u), np.random.permutation(idx_v)
	#================================================================================================================================

	

	#UV decomposition algorithm begins now. The names of the elements (ex. vsj, mrj) are explained in the report
	#================================================================================================================================
	RMSE, MAS = [], [] #lists for RMSE and MAS on training set
	RMSE_test, MAS_test = [], [] #lists for RMSE and MAS on test set

	terminate = False
	start_time = time.time()
	iteration_u = 0
	iteration_v = 0
	iteration = 0
	while not terminate: #we stop the algorithm after the RMSE on the test set has converged

		#update the element u_rs from the U matrix
		r, s = idx_u[iteration_u][0], idx_u[iteration_u][1]
		j0 = np.where(R_alg[r, :] != 0)[0]   #column where there is not a missing value in the utility matrix
		if j0.size != 0:
			vsj = V[s, :][j0]
			mrj = R_alg[r, :][j0]
			vkj = np.delete(V[:, j0], s, axis = 0) 
			urk = np.array([np.delete(U[r, :], s)])   #we don't want row s
			p = mrj - np.matmul(urk, vkj)[0]
			x = np.matmul(vsj, p) / np.matmul(vsj, vsj)  #
			U[r, s] = x
	        
	    #update the element V_sr from the V matrix
		s, r = idx_v[iteration_v][0], idx_v[iteration_v][1]
		i0 = np.where(R_alg[:, s] != 0)[0]    #row where there is not a missing value in the utility matrix
		if i0.size != 0:
			uir = U[:, r][i0]
			mis = R_alg[:, s][i0]
			uik = np.delete(U[i0, :], r, axis = 1)
			vks = np.array([np.delete(V[:, s], r)]) #we don't want column r
			p = mis - np.matmul(uik, vks.T).T[0]
			y = np.matmul(uir, p) / np.matmul(uir, uir)
			V[r, s] = y
	               

		#we compute the MSE and MAS only every 200 iterations because it is time consumming 
		if iteration % 200 == 0:
			#calculate rmse
			rmse, mas = frmse(R, U, V, i, j, N, av_user, av_item)
			RMSE.append(rmse)
			MAS.append(mas)
			rmse_test, mas_test = frmse(R_test, U, V, i_test, j_test, N_test, av_user, av_item)
			RMSE_test.append(rmse_test)
			MAS_test.append(mas_test)


		if iteration > 6000:   #check for convergence only after 6000 iterations, the algorithm does not always learn straight-away
			if (RMSE_test[-1] >= RMSE_test[-2]) and (RMSE_test[-1] >= RMSE_test[-3]):
				terminate = True 

		
		#ensures that we keep looping through U and V
		if (iteration_u + 1) % (idx_u.shape[0]) != 0:
			iteration_u += 1
		else:
			iteration_u == 0
		if (iteration_v + 1) % (idx_v.shape[0]) != 0:
			iteration_v += 1
		else:
			iteration_v == 0
		iteration += 1
	#================================================================================================================================
	#algorithm is finished we now save everything of interest

	print("--- %s seconds ---" % (time.time() - start_time))

	
	np.save(path + '/U_{}.npy'.format(name_end), U)
	np.save(path + '/V_{}.npy'.format(name_end), V)
	np.save(path + '/RMSE_{}.npy'.format(name_end), RMSE)
	np.save(path + '/MAS_{}.npy'.format(name_end), MAS)

	np.save(path + '/R_{}.npy'.format(name_end), R)
	np.save(path + '/Rtest_{}.npy'.format(name_end), R_test)

	np.save(path + '/RMSE_test_{}.npy'.format(name_end), RMSE_test)
	np.save(path + '/MAS_test_{}.npy'.format(name_end), MAS_test)
	
	#compute RMSE and MAS on training  set a final time
	P = np.matmul(U, V)
	if (init == 'normalize') or (init == 'perturbe'):
		P[i, j] = P[i, j] + av_user[i] / 2 + av_item[j] / 2   #this line takes time
	P[P<1], P[P>5] = 1, 5                                 #this line takes time (so track rmse on normalized data only)
	err = np.sum((R[i, j] - P[i, j])**2)   #index i because you are looking at movies (i is the y-axis)
	rmse = np.sqrt(err / N)
	err_abs = abs(R[i, j] - P[i, j])
	mas = np.sum(err_abs) / N
	print('RMSE and MAS on the training set:', rmse, mas)

   	#compute RMSE and MAS on test set a final time
	P = np.matmul(U, V)
	if init == 'normalize' or init == 'perturbe':
		P[i_test, j_test] = P[i_test, j_test] + av_user[i_test] / 2 + av_item[j_test] / 2   #this line takes time
	P[P<1], P[P>5] = 1, 5                                 #this line takes time (so track rmse on normalized data only)
	err = np.sum((R_test[i_test, j_test] - P[i_test, j_test])**2)   #index i because you are looking at movies (i is the y-axis)
	rmse = np.sqrt(err / N_test)
	err_abs = abs(R_test[i_test, j_test] - P[i_test, j_test])
	mas = np.sum(err_abs) / N_test
	print('RMSE and MAS on the test set:', rmse, mas)







def ut_matrices(sets_index, set_index, df, del_col, sets):
    #function for generating the training utility matrix and test utility matrix
    #sets_index: indicates which 4 sets are included in the training matrix
    #set_index: indicates which set is used for the testing
    
    x, y = np.amax(df['UserID']), np.amax(df['MovieID'])
    R = np.zeros((x, y))  #training data, missing rating are defined as 0

    for j in sets_index:  #we now fill the utility matrix with the training data
        U, M, RA = df['UserID'].iloc[sets['set{}'.format(j)]], df['MovieID'].iloc[sets['set{}'.format(j)]], df['Rating'].iloc[sets['set{}'.format(j)]]
        for u, m, ra in zip(U, M, RA):
            R[u - 1, m - 1] = ra #-1 is needed because the smallest ID is 1
    
    R = np.delete(R, del_col, axis = 1)  #get rid of these completely empty columns, these movies have no ratings
                
    R_test = np.zeros((x, y))   #test data, missing rating are defined as 0
    U, M, RA = df['UserID'].iloc[sets['set{}'.format(set_index)]], df['MovieID'].iloc[sets['set{}'.format(set_index)]], df['Rating'].iloc[sets['set{}'.format(set_index)]]
    for u, m, ra in zip(U, M, RA):
        R_test[u - 1, m - 1] = ra #-1 is needed because the smallest ID is 1
    
    R_test = np.delete(R_test, del_col, axis = 1) #get rid of these completely empty columns, these movies have no ratings
    
    return R, R_test



def frmse(R, U, V, i, j, N, av_user, av_item):
	#i, j: the indiced of the non-zero elements of R, these are the ones we want to predict
	#N: the number of non-zero elements in R
    #function to compute the rmse and mas
    P = np.matmul(U, V)  #compute the prediction
    P[i, j] = P[i, j] + av_user[i] / 2 + av_item[j] / 2   #unnormalize the prediction
    P[P<1], P[P>5] = 1, 5                                 #prediction <1 are set to 1 and prediction >5 are set to 5
    err = np.sum((R[i, j] - P[i, j])**2)   
    rmse = np.sqrt(err / N)       #the rmse
    err_abs = abs(R[i, j] - P[i, j])
    mas = np.sum(err_abs) / N     #the mas
    return rmse, mas


def index_array(d):
    #function to generate arrays for the indices of the U and V matrix, so we can loop through it during the optimization algorithm
    idx_u = np.zeros((d*6040, 2))    
    for i in range(d):
        idx_u[6040*i:6040*(i + 1), 1] += i
        idx_u[6040*i:6040*(i + 1), 0] = np.arange(0, 6040, 1)
    idx_v = np.zeros((d*3706, 2))
    for i in range(d):
        idx_v[3706*i:3706*(i + 1), 1] += i
        idx_v[3706*i:3706*(i + 1), 0] = np.arange(0, 3706, 1)
    
    return idx_u.astype(int), idx_v.astype(int)

