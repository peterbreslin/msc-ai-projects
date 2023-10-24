import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


#defining some matplot stuff for the standard font style
rfont = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rfont)
plt.rcParams["legend.labelspacing"] = 0.001
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

path = '/Users/louissiebenaler/Dropbox/Private_7aler/Louis/University Material/Leiden University/Reinforcement Learning/Assignments/RL_A3/ALICE/learning_rate/new_new/'

n_values = [0.01, 0.001, 0.0001]
max_len = 30001

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

fig, ax = plt.subplots(1, 1, figsize = (8, 6))


for col, n in enumerate(n_values):
    s_tot = np.zeros(max_len) + 35.
    for i in range(5):
        file = 'score_{}_{}.npy'.format(n, i)
        s = np.load(path + file)
        
        if i == 0:
            length = len(s)
            s_tot[:len(s)] = s
        elif i == 1:
            length_new = len(s)
            if length_new > length:
                length = length_new
                
    
            tmp = np.zeros(max_len) + 35.
            tmp[:len(s)] = s
            s_tot = np.concatenate([[s_tot], [tmp]], axis=0)
        else:
            length_new = len(s)
            if length_new > length:
                length = length_new
                
            tmp = np.zeros(max_len) + 35.
            tmp[:len(s)] = s
            s_tot = np.concatenate([s_tot, [tmp]], axis=0)
        
    
    r_mean = s_tot.mean(axis = 0)
    
    r = savgol_filter(r_mean[:length], 301, 1)
    
    std = (np.std(s_tot, axis = 0)) / np.sqrt(5)
    
    std = std[:length]
    
    
    ax.fill_between(np.arange(len(r)), r - std, r + std, color = colors[col], alpha = 0.2)
    ax.plot(np.arange(len(r)), r, label = r'$\alpha = {}$'.format(n))
    
ax.set_ylabel('Score', fontsize = 17)
ax.set_xlabel('Epsiode', fontsize  = 17)


ax.plot([0, 10000], [35, 35], color = 'black', ls = '--', label = 'Optimal policy')
ax.set_xlim(0, 4400)
ax.legend(frameon = True, fontsize = 12)

ax.minorticks_on()
ax.tick_params(which = 'both', bottom = True, top = False, left = True, right = False)
ax.tick_params(which = 'major', length = 10, direction = 'in', labelsize = 14)
ax.tick_params(which = 'minor', length = 2, direction = 'in', labelsize = 14)



#plt.savefig('figures/learning_rate_tuning.pdf') 