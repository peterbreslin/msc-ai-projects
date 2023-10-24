import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

rfont = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rfont)
plt.rcParams["legend.labelspacing"] = 0.001
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def get_data(path, n, ep, filter_size = 101, clip = False):
    r = np.zeros(shape = (n, ep))
    
    for i in range(n):
        tmp = np.load(path + '/reward_list_{}.npy'.format(i))
        if clip:
            tmp = np.clip(tmp, 0, 200)
        r[i] = tmp
    
    
    r = savgol_filter(r, filter_size, 1)
    std = (np.std(r, axis = 0))
    r_tot = r.sum(axis = 0)
    r_tot = r_tot / n
    
    return r_tot, std



# activation function
path = 'activation_functions/'
paths = ['elu', 'relu', 'softplus', 'linear' ]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
n, ep = 10, 750
for i in range(len(paths)):
    r, std = get_data(path + paths[i], n, ep, 101)
    
    plt.fill_between(np.arange(ep), r - std, r + std, color = colors[i], alpha = 0.4)
    plt.plot(np.arange(ep), r, color = colors[i], label = paths[i])
    

plt.legend(frameon = False, fontsize = 12)
plt.ylabel('Reward', fontsize = 12)
plt.xlabel('Episode', fontsize = 12)
plt.tick_params(axis='both', labelsize = 12)
plt.xlim(0, 750)
plt.title('Activation Functions', fontsize = 12)

#plt.savefig('Figures/activation_functions.pdf')
plt.show()