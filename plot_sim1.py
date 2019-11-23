import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt

lucb = "LUCB_num_matches.txt"
kl_lucb = "KL_LUCB_num_matches.txt"

def avg_timesteps(filename):
	num_teams = len(list(range(5,105,5)))
	running_avg_timesteps = [0 for i in range(num_teams)]
	num_of_experiments = [0 for i in range(num_teams)]
	with open(filename) as f:
		for line in f:
			words = line.strip().split()
			if( not(words[0] == '--------------------------------------') ):
				num_timesteps = float(words[7])
				index = int(int(words[4])/5) - 1
				running_avg_timesteps[index] = (running_avg_timesteps[index]*num_of_experiments[index])/(num_of_experiments[index]+1) \
											  + num_timesteps/(num_of_experiments[index]+1)
				num_of_experiments[index] += 1
		f.close()
	return running_avg_timesteps

plt1 = avg_timesteps(lucb)
plt2 = avg_timesteps(kl_lucb)

plt.figure()
plt.xlabel('Number of Teams')
plt.ylabel('Number of Matches(Timesteps)')
plt.plot(list(range(5,105,5)),plt1,label='LUCB')
plt.plot(list(range(5,105,5)),plt2,label='KL-LUCB')
plt.legend()
plt.show()
