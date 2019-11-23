import numpy as np
import tqdm
import operator
import pickle
import os
import concurrent.futures

from tournament_ranking import TournamentRanking

num_teams_range = np.arange(11,30,3) 
# num_teams_range = [20]
num_expts = 1000
print("num_teams_range", num_teams_range)
average_time = []
mistake_interval = 40
# average_correct_fraction_list = []
num_mistake_points = 20
epsilon_values = [0.1]
delta = 0.1
mistake_array = [0]*num_mistake_points
filename_template = "./sim2/epsilon_{}_teams_{}_expt_{}.pkl"

def run_expt(expt_info):
    num_teams, epsilon, expt_id = expt_info
    curr_filename = filename_template.format(str(epsilon).replace(".", "p"), num_teams, expt_id)
    if not os.path.exists(curr_filename):
        np.random.seed(expt_id)
        timesteps = []
        topK = int(0.3*num_teams)
        win_probs = np.zeros((num_teams,num_teams))

        for i in range(num_teams):
            for j in range(i+1,num_teams):
                p_ij = np.random.beta((i+1)**2, (j+1)**2)
                win_probs[i,j] = p_ij
                win_probs[j,i] = 1 - p_ij

        curr_tournament = TournamentRanking(num_teams, win_probs)
        t, arms, correct_fraction_list = curr_tournament.KL_LUCB(topK, epsilon=epsilon, delta=delta)
        pickle.dump(curr_tournament, open(curr_filename, "wb"))
    else:
        curr_tournament = pickle.load(open(curr_filename, "rb"))
        t, arms, correct_fraction_list = curr_tournament.t, curr_tournament.high, curr_tournament.correct_fraction_list
    return t, arms, correct_fraction_list

for epsilon in epsilon_values:
    for num_teams in num_teams_range:
        print("num_teams:", num_teams)
        time_array = []
        EXPTS = list(zip([num_teams]*num_expts, [epsilon]*num_expts, np.arange(num_expts)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            for expt_info, output in zip(EXPTS, executor.map(run_expt, EXPTS)):
                t, arms, correct_fraction_list = output
                time_array.append(t)
                for i in range(num_mistake_points):
                    # mistake_array[i] += 1 - correct_fraction_list[mistake_interval*(i+1)]
                    pass
        average_time.append(np.mean(time_array))
        mistake_array = np.array(mistake_array)/num_expts

import matplotlib.pyplot as plt
plt.figure()
plt.ylabel("Time steps")
plt.xlabel("Number of teams")
plt.plot(num_teams_range, average_time)
plt.xticks(num_teams_range, num_teams_range)
plt.show()

# plt.figure()
# plt.ylabel("Mistake Prob")
# plt.xlabel("Samples")
# plt.plot(np.arange(mistake_interval,mistake_interval*num_mistake_points+1,mistake_interval), mistake_array)
# plt.xticks(np.arange(mistake_interval,mistake_interval*num_mistake_points+1,mistake_interval), 
           # np.arange(mistake_interval,mistake_interval*num_mistake_points+1,mistake_interval))
# plt.show()
