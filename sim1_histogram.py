import numpy as np
import tqdm
import operator
import pickle
import os
import concurrent.futures
import matplotlib.pyplot as plt
from tournament_ranking import TournamentRanking

# num_teams_range = np.arange(5,50,5) 
num_teams_range = [20]
num_expts = 1000
print("num_teams_range", num_teams_range)
average_time = []
mistake_interval = 40
# average_correct_fraction_list = []
num_mistake_points = 20
epsilon_values = [0.04, 0.1]
epsilon_colors = ["b", "r"]
delta = 0.1
mistake_array = [0]*num_mistake_points
filename_template = "./sim1/epsilon_{}_teams_{}_expt_{}.pkl"

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
        t, arms, correct_fraction_list = curr_tournament.LUCB(topK, epsilon=epsilon, delta=delta)
        pickle.dump(curr_tournament, open(curr_filename, "wb"))
    else:
        curr_tournament = pickle.load(open(curr_filename, "rb"))
        t, arms, correct_fraction_list = curr_tournament.t, curr_tournament.high, curr_tournament.correct_fraction_list
    return t, arms, correct_fraction_list

bin_size = 1000
plt.figure()
plt.ylabel("Fraction of runs")
plt.xlabel("Samples/{}".format(bin_size))

for epsilon, epsilon_color in zip(epsilon_values, epsilon_colors):
    print("epsilon:", epsilon)
    for num_teams in num_teams_range:
        print("num_teams:", num_teams)
        time_array = []
        time_histogram = [0]*(80000//bin_size)
        EXPTS = list(zip([num_teams]*num_expts, [epsilon]*num_expts, np.arange(num_expts)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            for expt_info, output in zip(EXPTS, executor.map(run_expt, EXPTS)):
                t, arms, correct_fraction_list = output
                time_array.append(t)
                time_histogram[int(np.round(t/bin_size))] += 1
                for i in range(num_mistake_points):
                    mistake_array[i] += 1 - correct_fraction_list[mistake_interval*(i+1)]
        
        average_time.append(np.mean(time_array))
        time_histogram = np.array(time_histogram)/np.sum(time_histogram)
        mistake_array = np.array(mistake_array)/num_expts
        plt.stem(time_histogram, 
                 markerfmt=" ", 
                 basefmt=" ", 
                 linefmt=epsilon_color, 
                 label="Epsilon: {}, Teams: {}".format(epsilon, num_teams))

plt.legend()
plt.show()
