import numpy as np
import tqdm
import operator

class TournamentRanking:
    def __init__(self, num_teams, win_probs):
        self.num_teams = num_teams
        self.win_probs = win_probs

        self.score = np.sum(self.win_probs, axis=1)/(self.num_teams - 1)
        self.pulls = np.zeros(num_teams)
        self.wins = np.zeros(num_teams)
        self.empirical_score = dict([(i,0) for i in range(self.num_teams)])

    def update_arm(self, arm):
        opponent = np.random.choice(np.concatenate((np.arange(arm-1),np.arange(arm+1,self.num_teams))))
        self.pulls[arm] += 1
        self.wins[arm] += np.random.binomial(n=1,p=self.win_probs[arm][opponent])
        self.empirical_score[arm] = self.wins[arm]/self.pulls[arm]

    def boundfn(self, arm, t, delta):
        k1 = 5/4
        # bound = np.sqrt((self.num_teams-1)*np.log(k1*self.num_teams*(t**4)/delta)/(2*self.pulls[arm]))
        bound = np.sqrt(np.log(k1*self.num_teams*(t**4)/delta)/(2*self.pulls[arm]))
        return bound

    def correct_fraction(self, topK_predicted, topK_epsilon, topK):
        topK_predicted = set([team[0] for team in topK_predicted])
        cf = len(topK_predicted & topK_epsilon)/topK
        return cf

    def LUCB(self, K, epsilon=0.1, delta=0.1, print_every=1000,horizon=float('inf')):
        for i in range(self.num_teams):
            self.update_arm(i)

        actual_scores = sorted([x for x in enumerate(self.score)], key=operator.itemgetter(1), reverse=True)
        kth_score = actual_scores[K-1][1]
        topK_epsilon = set([x[0] for x in actual_scores if (kth_score-x[1]<epsilon)])
        correct_fraction_list = []
        t = self.num_teams
        while True:
            sorted_empirical_scores = sorted(self.empirical_score.items(), key=operator.itemgetter(1), reverse=True)
            high = sorted_empirical_scores[:K]
            low = sorted_empirical_scores[K:]

            h_min = float('inf')
            h_star = -1
            l_max = float('-inf')
            l_star = -1
            for i in range(K):
                bound = self.boundfn(high[i][0], t, delta)
                if high[i][1] - bound  < h_min:
                    h_min = high[i][1] - bound
                    h_star = high[i][0]


            for j in range(K, self.num_teams):
                bound = self.boundfn(low[j-K][0], t, delta)
                if low[j-K][1] + bound > l_max:
                    l_max = low[j-K][1] + bound
                    l_star = low[j-K][0]

            correct_fraction_list.append(self.correct_fraction(high,topK_epsilon, K))

            if l_max - h_min < epsilon:
                break

            self.update_arm(h_star)
            self.update_arm(l_star)

            t += 1
            if t >= horizon:
                break
            # if t % print_every == 0:
                # print("=====================================================")
                # print("l_max : " + str(l_max))
                # print("h_min : " + str(h_min))
                # print("l_star : " + str(l_star))
                # print("h_star : " + str(h_star))
                # print("l_star bound function : " + str(self.boundfn(l_star, t, delta)))
                # print("h_star bound function : " + str(self.boundfn(h_star, t, delta)))
                # print("Empirical scores....")
                # print(self.empirical_score)
        self.t, self.high, self.correct_fraction_list = t, high, correct_fraction_list
        return t, high, correct_fraction_list

import pickle
import os
import concurrent.futures
import matplotlib.pyplot as plt

# num_teams_range = np.arange(5,50,5)
num_teams_range = [100]
num_expts = 100
print("num_teams_range", num_teams_range)
average_time = []
mistake_interval = 40
# average_correct_fraction_list = []
num_mistake_points = 20
epsilon = 0.1
delta = 0.1
mistake_array = [0]*num_mistake_points
filename_template = "./sim1/topK_{}_teams_{}_expt_{}.pkl"
horizon = 100
def run_expt(expt_info):
    num_teams, topK, expt_id = expt_info
    curr_filename = filename_template.format(topK, num_teams, expt_id)
    if not os.path.exists(curr_filename):
        np.random.seed(expt_id)
        timesteps = []
        win_probs = np.zeros((num_teams,num_teams))

        for i in range(num_teams):
            for j in range(i+1,num_teams):
                p_ij = np.random.beta((i+1)**2, (j+1)**2)
                win_probs[i,j] = p_ij
                win_probs[j,i] = 1 - p_ij

        curr_tournament = TournamentRanking(num_teams, win_probs)
        t, arms, correct_fraction_list = curr_tournament.LUCB(topK, epsilon=epsilon, delta=delta, horizon=horizon)
        pickle.dump(curr_tournament, open(curr_filename, "wb"))
    else:
        curr_tournament = pickle.load(open(curr_filename, "rb"))
        t, arms, correct_fraction_list = curr_tournament.t, curr_tournament.high, curr_tournament.correct_fraction_list
    return t, arms, correct_fraction_list

plt.figure()
plt.ylabel("K")
plt.xlabel("Empirical mistake probability after {} samples".format(horizon))

topK_vals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
mistake_array = [0.0]*14

for topK in topK_vals:
    idx = 0
    print("K : ", topK)
    for num_teams in num_teams_range:
        print("num_teams : ", num_teams)
        EXPTS = list(zip([num_teams]*num_expts, [topK]*num_expts, np.arange(num_expts)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for expt_info, output in zip(EXPTS, executor.map(run_expt, EXPTS)):
                t, arms, correct_fraction_list = output
                mistake_array[idx] += (1 - correct_fraction_list[-1])/num_expts
    idx = idx + 1

plt.plot(topK_vals,mistake_array)
plt.show()
