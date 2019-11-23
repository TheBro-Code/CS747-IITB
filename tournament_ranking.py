import numpy as np
import operator

class TournamentRanking:
    def __init__(self, N, win_probs):
        self.N = N
        self.win_probs = win_probs

        self.score = np.sum(self.win_probs, axis=1)/(self.N - 1)
        self.pulls = np.zeros(N)
        self.wins = np.zeros(N)    
        self.empirical_score = dict([(i,0) for i in range(self.N)])

    def update_arm(self, arm):
        # for i in range(self.N):
              # if i != arm:
              #     res = np.random.binomial(1, self.win_probs[arm,i])
              #     self.wins[arm] += res
              #     self.pulls[arm] += 1
        # self.wins[arm] += np.sum(np.random.binomial(n=1,p=self.win_probs[arm],size=self.N))
        # self.pulls[arm] += (self.N-1)
        # self.empirical_score[arm] = self.wins[arm]/self.pulls[arm]
        opponent = np.random.choice(np.concatenate((np.arange(arm-1),np.arange(arm+1,self.N))))
        self.pulls[arm] += 1
        self.wins[arm] += np.random.binomial(n=1,p=self.win_probs[arm][opponent])
        self.empirical_score[arm] = self.wins[arm]/self.pulls[arm]

    def update_duo(self, arm1, arm2):
        res = np.random.binomial(1, self.win_probs[arm1,arm2])
        self.wins[arm1] += res
        self.wins[arm2] += (1-res)
        self.pulls[arm1] += 1
        self.pulls[arm2] += 1
        self.empirical_score[arm1] = self.wins[arm1]/self.pulls[arm1]
        self.empirical_score[arm2] = self.wins[arm2]/self.pulls[arm2]

    def boundfn(self, arm, t, delta):
        k1 = 5/4
        # bound = np.sqrt((self.N-1)*np.log(k1*self.N*(t**4)/delta)/(2*self.pulls[arm]))
        bound = np.sqrt(np.log(k1*self.N*(t**4)/delta)/(2*self.pulls[arm]))
        return bound

    def kl_exploration_rate(self, arm, t, delta):
        k1 = 9
        return np.log(k1*self.N*(t**4)/delta) + np.log(np.log(k1*self.N*(t**4)/delta))
    
    def KL_div(self, a, b):
        if a == 0: 
            if b == 1:
                return float("inf")
            else:
                return (1-a)*np.log((1-a)/(1-b))
        elif a == 1:
            if b == 0:
                return float("inf")
            else:
                return a*np.log(a/b)
        else:
            if b == 0 or b == 1:
                return float("inf")
            else:
                return a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b))

    def get_kl_bound(self, arm, t, delta, lower_limit, upper_limit, is_increasing):
        eps = 1e-3
        v1 = self.kl_exploration_rate(arm, t, delta)
        p = self.empirical_score[arm]
        lo = lower_limit
        hi = upper_limit
        q = (lo+hi)/2
        v2 = self.pulls[arm]*self.KL_div(p,q)
        while (np.abs(v1-v2) > eps) or (v1 < v2):
            if abs(hi-lo) < 1e-5:
                q = lo
                break
            if v2 > v1:
                if(is_increasing):
                  hi = q
                else:
                  lo = q
            else:
                if(is_increasing):
                  lo = q
                else:
                  hi = q
            q = (lo + hi)/2
            v2 = self.pulls[arm]*self.KL_div(p,q)
        return q

    def correct_fraction(self, topK_predicted, topK_epsilon, topK):
        topK_predicted = set([team[0] for team in topK_predicted])
        cf = len(topK_predicted & topK_epsilon)/topK
        return cf

    def LUCB(self, K, epsilon=0.1, delta=0.1, print_every=1000):
        for i in range(self.N):
            self.update_arm(i)

        actual_scores = sorted([x for x in enumerate(self.score)], key=operator.itemgetter(1), reverse=True)
        kth_score = actual_scores[K-1][1]
        topK_epsilon = set([x[0] for x in actual_scores if (kth_score-x[1]<epsilon)])
        correct_fraction_list = []
        t = self.N
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
            

            for j in range(K, self.N):
                bound = self.boundfn(low[j-K][0], t, delta)
                if low[j-K][1] + bound > l_max:
                    l_max = low[j-K][1] + bound
                    l_star = low[j-K][0]
            
            correct_fraction_list.append(self.correct_fraction(high,topK_epsilon, K))

            if l_max - h_min < epsilon:
                break

            # self.update_duo(l_star, h_star)
            self.update_arm(h_star)
            self.update_arm(l_star)

            t += 1
            
            # if t % print_every == 0:
            #     print("=====================================================")
            #     print("l_max : " + str(l_max))
            #     print("h_min : " + str(h_min))
            #     print("l_star : " + str(l_star))
            #     print("h_star : " + str(h_star))
            #     print("l_star bound function : " + str(self.boundfn(l_star, t, delta)))
            #     print("h_star bound function : " + str(self.boundfn(h_star, t, delta)))
            #     print("Empirical scores....")
            #     print(self.empirical_score)

        return t, high, correct_fraction_list

    def KL_LUCB(self, K, epsilon=0.1, delta=0.1, print_every=1000):
      for i in range(self.N):
        self.update_arm(i)

      actual_scores = sorted([x for x in enumerate(self.score)], key=operator.itemgetter(1), reverse=True)
      kth_score = actual_scores[K-1][1]
      topK_epsilon = set([x[0] for x in actual_scores if (kth_score-x[1]<epsilon)])
      correct_fraction_list = []
      t = self.N

      while True:
          sorted_empirical_scores = sorted(self.empirical_score.items(), key=operator.itemgetter(1), reverse=True)
          high = sorted_empirical_scores[:K]
          low = sorted_empirical_scores[K:]

          h_min = float('inf') 
          h_star = -1
          l_max = float('-inf')
          l_star = -1
          for i in range(K):
              lower_bound = self.get_kl_bound(high[i][0], t, delta, 0, self.empirical_score[high[i][0]], False)
              # bound = self.boundfn(high[i][0], t, delta)
              if lower_bound  < h_min:
                  h_min = lower_bound
                  h_star = high[i][0]

          for j in range(K, self.N):
              upper_bound = self.get_kl_bound(low[j-K][0], t, delta, self.empirical_score[low[j-K][0]], 1, True)
              # bound = self.boundfn(low[j-K][0], t, delta)
              if upper_bound > l_max:
                  l_max = upper_bound
                  l_star = low[j-K][0]
          
          correct_fraction_list.append(self.correct_fraction(high,topK_epsilon, K))
          if l_max - h_min < epsilon:
              break

          # self.update_duo(l_star, h_star)
          self.update_arm(h_star)
          self.update_arm(l_star)

          t += 1

          if t % print_every == 0:
              print("=====================================================")
              print("l_max : " + str(l_max))
              print("h_min : " + str(h_min))
              print("l_star : " + str(l_star))
              print("h_star : " + str(h_star))
              print("l_star bound function : " + str(self.boundfn(l_star, t, delta)))
              print("h_star bound function : " + str(self.boundfn(h_star, t, delta)))
              print("Empirical scores....")
              print(self.empirical_score)

      return t, high, correct_fraction_list