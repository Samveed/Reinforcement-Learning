import numpy as np
import matplotlib.pyplot as plt
import random

#k is the number of arms
def thompson(k):
    theta = [0.6, 0.7, 0.9, 0.5, 0.4, 0.2, 0.46, 0.8, 0.1, 0.55]
    mu_star = 0
    optimal_arm = 0
    for l in range(k):
        if theta[l]>mu_star:
            mu_star = theta[l]
            optimal_arm = l

    t = 20000
    regret_per_time = [0]*t
    no_of_optimal_arms = [0]*t
    total_reward = 0;

    for x in range(50):
        alpha = [1]*k
        beta = [1]*k

        distribution = [0]*k
        chosen = [0]*k

        for r in range(t):
            for i in range(k):
                res = np.random.beta(alpha[i],beta[i])
                distribution[i]=res

            max_index = distribution.index(max(distribution));

            chosen[max_index]=chosen[max_index]+1
            no_of_optimal_arms[r] = chosen[optimal_arm]

            num = random.random()
            if num <= theta[max_index]:
                alpha[max_index]=alpha[max_index]+1
                total_reward = total_reward+1;
            else:
                beta[max_index]=beta[max_index]+1

            sum=0
            for j in range(k):
                sum = sum + (theta[j]*chosen[j])
            regret = (mu_star*(r+1))-sum

            regret_per_time[r] = regret_per_time[r] + (regret/(r+1))

    time = []
    for m in range(t):
        regret_per_time[m]=float(regret_per_time[m])/50
        no_of_optimal_arms[m] = float(no_of_optimal_arms[m]*2)/(m+1)*50
        time.append(m+1)

    return regret_per_time, no_of_optimal_arms,time



# print ("Average reward",total_reward/(t*50))

# [r1,o1,time]=thompson(2)
# [r2,o2,time]=thompson(5)
# [r3,o3,time]=thompson(10)
#
# plt.plot(time, r1, label='k=2')
# plt.plot(time, r2, label='k=5')
# plt.plot(time, r3, label='k=10')
# plt.xlabel('Rounds')
# plt.ylabel('Regret per time')
# plt.legend()
# plt.title('Thompson Sampling (Regret vs Rounds)')
# plt.show()
#
# plt.plot(time, o1, label='k=2')
# plt.plot(time, o2, label='k=5')
# plt.plot(time, o3, label='k=10')
# plt.xlabel('Rounds')
# plt.ylabel('Optimal Action Percentage')
# plt.legend()
# plt.title('Thompson Sampling (Optimal Action Percentage vs Rounds)')
# plt.show()
