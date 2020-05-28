import math
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

u = [0.6, 0.7, 0.9, 0.5, 0.4, 0.2, 0.46, 0.8, 0.1, 0.55]

def generate_reward(u): #bernoulli
    n = random.randint(1,100)
    if(n > (100*u)):
        return 0
    else:
        return 1

no_of_rounds = 20000

def ucb(total_exp, no_of_arms):
    # print(type(total_exp),type(no_of_arms))
    no_of_times_arm_picked = np.zeros((total_exp,no_of_arms),dtype=int)
    sums_of_rewards = np.zeros((total_exp,no_of_arms))
    arms_selected = np.zeros((total_exp,no_of_rounds),dtype=int)
    regret = np.zeros((no_of_rounds,1))
    optimal_arm_pulls = np.zeros((no_of_rounds,1))

    for t in range(0, no_of_rounds):
        argmax = np.zeros((total_exp,1),dtype=int)
        upper_bound_max = np.zeros((total_exp,1))
        average_reward = np.zeros((total_exp,1))
        upper_bound = np.zeros((total_exp,1))
        for j in range(0,total_exp):
            for i in range(0,no_of_arms):
                if (no_of_times_arm_picked[j][i] > 0):
                    average_reward[j][0] = sums_of_rewards[j][i] / no_of_times_arm_picked[j][i]
                    delta_ji = math.sqrt(2*math.log(t+1) / no_of_times_arm_picked[j][i])
                    upper_bound[j][0] = average_reward[j][0] + delta_ji
                else:
                    upper_bound[j][0] = 1e400
                if upper_bound[j][0] > upper_bound_max[j][0]:
                    upper_bound_max[j][0] = upper_bound[j][0]
                    argmax[j][0] = i
            # print("lalala",arms_selected,"sjdhfjsd",arms_selected[j],"sdjfshd",argmax[j],j)
            # np.append(arms_selected[j],argmax[j][0])#(arms_selected[j]).append(argmax[j])
            # np.concatenate(arms_selected[j],argmax[j])
            arms_selected[j][t] = argmax[j][0]
            # print("lalala22",arms_selected[0] , arms_selected[1])
            no_of_times_arm_picked[j][argmax[j][0]] = no_of_times_arm_picked[j][argmax[j][0]] + 1
            # print(type(argmax[j]))
            reward = generate_reward(u[argmax[j][0]])
            sums_of_rewards[j][argmax[j]] = sums_of_rewards[j][argmax[j]] + reward
            # total_reward = total_reward + reward
            # print("selected", arms_selected)

        expected_reward = 0
        for h in range(0,no_of_arms):
            reward_h = 0
            for exp in range(0,total_exp):
                reward_h = reward_h + u[h]* no_of_times_arm_picked[exp][h]
            expected_reward = expected_reward + (reward_h/total_exp)
        regret[t][0] = ((max(u[0:no_of_arms])*(t+1)) - (expected_reward))/(t+1)

        optimal_arm = u.index(max(u[0:no_of_arms]))
        pulls = 0
        for exp in range(0,total_exp):
            pulls = pulls + no_of_times_arm_picked[exp][optimal_arm]
        optimal_arm_pulls[t][0] = ((pulls/total_exp)/(t+1))*100

    print(regret , "=regret")
    print(optimal_arm_pulls, "=optimal arm pulls")
    return [regret, optimal_arm_pulls]

# uc = ucb(50,5)
# # regret2 = ucb(50,2)[0]
# regret5 = uc[0]
# # regret10 = ucb(50,10)[0]
#
# # optimal_arm_pulls2 = ucb(50,2)[1]
# optimal_arm_pulls5 = uc[1]
# optimal_arm_pulls10 = ucb(50,10)[1]

# a = plt.figure(1)
# plt.plot(list(range(0,no_of_rounds)),regret2[0:])
# plt.plot(list(range(0,no_of_rounds)),regret5[0:])
# plt.plot(list(range(0,no_of_rounds)),regret10[0:])
# plt.legend(('k = 2', 'k = 5' , 'k = 10'),
#            loc='upper right')
# plt.xlabel('Round')
# plt.ylabel('Regret per time')
# a.show()
#
# b = plt.figure(2)
# plt.plot(list(range(0,no_of_rounds)),optimal_arm_pulls2[0:])
# plt.plot(list(range(0,no_of_rounds)),optimal_arm_pulls5[0:])
# plt.plot(list(range(0,no_of_rounds)),optimal_arm_pulls10[0:])
# plt.legend(('k = 2', 'k = 5' , 'k = 10'),
#            loc='upper right')
# plt.xlabel('Round')
# plt.ylabel('optimal_arm_pulls(Percentage)')
# b.show()
#
# input()
