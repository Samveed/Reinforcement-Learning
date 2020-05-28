import numpy as np
import sys
import random
import matplotlib.pyplot as plt

u = [0.6, 0.7, 0.9, 0.5, 0.4, 0.2, 0.46, 0.8, 0.1, 0.55]

def generate_reward(u): #bernoulli
    n = random.randint(1,100)
    if(n > (100*u)):
        return 0
    else:
        return 1

no_of_rounds = 20000

def softmax(total_exp, no_of_arms,temp):
    # print(type(total_exp),type(no_of_arms))
    no_of_times_arm_picked = np.zeros((total_exp,no_of_arms),dtype=int)
    sums_of_rewards = np.zeros((total_exp,no_of_arms))
    arms_selected = np.zeros((total_exp,no_of_rounds),dtype=int)
    regret = np.zeros((no_of_rounds,1))
    average_reward = np.zeros((total_exp,no_of_arms))
    softmax_probs_num = np.zeros((total_exp,no_of_arms))
    softmax_probs = np.zeros((total_exp,no_of_arms))
    optimal_arm_pulls = np.zeros((no_of_rounds,1))

    for t in range(0, no_of_rounds):
        if temp>0:
            argmax = np.zeros((total_exp,1),dtype=int)
            for j in range(0,total_exp):
                for i in range(0,no_of_arms):
                    if (no_of_times_arm_picked[j][i] > 0):
                        average_reward[j][i] = sums_of_rewards[j][i] / no_of_times_arm_picked[j][i]
                    else:
                        average_reward[j][i] = 1
                    softmax_probs_num[j][i] = np.exp(average_reward[j][i]/temp)

                for i in range(0,no_of_arms):
                    softmax_probs[j][i] = softmax_probs_num[j][i] / np.sum(softmax_probs_num[j])

                # print("jajsdh",np.where(softmax_probs[j] == np.amax(softmax_probs[j]))[0])
                # print("max element=", softmax_probs[j])
                argmax[j][0] = np.where(softmax_probs[j] == np.amax(softmax_probs[j]))[0][0]#softmax_probs[j].index(max(softmax_probs[j]))
                arms_selected[j][t] = argmax[j][0]
                # print("lalala22",arms_selected[0] , arms_selected[1])
                no_of_times_arm_picked[j][argmax[j][0]] = no_of_times_arm_picked[j][argmax[j][0]] + 1
                # print(type(argmax[j]))
                reward = generate_reward(u[argmax[j][0]])
                sums_of_rewards[j][argmax[j]] = sums_of_rewards[j][argmax[j]] + reward
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

# temp = 1000
#
# s = softmax(50,5,temp)
# # regret2 = softmax(50,2,temp)[0]
# regret5 = s[0]
# # regret10 = softmax(50,10,temp)[0]
# #
# # optimal_arm_pulls2 = softmax(50,2,temp)[1]
# optimal_arm_pulls5 = s[1]
# # optimal_arm_pulls10 = softmax(50,10,temp)[1]
#

# regret2 = softmax(50,5,100)[0]
# regret5 = softmax(50,5,1000)[0]
# regret10 = softmax(50,5,10000)[0]
#
# optimal_arm_pulls2 = softmax(50,5,100)[1]
# optimal_arm_pulls5 = softmax(50,5,1000)[1]
# optimal_arm_pulls10 = softmax(50,5,10000)[1]


# a = plt.figure(1)
# plt.plot(list(range(0,no_of_rounds)),regret2[0:])
# plt.plot(list(range(0,no_of_rounds)),regret5[0:])
# plt.plot(list(range(0,no_of_rounds)),regret10[0:])
# # plt.legend(('k = 2', 'k = 5' , 'k = 10'),
# #            loc='upper right')
# plt.legend(('k = 5, temp =100', 'k = 5, temp =1000' , 'k = 5, temp =10000'),
#            loc='upper right')
# plt.xlabel('Round')
# plt.ylabel('Regret per time')
# plt.title('Softmax(Round vs Regret per time)')
# a.show()
#
# b = plt.figure(2)
# plt.plot(list(range(0,no_of_rounds)),optimal_arm_pulls2[0:])
# plt.plot(list(range(0,no_of_rounds)),optimal_arm_pulls5[0:])
# plt.plot(list(range(0,no_of_rounds)),optimal_arm_pulls10[0:])
# # plt.legend(('k = 2', 'k = 5' , 'k = 10'),
# #            loc='upper right')
# plt.legend(('k = 5, temp =100', 'k = 5, temp =1000' , 'k = 5, temp =10000'),
#            loc='upper right')
# plt.xlabel('Round')
# plt.ylabel('optimal_arm_pulls(Percentage)')
# plt.title('Softmax(Round vs optimal_arm_pulls(Percentage))')
# b.show()
#
# input()
