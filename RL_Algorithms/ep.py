import numpy as np
from matplotlib import pyplot as plt

np.random.seed(121)
arms = 5
steps = 20000
epsilon = 0.1
runs = 50
expected_action_value = [0.6, 0.7, 0.9, 0.5, 0.4]

#Bernoulli Reward Distribution
def reward_function(action_taken, expected_action_value):
    if (np.random.uniform(0,1)<= expected_action_value[action_taken]):
        return(1)
    else:
        return(0)

def regret(s,arm_pulls,arms):
    regret1 = max(expected_action_value)*s
    sum1 = 0
    for i in range(arms):
        sum1 = sum1 + arm_pulls[i]*expected_action_value[i]
        regret1 = regret1 + sum1
        regret1 = regret1/(s+1)
    return regret1

def epsilon_greedy(arms, steps, epsilon,expected_action_value):
    optimal_action, overall_regret = [], []
    estimate_action_value = np.zeros(arms)

    arm_pulls = np.zeros(arms)

    for s in range(0, steps):

        e_estimator = np.random.uniform(0,1)

        action = np.argmax(estimate_action_value) if e_estimator > epsilon else np.random.choice(np.arange(arms))

        regret1 = regret(s,arm_pulls,arms)
        reward = reward_function(action, expected_action_value)

        estimate_action_value[action] = estimate_action_value[action] + (1/(arm_pulls[action]+1)) * (reward - estimate_action_value[action])
        #print(np.argmax(expected_action_value))
        #overall_reward.append(reward)
        optimal_action.append(((arm_pulls[np.argmax(expected_action_value)])/(s+1))*100)
        overall_regret.append(regret1)
        arm_pulls[action] += 1
        #print(overall_regret)
        #print(arm_pulls)
    return(optimal_action, overall_regret)

def run(runs, steps, arms,expected_action_value,epsilon):
    #rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))
    total_regret = np.zeros((runs, steps))
    final_optimal = np.zeros(steps)
    final_regret = np.zeros(steps)
    for run in range(0, runs):
        optimal_actions[run][:],total_regret[run][:]  = epsilon_greedy(arms, steps, epsilon, expected_action_value)
        final_optimal = final_optimal + optimal_actions[run][:]
        final_regret = final_regret + total_regret[run][:]
    #rewards_avg = np.average(rewards, axis = 1)
    #optimal_action_perc = np.average(optimal_actions, axis = 1)
    #t_regret = np.average(total_regret, axis = 1);
    #print(overall_regret)
    # print(steps)
    # return(rewards_avg, optimal_action_perc)
    final_regret = final_regret/runs
    final_optimal = final_optimal/runs
    return final_regret, final_optimal


# arms = 5
# steps = 20000
# epsilon = 0.1
# runs = 50
# expected_action_value = [0.6, 0.7, 0.9, 0.5, 0.4]
# [r1,o1]=run(runs, steps, arms, expected_action_value,epsilon)
#
# arms = 5
# steps = 20000
# epsilon = 0.4
# runs = 50
# expected_action_value = [0.6, 0.7, 0.9, 0.5, 0.4]
# [r2,o2]=run(runs, steps, arms, expected_action_value,epsilon)
#
# arms = 5
# steps = 20000
# epsilon = 0.8
# runs = 50
# expected_action_value = [0.6, 0.7, 0.9, 0.5, 0.4]
# [r3,o3]=run(runs, steps, arms, expected_action_value,epsilon)

# time = range(steps)
#
# plt.plot(time, r1, label='e = 0.1',color='r')
# plt.plot(time, r2, label='e = 0.4',color='g')
# plt.plot(time, r3, label='e = 0.8',color='b')
# plt.xlabel('Rounds')
# plt.ylabel('Regret per time')
# plt.legend()
# plt.title('Epsilon Greedy (Regret vs Rounds)')
# plt.show()
#
# plt.plot(time, o1, label='e = 0.8',color='b')
# plt.plot(time, o2, label='e = 0.4',color='g')
# plt.plot(time, o3, label='e = 0.1',color='r')
# plt.xlabel('Rounds')
# plt.ylabel('Optimal Action Percentage')
# plt.legend()
# plt.title('Epsilon Greedy (Optimal Action Percentage vs Rounds)')
# plt.show()
