import math
import random
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np

def prob_1_b():
    theta = 0.75
    n = 4
    y = np.linspace(0,4,5)

    factorial = np.vectorize(math.factorial)
    n_choose_k = math.factorial(n) / (factorial(y) * factorial(n - y))

    pow = np.vectorize(math.pow)
    theta_pow_y = pow(theta, y)
    opp_theta_pow_n_diff_y = pow((1 - theta), n - y)

    likelihood = n_choose_k * theta_pow_y * opp_theta_pow_n_diff_y

    plt.plot(y, likelihood)
    plt.title("Coin Flip Likelihood")
    plt.ylabel("P(y|theta, n)")
    plt.xlabel("y")
    plt.show()

def prob_1_c():
    calc_dist(1,1)
    calc_dist(2,2)
    calc_dist(3,2)
    calc_dist(4,3)

def calc_dist(n, y):
    theta_domain = np.linspace(0,1,100)

    n_choose_y = math.factorial(n) / math.factorial(y) * math.factorial(n - y)

    pow = np.vectorize(math.pow)
    theta_pow_y = pow(theta_domain, y)
    opp_theta_pow_n_diff_y = pow((1 - theta_domain), n - y)

    post_dist = n_choose_y * theta_pow_y * opp_theta_pow_n_diff_y * (n + 1)
    plt.plot(theta_domain, post_dist)
    plt.ylabel("Posterior distribution")
    plt.xlabel("theta")
    plt.show()

def prob_2_a(show=True):
    #proportion of cherry candies
    h_i = [1, 0.75, 0.5, 0.25, 0]

    sample = list()

    official_cherries = 0
    for i in range(1,101):
        num = random.randint(0,1)
        sample.append(num)
        if num == 1:
            official_cherries += 1

    prior_dist = [0.1, 0.2, 0.4, 0.2, 0.1]

    posterior_prob = np.ndarray((100,5))

    total_dist = [prior_dist]
    next_candy_dist = list()
    overall_prediction = list()
    num_cherries = 0

    for val in range(0, len(sample)):
        n = val + 1
        curr_dist = list()
        if sample[val] == 1:
            num_cherries += 1

        alpha_inverse = 0
        for i in range(0,5):
            prob_data_given_hypo = Decimal(math.pow(h_i[i], num_cherries) * math.pow((1 - h_i[i]), n - num_cherries))

            #P(h_i | d) = P(d|h_i)P(h_i)
            posterior_prob = prob_data_given_hypo * Decimal(prior_dist[i])
            curr_dist.append(posterior_prob)
            alpha_inverse += posterior_prob

        for i in range(0, len(curr_dist)):
            curr_dist[i] /= alpha_inverse

        prob_next_candy = 0
        curr_next_candy_dist = list()
        for i in range(0, 5):
            curr_val = Decimal((1-h_i[i])) * curr_dist[i]
            curr_next_candy_dist.append(curr_val)
            prob_next_candy += curr_val

        overall_prediction.append(prob_next_candy)
        next_candy_dist.append(curr_next_candy_dist)
        total_dist.append(curr_dist)

    total_dist = np.array(total_dist)
    next_candy_dist = np.array(next_candy_dist)
    overall_prediction = np.array(overall_prediction)

    for i in range(0,5):
        plt.plot(total_dist[:,i], label="h"+str(i+1))
    plt.title("Hypothesis distributions")

    if show:
        plt.legend()
        plt.show()

    for i in range(0,5):
        plt.plot(next_candy_dist[:,i], label="h"+str(i+1))
    plt.plot(overall_prediction, label="Overall prediction")


    if show:
        plt.legend()
        plt.show()

    plt.clf()

    return total_dist

def prob_2_c():
    datasets = list()
    for i in range(0,20):
        datasets.append(prob_2_a(show=False))

    variability = list()
    for j in range(0, len(datasets[0])):
        curr_list = list()
        for i in range(0, 5):
            curr_sum = 0
            for k in range(0, len(datasets)):
                curr_sum += datasets[k][j][i]
            curr_list.append(curr_sum / len(datasets))
        variability.append(curr_list)
    variability = np.array(variability)
    print(variability[:,1])

    for i in range(0,5):
        plt.plot(variability[:,i], label="h"+str(i+1))
    plt.legend()
    plt.title("Variability of curves")
    plt.show()

prob_2_c()