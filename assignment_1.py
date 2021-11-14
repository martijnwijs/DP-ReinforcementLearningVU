import math

import matplotlib.pyplot as plt
import numpy as np


def partA():
    fares = [500, 300, 200]
    C = 100  # capacity
    T = 600  # time limit
    mu = [0.001, 0.015, 0.05]  # first for capactiy 500, 300, 200
    vu = [0.01, 0.005, 0.0025]  # first for capactiy 500, 300, 200 as time increases, chances for high fair go up

    V_m = np.empty((C + 1, T + 1, 3))  # value matrix, 2d for each fare
    V_m[0] = 0  # at capacity zero there are no values
    V_m[:, T, :] = np.zeros((101, 3))  # at t = T there are no values

    for t in reversed(range(T)):  # loop over the time states starting with the last state
        for x in range(1, C + 1):  # loop over the capacities starting with 0 capactiy
            for c in range(3):  # loop over the different classes
                V_xtc = 0  # initialize V(xtc)
                sum_lab = 0  # initialize lambda

                # get the sum of lambda
                for i in range(c + 1):  # the one that is able to pay 500 is also able to pay 300 and 200
                    sum_lab += mu[i] * math.exp(vu[i] * t)

                # loop over the different classes for the next state to get the different Q values, and by this the policy value V
                for c_n in range(3):
                    Q_xtf = sum_lab * (fares[c] + V_m[x - 1][t + 1][c_n]) + (1 - sum_lab) * V_m[x][t + 1][
                        c_n]  # Q value state X, at time t, given action c_n
                    if Q_xtf > V_xtc:
                        V_xtc = Q_xtf
                V_m[x][t][c] = V_xtc  # store the maximum value in the value matrix

    optimal_p = np.argmax(V_m, axis=2)  # get optimal values over the third dimension -> this should be the policy
    exp_rev = np.max(V_m)  # the expected revenue
    print("Expected revenue: ", exp_rev)
    return optimal_p


def partB(optimal_p):
    # plotting
    plt.imshow(optimal_p, cmap="RdYlGn", vmin=0, vmax=2, aspect='auto', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Capacity')
    plt.show()


def partC(simulationCount, capacity, T, fares, optimal_p):
    #     Simulation
    simulationCount = simulationCount
    simulationPeriod = T
    simulationProfits = []
    simulationActions = []
    simulationCapacities = []
    simulationTickets = []
    np.random.seed(0)
    for simulation in range(simulationCount):
        rewards = np.zeros(simulationPeriod)
        actions = np.zeros(simulationPeriod)
        capacities = np.empty(simulationPeriod)
        ticket_sold = np.zeros(simulationPeriod)
        capacity_left = 100
        total_revenue = 0
        states = np.random.randint(0, capacity, size=simulationPeriod)
        random_demand = np.random.choice(np.append(fares, 0), size=simulationPeriod)
        for time in range(simulationPeriod):
            if capacity_left <= 0:
                break
            action = optimal_p[states[time], time]
            actions[time] = fares[int(action)]
            capacities[time] = capacity_left
            if random_demand[time] >= action:
                ticket_sold[time] = 1
                capacity_left -= 1
        rewards = np.dot(ticket_sold, actions)
        simulationProfits.append(np.sum(rewards))
        simulationActions.append(actions)
        simulationCapacities.append(capacities)
        simulationTickets.append(ticket_sold)
    print("Simulation profit Average: ", np.average(np.array(simulationProfits)))
    print("Simulation max profit: ", max(simulationProfits))
    # print(simulationProfits[0])
    print("Prices at every moment :", simulationActions[0][:10])
    print("Sold tickets: ", simulationTickets[0][:10])
    print("Remaining capacities: ", simulationCapacities[0][:10])


def partD(simulationCount, capacity, T, fares, optimal_p):
    #     Simulation
    simulationCount = simulationCount
    simulationPeriod = T
    simulationProfits = []
    simulationActions = []
    simulationCapacities = []
    simulationTickets = []
    np.random.seed(0)
    for simulation in range(simulationCount):
        rewards = np.zeros(simulationPeriod)
        actions = np.zeros(simulationPeriod)
        capacities = np.empty(simulationPeriod)
        ticket_sold = np.zeros(simulationPeriod)
        capacity_left = 100
        total_revenue = 0
        states = np.random.randint(0, capacity, size=simulationPeriod)
        random_demand = np.random.choice(np.append(fares, 0), size=simulationPeriod)
        previousPrice = 0
        for time in range(simulationPeriod):
            if capacity_left <= 0:
                break
            action = optimal_p[states[time], time]
            if fares[int(action)] < previousPrice:
                actions[time] = 0
            else:
                actions[time] = fares[int(action)]
            previousPrice = fares[int(action)]
            capacities[time] = capacity_left
            if random_demand[time] >= action:
                ticket_sold[time] = 1
                capacity_left -= 1
        rewards = np.dot(ticket_sold, actions)
        simulationProfits.append(np.sum(rewards))
        simulationActions.append(actions)
        simulationCapacities.append(capacities)
        simulationTickets.append(ticket_sold)
    print("Expected Revenue: ", max(simulationActions))


if __name__ == "__main__":
    optimal_p = partA()
    partB(optimal_p)
    fares = [500, 300, 200]
    partC(10000, 100, 600, fares, optimal_p)
    partD(10000, 100, 600, fares, optimal_p)
