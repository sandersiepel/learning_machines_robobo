import numpy as np
import sys
import random
import pickle
import copy

W_MULTIPLIER = 10
# test = np.random.uniform(low=-W_MULTIPLIER, high=W_MULTIPLIER, size=112)

class Controller:
    N_HIDDEN = 10
    N_OUTPUT = 2

    def sigmoid(self, matrix):
        newValues = np.empty(matrix.shape)
        for i in range(len(newValues)):
            newValues[i] = 1/(1+np.exp(-matrix[i]))
        return newValues

    def forward(self, irs_input, weights):
        bias1 = weights[:self.N_HIDDEN].reshape(1, self.N_HIDDEN)

        weight1_slice = len(irs_input) * self.N_HIDDEN + self.N_HIDDEN
        weight1 = weights[self.N_HIDDEN:weight1_slice].reshape((len(irs_input), self.N_HIDDEN))

        bias2 = weights[weight1_slice:weight1_slice + self.N_OUTPUT].reshape(1, self.N_OUTPUT)
        weight2 = weights[weight1_slice + self.N_OUTPUT:].reshape((self.N_HIDDEN, self.N_OUTPUT))

        output1 = self.sigmoid(irs_input.dot(weight1) + bias1)
        output2 = output1.dot(weight2) + bias2

        return output2[0]


class Individual:
    N_WEIGHTS = 112

    def __init__(self):
        self.weights = []
        self.fitness = -1000000

    def init_weights(self):
        self.weights = np.random.uniform(low=-W_MULTIPLIER, high=W_MULTIPLIER, size=self.N_WEIGHTS)
        # self.weights = test

    def mutate_individual(self):
        for i in range(len(self.weights)):
            if random.random() < 10:
                self.weights[i] = random.uniform(-W_MULTIPLIER, W_MULTIPLIER)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def create_with_parents(self, parent1, parent2):
        self.weights = copy.deepcopy(parent1.weights)
        for i in range(len(self.weights)):
            if random.random() < 50:
                self.weights[i] = parent2.weights[i]


class Population:

    def __init__(self, size):
        self.pop_list = []
        self.size = size
        self.best_fitness = -1000000
        self.avg_fitness = -1000000

    def create_new(self):
        for i in range(self.size):
            ind = Individual()
            ind.init_weights()
            self.pop_list.append(ind)

    def next_gen(self):
        self.pop_list.sort(reverse=True)
        best = self.pop_list[0]
        second_best = self.pop_list[1]

        self.best_fitness = best.fitness
        self.store_best_weights(best)

        # print("best " + str(best.fitness))
        # print("sbest " + str(second_best.fitness))

        self.calculate_avg_fitness()

        new_pop = [best, second_best]

        for i in range(self.size - 2):
            ind = Individual()
            ind.create_with_parents(best, second_best)
            # ind.init_weights()
            ind.mutate_individual()
            new_pop.append(ind)
        self.pop_list = new_pop

    def calculate_avg_fitness(self):
        total = 0
        for i in range(len(self.pop_list)):
            total += self.pop_list[i].fitness
        avg_fitness = total / len(self.pop_list)
        self.avg_fitness = avg_fitness

    @staticmethod
    def store_best_weights(best):
        with open(f"best_weights", 'wb') as fp:
            pickle.dump(best.weights, fp)
