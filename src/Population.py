import numpy as np
import sys
import random
import pickle
import copy

W_MULTIPLIER = 1

class Controller:
    N_INPUT = 8
    N_HIDDEN = 10
    N_OUTPUT = 5

    @staticmethod
    def sigmoid(matrix):
        newValues = np.empty(matrix.shape)
        for i in range(len(newValues)):
            newValues[i] = 1/(1+np.exp(-matrix[i]))
        return newValues

    @staticmethod
    def softmax(matrix):
        exponents = np.exp(matrix - np.max(matrix))
        return exponents / np.sum(exponents)

    def forward(self, irs_input, weights):
        bias1 = weights[:self.N_HIDDEN].reshape(1, self.N_HIDDEN)

        weight1_slice = len(irs_input) * self.N_HIDDEN + self.N_HIDDEN
        weight1 = weights[self.N_HIDDEN:weight1_slice].reshape((len(irs_input), self.N_HIDDEN))

        bias2 = weights[weight1_slice:weight1_slice + self.N_OUTPUT].reshape(1, self.N_OUTPUT)
        weight2 = weights[weight1_slice + self.N_OUTPUT:].reshape((self.N_HIDDEN, self.N_OUTPUT))

        output1 = self.sigmoid(irs_input.dot(weight1) + bias1)
        output2 = self.softmax(output1.dot(weight2) + bias2)

        return output2[0]


class Individual:
    N_WEIGHTS = (Controller.N_INPUT+1)*Controller.N_HIDDEN + (Controller.N_HIDDEN+1)*Controller.N_OUTPUT

    def __init__(self):
        self.weights = []
        self.fitness = -1000000

    def init_weights(self):
        self.weights = np.random.uniform(low=-W_MULTIPLIER, high=W_MULTIPLIER, size=self.N_WEIGHTS)

    def mutate_individual(self):
        for i in range(len(self.weights)):
            if random.random() < 0.1:
                # print("mutate")
                self.weights[i] = random.uniform(-W_MULTIPLIER, W_MULTIPLIER)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def initialize_with_parents(self, parent1, parent2):
        for i in range(len(self.weights)):
            if random.random() < 0.5:
                # print("Parent 1")
                self.weights[i] = parent1.weights[i]
            else:
                # print("Parent 2")
                self.weights[i] = parent2.weights[i]

    def read_weights(self, filename):
        # This function loads an existing weights.
        with open(filename, 'rb') as fp:
            weights = pickle.load(fp)
        self.weights = weights

    def copy_weights(self, parent):
        self.weights = copy.deepcopy(parent.weights)

class Population:

    def __init__(self, size, name):
        self.pop_list = []
        self.size = size
        self.best_fitness = -1000000
        self.avg_fitness = -1000000
        self.name = name

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
        self.store_best_weights(best, self.name)

        self.calculate_avg_fitness()

        # self.pop_list[2].copy_weights(self.pop_list[0])
        for i in range(2, len(self.pop_list)):
            self.pop_list[i].initialize_with_parents(best, second_best)
            self.pop_list[i].mutate_individual()

    def calculate_avg_fitness(self):
        total = 0
        for i in range(len(self.pop_list)):
            total += self.pop_list[i].fitness
        avg_fitness = total / len(self.pop_list)
        self.avg_fitness = avg_fitness

    @staticmethod
    def store_best_weights(best, name):
        with open(f"results_EC/{name}.pickle", 'wb') as fp:
            pickle.dump(best.weights, fp)
