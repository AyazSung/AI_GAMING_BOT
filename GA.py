import random
from typing import List

import cv2
import gym
import numpy as np
import tensorflow as tf
from MODEL_CNN import ModelCNN


class GA:

    def __init__(self, population_size: int,
                 gen_limit: int,
                 p_crossover: float,
                 p_mutation: float,
                 env: gym.Env,
                 number_of_weights: int,
                 model: ModelCNN):
        self.population_size = population_size
        self.gen_limit = gen_limit
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.LOW = -5
        self.UP = 5
        self.env = env
        self.number_of_weights = number_of_weights
        self.modelCNN = model

    def create_weight(self) -> float:
        return random.uniform(self.LOW, self.UP)

    def create_individual(self) -> List[float]:
        return [self.create_weight() for _ in range(self.number_of_weights)]

    def create_population(self) -> List[List[float]]:
        return [self.create_individual() for _ in range(self.population_size)]

    def crossover(self, parent1: List[float], parent2: List[float]) -> (List[float], List[float]):
        crossover_point = random.randint(0, self.number_of_weights - 1)

        if random.random() < self.p_crossover:
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        else:
            return parent1, parent2

    def selection(self, population: List[List[float]]) -> (List[float], List[float]):
        return random.choices(
            population=population,
            weights=[self.fitness(individual) for individual in population],
            k=2)

    def mutation(self, individual) -> List[float]:
        for i in range(self.number_of_weights):
            if random.random() < self.p_mutation:
                individual[i] = self.create_weight()
        return individual

    def run_ga(self, pop_size: int, gen_limit: int, p_crossover: float, p_mutation: float) -> List[float]:
        print("Creating initial population...")
        population_main = self.create_population()
        print("Evaluating initial population...")
        population_main.sort(key=self.fitness, reverse=True)
        print("Main population sorted.")
        print(f"Best fitness, GEN INIT : {self.fitness(population_main[0])}")

        for generation in range(gen_limit):
            population_next = population_main[:2]

            print("Creating next population...")
            for _ in range(pop_size // 2 - 1):
                parent1, parent2 = self.selection(population_main)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                population_next.append(child1)
                population_next.append(child2)

            population_main = population_next
            print("Evaluating next population...")
            population_main.sort(key=self.fitness, reverse=True)
            print("Next population sorted.")
            best_f = self.fitness(population_main[0])
            print(f"Best fitness, GEN {generation} : {best_f}")

        population_main.sort(key=self.fitness, reverse=True)

        return population_main[0]

    def fitness(self, individual: List[float]) -> float:

        self.modelCNN.set_weights(individual)

        observation = self.env.reset()[0]
        done = False
        action_count = 0
        total_reward = 0
        while not done and action_count < 100:
            observation = observation[35:170, 20:]
            observation = cv2.resize(observation, dsize=(70, 70), interpolation=cv2.INTER_CUBIC)
            action = self.modelCNN.model.predict(np.array([observation]), verbose=0)

            if random.random() > 0.3:
                action = np.argmax(action)
            else:
                action = random.choice(range(0, 6))

            observation, reward, truncted, terminated, info = self.env.step(action)
            action_count += 1
            done = terminated or truncted
            total_reward += reward

        return total_reward

    def find_best_weights(self) -> List[float]:
        return self.run_ga(self.population_size, self.gen_limit, self.p_crossover, self.p_mutation)
