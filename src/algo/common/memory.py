#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-10 15:27:16
@LastEditor: John
LastEditTime: 2021-03-13 11:37:15
@Discription: 
@Environment: python 3.7.7
'''
import random
import os
import pickle
from collections import deque

import numpy as np


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        A = (state, action, reward, next_state, done)
        self.buffer[self.position] = A
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Expert_Buffer(ReplayBuffer):
    def __init__(self, capacity):
        super(Expert_Buffer, self).__init__(capacity)
        self.folder_path = os.path.join(os.getcwd(), "algo/SPI/expert_data/datasets/")
        # self.folder_path = os.path.join(os.getcwd(), "expert_data/datasets/")
        self.data_name = self.folder_path + 'processed_data.pkl'
        self.data = None
        self.load_data()
        self.process_data()

    def load_data(self):
        with open(self.data_name, 'rb') as file:
            self.data = pickle.load(file)
        print("Data has been loaded.")

    def process_data(self):
        self.push_data()

    def push_data(self):
        states = self.data['state']
        actions = self.data['action']
        for i in range(len(states)):
            actions[i][0] = (actions[i][0] - 7 + 1.75) / 5.25
            actions[i][1] = actions[i][1] / 10 - 1
            A = (states[i], actions[i])
            self.buffer.append(A)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action = zip(*batch)
        return state, action


class Elite_Buffer:
    def __init__(self):
        self.position = 0
        self.buffer = []
        self.elite_data = []
        self.elite_top_data = []

    def elite_data_select(self, data):
        ave_reward = np.zeros((len(data)))
        for i in range(len(data)):
            reward = 0.0
            for j in range(len(data[i])):
                reward += data[i][j][2]
            ave_reward[i] = reward
        max_indices = np.argsort(ave_reward)[-2:]
        for i in range(len(max_indices)):
            self.elite_data.append([data[max_indices[i]], ave_reward[i]])

    def push(self, data):
        self.elite_data_select(data)
        if len(self.buffer) > 3000:
            mid = len(self.buffer) // 2
            self.buffer = self.buffer[mid:]

        if len(self.elite_data) > 10:
            ave_reward = np.zeros(len(self.elite_data))
            elite_top_data = []
            for i in range(len(self.elite_data)):
                ave_reward[i] = self.elite_data[i][1]

            sorted_indices = np.argsort(ave_reward)
            top_indices = sorted_indices[-5:]
            for i in range(len(top_indices)):
                elite_top_data.append(self.elite_data[top_indices[i]])
            self.elite_data = elite_top_data
            for i in range(len(self.elite_data)):
                for j in range(len(self.elite_data[i][0])):
                    state = self.elite_data[i][0][j][0]
                    action = self.elite_data[i][0][j][1]
                    A = (state, action)
                    self.buffer.append(A)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action = zip(*batch)
        return state, action
