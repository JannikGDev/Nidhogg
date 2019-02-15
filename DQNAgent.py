import random
import numpy as np
from collections import deque
import math

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Input
from keras.models import Sequential
from keras.optimizers import Adam

MEM_CAPACITY = 10000
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001


# Deep Q-learning Agent
# Taken from: https://keon.io/deep-q-learning/ 2019-02-08
class DQNAgent:

    def __init__(self, observation_shape, num_of_actions):
        self.state_size = observation_shape
        self.action_size = num_of_actions
        self.memory = deque(maxlen=MEM_CAPACITY)
        self.gamma = GAMMA    # discount rate
        self.epsilon = EPSILON_START  # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE

        self.log = []
        self.plays = 0
        self.max_steps = 0

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(3, 3), padding='same', input_shape=self.state_size))
        model.add(Conv2D(8, kernel_size=(3, 3), padding='same'))
        model.add(MaxPool2D())
        model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        model.summary()

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array([state]), action, reward, np.array([next_state]), done))

    def agent_step(self, last_obs, last_action, reward, new_obs, done):

        self.remember(last_obs, last_action, reward, new_obs, done)

        if done:
            self.plays += 1
            if self.plays % 100 == 0:
                self.log.append(self.max_steps)
                self.max_steps = 0
                self.print_status()

        return self.act(np.array([new_obs]))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        batch_size = min(batch_size, len(self.memory))

        minibatch = random.sample(self.memory, batch_size)

        train_x = []
        train_y = []

        for state, action, reward, next_state, done in minibatch:

            target = reward

            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            train_x.append(state[0])
            train_y.append(target_f[0])

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        self.model.fit(train_x, train_y, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def survived_steps(self, steps):

        if self.max_steps < steps:
            self.max_steps = steps

    def print_status(self):

        print("")
        print("Epsilon: " + str(self.epsilon))
        print("Memory: " + str(len(self.memory)) + "/" + str(MEM_CAPACITY))
        print("Plays: " + str(self.plays))
        print("Log: " + str(self.log))

    def load_weights(self, filepath):
        """
        Loads the neural network weights from a file
        :param filepath: The file containing the weights
        :return: None
        """
        self.model.load_weights(filepath + ".h5")
        print("Loaded weights from disk")

    def save_weights(self, filepath):
        """
        Saves the neural network weights to a file
        :param filepath: Where to save the file
        :return: None
        """
        self.model.save_weights(filepath + ".h5")
        print("Saved weights to disk")
