import random
import numpy as np
from collections import deque
import math
import os

# Set Backend according to your setup
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


# Deep Q-learning Agent
# Taken from: https://keon.io/deep-q-learning/ 2019-02-08
class DQNAgent:

    def __init__(self, observation_shape, num_of_actions, load_from=None):
        """
        Initialises the Agent for the given environment parameters
        :param observation_shape: Tuple of integers, describing the shape of the observation space
        :param num_of_actions: positive Integer, the number of possible actions
        :param load_from: If set, loads the weights for the neural network from given filepath
        """
        self.state_size = observation_shape
        self.action_size = num_of_actions
        self.stats = []

        # Memory saves each step
        self.memory = deque(maxlen=10000)

        # Discount rate: How much the current state reward is depending on future rewards
        self.gamma = 0.95

        # Exploration Rate: How much the agent is randomly exploring vs trying to maximize reward
        self.epsilon = 1.0
        self.epsilon_amplitude_min = 0.05
        self.epsilon_decay = 0.995
        self.epsilon_amplitude = 0.5

        self.epsilon_factor = 0

        # Learning Rate for the Neural Network
        self.learning_rate = 0.001

        # Construct the Neural Network
        self.model = self._build_model()

        if load_from is not None:
            self.load_weights(load_from)

    def _build_model(self):
        """
        Initialises the neural network for the Deep-Q-Learning
        :return: The created neural network model
        """
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(3, 3), padding='same', input_shape=self.state_size))
        model.add(Conv2D(8, kernel_size=(3, 3), padding='same'))
        model.add(MaxPool2D())
        model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        # Prints model architecture to console
        model.summary()

        return model

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

    def remember(self, state, action, reward, next_state, done):
        """
        Adds one game step to memory
        :param state: The state before action was taken
        :param action: The action that was taken
        :param reward: The reward after action was taken
        :param next_state: The state after action was taken
        :param done: If game state is a final state
        :return:
        """
        self.memory.append((np.array([state]), action, reward, np.array([next_state]), done))

    def agent_step(self, last_obs, last_action, reward, new_obs, done, explore=True):
        """
        Agent determines action for the new game state
        :param last_obs: Observation of the last game state(should be the same as new_obs for the first step)
        :param last_action: The last action taken by the agent(should None for the first step)
        :param reward: A number that shows the reward for the current game state
        :param new_obs: Observation of the current game state
        :param done: True if state is last game state
        :param explore: If exploration should be enabled
        :return: An action < num_of_actions
        """

        # First Step sets last_obs/last_action/reward None, so dont save them
        if last_obs is not None and last_action is not None and reward is not None:
            self.remember(last_obs, last_action, reward, new_obs, done)

        return self.act(np.array([new_obs]), explore=explore)

    def act(self, state, explore=True):
        """
        Determines an action from current state
        :param state: The current game state
        :param explore: If exploration should be enabled
        :return: An action < num_of_actions
        """

        # Decide randomly to explore or exploit
        if np.random.rand() <= self.epsilon and explore:
            # Do random action (explore)
            return random.randrange(self.action_size)

        # Predict best action and do it (exploit)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """
        Train the neural network from experience
        :param batch_size: Amount of random memories used for training
        :return: None
        """

        batch_size = min(batch_size, len(self.memory))

        # Sample random memories
        minibatch = random.sample(self.memory, batch_size)

        # Create training data from memories
        train_x = []
        train_y = []

        for state, action, reward, next_state, done in minibatch:

            target = reward

            # A game states value is determined by current and future reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))

            # Target vector contains the predicted values for each state, except from the chosen action
            # The value for the chosen action is set according to Q calculation from current and expected future reward
            target_f = self.model.predict(state)
            target_f[0][action] = target

            train_x.append(state[0])
            train_y.append(target_f[0])

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        # Train the agent
        self.model.fit(train_x, train_y, epochs=1, verbose=0)

        # Change Epsilon
        self.epsilon_factor += 0.1
        if self.epsilon_amplitude > self.epsilon_amplitude_min:
            self.epsilon_amplitude *= self.epsilon_decay
        self.epsilon = ((math.sin(self.epsilon_factor) + 1) / 2)*self.epsilon_amplitude

        if self.epsilon < 0.01:
            self.epsilon = 0

    def log_stats(self, episode, min_survived, max_survived, max_reward):

        data = (episode, min_survived, max_survived, max_reward, self.epsilon)

        print(str(data))

        self.stats.append(data)


