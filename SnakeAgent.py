import random
from Constants import *
from DQNAgent import DQNAgent
import numpy as np


class SnakeAgent:

    def __init__(self, agent, color):

        self.last_action = NORTH
        self.steps = 0
        self.DQN = agent
        self.last_obs = None
        self.color = color

    def step(self, board, reward, done, explore=True):

        self.steps += 1

        food_feature = (board == FOOD)
        player_head_feature = (board == self.color)
        snake_body_feature = np.logical_and(board % 2 == 1, board > 2)
        snake_head_feature = np.logical_and(board % 2 == 0, board >= 2)

        obs = np.stack([food_feature, player_head_feature, snake_body_feature, snake_head_feature], axis=2)

        if self.last_obs is None:
            self.last_obs = obs

        action = self.DQN.agent_step(last_obs=self.last_obs, last_action=self.last_action,
                                     reward=reward, new_obs=obs, done=done, explore=explore)

        self.last_action = action
        self.last_obs = obs

        return action
