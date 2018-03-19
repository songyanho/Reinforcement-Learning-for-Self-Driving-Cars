import numpy as np
import math
from collections import deque


class Player:
    def __init__(self, car, min_speed_range=(50, 100), agent=None):
        self.car = car
        self.min_speed = np.random.randint(min_speed_range[0], min_speed_range[1])
        self.max_speed = min_speed_range[1]
        self.agent = agent

        self.action_cache = 'M'
        self.agent_action = False

        self.actions = deque(['M', 'M', 'M', 'M'])

    def decide(self, end_episode, cache=False):
        # Move forward
        if self.car.speed < self.min_speed:
            action = 'A'
        elif self.car.speed > self.max_speed:
            action = 'D'
        else:
            action = 'M'

        self.actions.rotate(-1)
        self.actions[len(self.actions) - 1] = action

        self.car.move(action)

        # Direction
        self.car.switch_lane('M')

    def decide_with_vision(self, vision, score, end_episode, cache=False, is_training=True):
        pass

    def car_in_front(self, threshold=5):
        max_box = max(min(int(math.floor(self.car.y / 10.0)) - 1, 0), 99)
        min_box = max(min(max_box - threshold, 0), 99)

        for y in range(min_box, max_box + 1):
            if self.car.lane_map[y][self.car.lane - 1] != 0 and self.car.lane_map[y][self.car.lane - 1] != self.car:
                return True

        return False
