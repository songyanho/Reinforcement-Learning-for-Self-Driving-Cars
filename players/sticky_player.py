from players.player import Player
import numpy as np


class StickyPlayer(Player):
    def decide(self, end_episode, cache=False):
        # If car in front
        # and speed < min_speed
        # switch lane
        action = 'M'

        if self.car.speed > self.max_speed:
            action = 'D'
        elif self.car.speed < self.min_speed:
            action = 'A'
        else:
            action = np.random.choice(['A', 'M', 'D'])

        self.actions.rotate(-1)
        self.actions[len(self.actions) - 1] = action

        self.car.move(action)
