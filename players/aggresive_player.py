from players.player import Player
import numpy as np


class AggresivePlayer(Player):
    def decide(self, end_episode, cache=False):
        # If car in front
        # and speed < min_speed
        # switch lane
        action = 'M'

        if self.car.speed < self.min_speed and self.car.switching_lane < 0:
            # If car in front
            if self.car_in_front():
                self.car.switch_lane(np.random.choice(['L', 'R']))
                return

        if self.car.speed > self.max_speed:
            action = 'D'
        elif self.car.speed < self.min_speed:
            action = 'A'
        else:
            action = np.random.choice(['A', 'M'])

        self.car.move(action)

