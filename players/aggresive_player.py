from players.player import Player
import numpy as np


class AggresivePlayer(Player):
    def decide(self, end_episode, cache=False):
        # If car in front
        # and speed < min_speed
        # switch lane
        action = 'M'

        self.actions.rotate(-1)
        self.actions[len(self.actions) - 1] = 'M'

        if self.car.speed < self.min_speed and self.car.switching_lane < 0:
            # If car in front
            if self.car_in_front():
                plan_direction = np.random.choice(['L', 'R'])
                if plan_direction not in self.actions:
                    self.actions[len(self.actions) - 1] = plan_direction
                    self.car.switch_lane(plan_direction)
                    return

        if self.car.speed > self.max_speed:
            action = 'D'
        elif self.car.speed < self.min_speed:
            action = 'A'
        else:
            action = np.random.choice(['A', 'M'])

        self.actions[len(self.actions) - 1] = action

        self.car.move(action)

