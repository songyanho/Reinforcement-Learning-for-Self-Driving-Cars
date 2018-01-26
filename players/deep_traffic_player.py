from players.player import Player
import numpy as np


class DeepTrafficPlayer(Player):
    def decide_with_vision(self, vision, score, end_episode, cache=False, is_training=True):
        if cache:
            self.car.move(self.action_cache)
            return None, None

        action = 'M'  # Default action
        if is_training and not cache:
            self.agent.remember(score, vision, end_episode=end_episode, is_training=is_training)

        if self.car.switching_lane < 0:
            q_values, action = self.agent.act(vision, is_training=is_training)
            self.agent_action = True
        else:
            self.agent_action = False

        mismatch_direction = False
        resulted_direction = 'M'

        if self.agent_action:
            if action in ['A', 'D', 'M']:
                self.car.switch_lane('M')
                mismatch_direction = True
            else:
                resulted_direction = self.car.switch_lane(action)
                if resulted_direction[0] != action[0]:
                    self.agent.action = self.agent.action_names.index(resulted_direction)
                    mismatch_direction = True
                action = 'M'

        resulted_action = self.car.move(action)
        if resulted_action != action:
            self.agent.action = self.agent.action_names.index(resulted_action)
        self.action_cache = resulted_action
        result = resulted_direction if not mismatch_direction else resulted_action

        return q_values if self.agent_action else None, \
            result if self.agent_action else None
