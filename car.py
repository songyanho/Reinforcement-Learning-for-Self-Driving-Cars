import math
import pygame
import os
import numpy as np

from players.player import Player
from players.aggresive_player import AggresivePlayer
from players.sticky_player import StickyPlayer
from players.deep_traffic_player import DeepTrafficPlayer

from config import VISION_B, VISION_F, VISION_W, \
    VISUALENABLED, EMERGENCY_BRAKE_MAX_SPEED_DIFF, ROAD_VIEW_OFFSET, \
    VISUAL_VISION_B, VISUAL_VISION_F, VISUAL_VISION_W


MAX_SPEED = 110  # km/h

DEFAULT_CAR_POS = 700

IMAGE_PATH = './images'

if VISUALENABLED:
    red_car = pygame.image.load(os.path.join(IMAGE_PATH, 'red_car.png'))
    red_car = pygame.transform.scale(red_car, (34, 70))
    white_car = pygame.image.load(os.path.join(IMAGE_PATH, 'white_car.png'))
    white_car = pygame.transform.scale(white_car, (34, 70))

direction_weight = {
    'L': 0.01,
    'M': 0.98,
    'R': 0.01,
}

move_weight = {
    'A': 0.30,
    'M': 0.50,
    'D': 0.20
}


class Car():
    def __init__(self, surface, lane_map, speed=0, y=0, lane=4, is_subject=False, subject=None, score=None, agent=None):
        self.surface = surface
        self.lane_map = lane_map
        self.sprite = None if not VISUALENABLED else red_car if is_subject else white_car
        self.speed = min(max(speed, 0), MAX_SPEED)
        self.y = y
        self.lane = lane
        self.x = (self.lane - 1) * 50 + 15 + 8 + ROAD_VIEW_OFFSET
        self.is_subject = is_subject
        self.subject = subject
        self.max_speed = -1
        self.removed = False
        self.emergency_brake = None

        self.switching_lane = -1
        self.available_directions = ['M']
        self.available_moves = ['D']

        self.score = score

        self.player = np.random.choice([
                Player(self),
                AggresivePlayer(self),
                StickyPlayer(self)
            ]) if not self.is_subject else DeepTrafficPlayer(self, agent=agent)

        self.hard_brake_count = 0
        self.alternate_line_switching = 0

    def identify(self):
        min_box = int(math.floor(self.y / 10.0)) - 1
        max_box = int(math.ceil(self.y / 10.0))

        # Out of bound
        if self.y < -200 or self.y > 1200:
            self.removed = True
            return False

        if 0 <= min_box < 100:
            self.lane_map[min_box][self.lane - 1] = self
            if 1 <= self.switching_lane <= 7:
                self.lane_map[min_box][self.switching_lane - 1] = self
        for i in range(-1, 9):
            if 0 <= max_box + i < 100:
                self.lane_map[max_box + i][self.lane - 1] = self
                if 1 <= self.switching_lane <= 7:
                    self.lane_map[max_box + i][self.switching_lane - 1] = self
        return True

    def accelerate(self):
        # If in front has car then cannot accelerate but follow
        self.speed += 1.0 if self.speed < MAX_SPEED else 0.0

    def decelerate(self):
        if self.max_speed > -1:
            self.speed = self.max_speed
        else:
            self.speed -= 1.0 if self.speed > 0 else 0.0

    def check_switch_lane(self):
        if self.switching_lane == -1:
            return
        self.x += (self.switching_lane - self.lane) * 50
        if self.x == ROAD_VIEW_OFFSET + (self.switching_lane - 1) * 50 + 15 + 8:
            self.lane = self.switching_lane
            self.switching_lane = -1

    def move(self, action):
        moves = self.available_moves

        if action not in moves:
            action = moves[0]
            if self.subject is None:
                self.score.action_mismatch_penalty()

        if action == 'A':
            self.accelerate()
        elif action == 'D':
            self.decelerate()

        return action

    def switch_lane(self, direction):
        directions = self.available_directions
        if direction == 'R':
            if 'R' in directions:
                if self.lane < 7:
                    self.switching_lane = self.lane + 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return 'M'
        if direction == 'L':
            if 'L' in directions:
                if self.lane > 1:
                    self.switching_lane = self.lane - 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return 'M'
        return direction

    def identify_available_moves(self):
        self.max_speed = -1
        moves = ['M', 'A', 'D']
        directions = ['M', 'L', 'R']
        if self.switching_lane >= 0:
            directions = ['M']
        if self.lane == 1 and 'L' in directions:
            directions.remove('L')
        if self.lane == 7 and 'R' in directions:
            directions.remove('R')

        max_box = int(math.ceil(self.y / 10.0)) - 1
        # Front checking
        for i in range(-1, 7):
            if 0 <= max_box + i < 100:
                if self.lane_map[max_box + i][self.lane - 1] != 0 and self.lane_map[max_box + i][self.lane - 1] != self:
                    car_in_front = self.lane_map[max_box + i][self.lane - 1]
                    if 'A' in moves:
                        moves.remove('A')
                    if car_in_front.speed < self.speed:
                        if 'M' in moves:
                            moves.remove('M')
                        self.emergency_brake = self.speed - car_in_front.speed
                        self.max_speed = car_in_front.speed
                    break
        # Consider car in target switching lane
        for i in range(-1, 7):
            if 0 <= max_box + i < 100:
                if self.switching_lane > 0:
                    if self.lane_map[max_box + i][self.switching_lane - 1] != 0 and self.lane_map[max_box + i][
                                self.switching_lane - 1] != self:
                        if 'A' in moves:
                            moves.remove('A')
                        car_in_front = self.lane_map[max_box + i][self.switching_lane - 1]
                        if car_in_front.speed < self.speed:
                            if 'M' in moves:
                                moves.remove('M')
                            # emergency_brake = self.speed - car_in_front.speed
                            self.max_speed = car_in_front.speed \
                                if self.max_speed == -1 or self.max_speed > car_in_front.speed else self.max_speed

        # Left lane checking
        if 'L' in directions:
            for i in range(0, 9):
                if 0 <= max_box + i < 100:
                    if self.lane_map[max_box + i][self.lane - 2] != 0:
                        directions.remove('L')
                        break

        # Right lane checking
        if 'R' in directions:
            for i in range(0, 9):
                if 0 <= max_box + i < 100:
                    if self.lane_map[max_box + i][self.lane] != 0:
                        directions.remove('R')
                        break
        self.available_moves = moves
        self.available_directions = directions

        return moves, directions

    def random(self):
        moves, directions = self.identify_available_moves()

        ds = np.random.choice(direction_weight.keys(), 3, p=direction_weight.values())
        ms = np.random.choice(move_weight.keys(), 3, p=move_weight.values())
        for d in ds:
            if d in directions:
                self.switch_lane(d)
                break

        for m in ms:
            if m in moves:
                self.move(m)
                break

    def relative_pos_subject(self):
        if self.is_subject:
            if self.emergency_brake is not None and self.emergency_brake > EMERGENCY_BRAKE_MAX_SPEED_DIFF:
                self.score.brake_penalty()
                self.hard_brake_count += 1
            self.emergency_brake = None
            return
        dvdt = self.speed - self.subject.speed
        dmds = dvdt / 3.6
        dbdm = 1.0 / 0.25
        dsdf = 1.0 / 50.0
        dmdf = dmds * dsdf
        dbdf = dbdm * dmdf * 10.0
        self.y = self.y - dbdf

        if DEFAULT_CAR_POS - dbdf <= self.y < DEFAULT_CAR_POS:
            self.score.subtract()
        elif DEFAULT_CAR_POS - dbdf > self.y >= DEFAULT_CAR_POS:
            self.score.add()
        self.score.penalty()

    def decide(self, end_episode, cache=False, is_training=True):
        if self.subject is None:
            q_values, result = self.player.decide_with_vision(self.get_vision(),
                                                  self.score.score,
                                                  end_episode,
                                                  cache=cache,
                                                  is_training=is_training)
            # Check for recent lane switching
            if result == 'L' or result == 'R':
                if (result == 'L' and 4 in self.player.agent.previous_actions) or \
                        (result == 'R' and 3 in self.player.agent.previous_actions):
                    self.score.switching_lane_penalty()
                    self.alternate_line_switching += 1
            return q_values, result
        else:
            return self.player.decide(end_episode, cache=cache)

    def draw(self):
        self.relative_pos_subject()
        self.check_switch_lane()
        if VISUALENABLED:
            self.surface.blit(self.sprite, (self.x, self.y, 34, 70))

    def get_vision(self):
        min_x = min(max(0, self.lane - 1 - VISION_W), 6)
        max_x = min(max(0, self.lane - 1 + VISION_W), 6)
        input_min_xx = self.lane - 1 - VISION_W
        input_max_xx = self.lane - 1 + VISION_W

        input_min_y = int(math.floor(self.y / 10.0)) - VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        cars_in_vision = set([
            (self.lane_map[y][x].lane - 1, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0])

        vision = np.zeros((100, 7), dtype=np.int)
        for car in cars_in_vision:
            for y in range(7):
                vision[car[1] + y][car[0]] = 1

        # Crop vision from lane_map
        vision = vision[min_y: max_y + 1, min_x: max_x + 1]

        # Add padding if required
        vision = np.pad(vision,
                        ((min_y - input_min_y, input_max_y - max_y), (min_x - input_min_xx, input_max_xx - max_x)),
                        'constant',
                        constant_values=(-1))

        vision = np.reshape(vision, [VISION_F + VISION_B + 1, VISION_W * 2 + 1, 1])
        return vision

    def get_subjective_vision(self):
        min_x = min(max(0, self.lane - 1 - VISUAL_VISION_W), 6)
        max_x = min(max(0, self.lane - 1 + VISUAL_VISION_W), 6)
        input_min_xx = self.lane - 1 - VISUAL_VISION_W
        input_max_xx = self.lane - 1 + VISUAL_VISION_W

        input_min_y = int(math.floor(self.y / 10.0)) - VISUAL_VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISUAL_VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        cars = [
            (self.lane_map[y][x].lane, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0 and self.lane_map[y][x].subject is not None]

        return cars
