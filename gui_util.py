import math
import os
import pygame
from PIL import Image
from advanced_view.gauge import GaugeDraw
from pygame import gfxdraw
from random import randint

from car import MAX_SPEED, VISION_F, VISION_W, VISION_B
import config
from config import ROAD_VIEW_OFFSET, INPUT_VIEW_OFFSET_X, INPUT_VIEW_OFFSET_Y


pygame.font.init()
font_28 = pygame.font.Font(os.path.join('./advanced_view/fonts/digitize.ttf'), 28)
font_60 = pygame.font.Font(os.path.join('./advanced_view/fonts/digitize.ttf'), 60)


class Point:
    # constructed using a normal tupple
    def __init__(self, point_t = (0,0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])

    # define all useful operators
    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))

    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))

    def __mul__(self, scalar):
        return Point((self.x*scalar, self.y*scalar))

    def __div__(self, scalar):
        return Point((self.x/scalar, self.y/scalar))

    def __len__(self):
        return int(math.sqrt(self.x**2 + self.y**2))

    # get back values in original tuple format
    def get(self):
        return self.x, self.y


black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
grey = pygame.Color(128, 128, 128)
yellow = pygame.Color(255, 255, 0, 128)
orange = pygame.Color(255, 140, 0, 128)

IMAGE_PATH = './images'

if config.VISUALENABLED:
    accelerate_on = pygame.image.load(os.path.join(IMAGE_PATH, 'accelerate_on.png'))
    accelerate_off = pygame.image.load(os.path.join(IMAGE_PATH, 'accelerate_off.png'))
    brake_on = pygame.image.load(os.path.join(IMAGE_PATH, 'brake_on.png'))
    brake_off = pygame.image.load(os.path.join(IMAGE_PATH, 'brake_off.png'))
    left_on = pygame.image.load(os.path.join(IMAGE_PATH, 'left_on.png'))
    left_off = pygame.image.load(os.path.join(IMAGE_PATH, 'left_off.png'))
    right_on = pygame.image.load(os.path.join(IMAGE_PATH, 'right_on.png'))
    right_off = pygame.image.load(os.path.join(IMAGE_PATH, 'right_off.png'))


def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    if not config.VISUALENABLED:
        return

    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = len(displacement)
    slope = displacement/length
    loop = length / dash_length

    for index in range(0, loop, 2):
        start = origin + (slope * index * dash_length)
        end = origin + (slope * (index + 1) * dash_length)
        gfxdraw.filled_polygon(surf, (
            (int(start.x), int(start.y)),
            (int(start.x) + width, int(start.y)),
            (int(end.x) + width, int(end.y)),
            (int(end.x), int(end.y))
        ), color)


def draw_dashed_line_delay(surf, color, start_pos, end_pos, width=1, dash_length=10, delay=0):
    if not config.VISUALENABLED:
        return

    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = len(displacement)
    slope = displacement/length
    loop = length / dash_length

    origin = origin + (slope * delay * 10)

    for index in range(0, loop + 1, 2):
        start = origin + (slope * index * dash_length)
        end = origin + (slope * (index + 1) * dash_length)
        gfxdraw.filled_polygon(surf, (
            (int(start.x), int(start.y)),
            (int(start.x) + width, int(start.y)),
            (int(end.x) + width, int(end.y)),
            (int(end.x), int(end.y))
        ), color)


def draw_basic_road(surface, speed):
    if not config.VISUALENABLED:
        return

    surface.fill(white)
    # Left most lane marking
    pygame.draw.line(surface, black, (ROAD_VIEW_OFFSET + 13, -10), (ROAD_VIEW_OFFSET + 13, 1000), 5)
    # Right most lane marking
    pygame.draw.line(surface, black, (ROAD_VIEW_OFFSET + 367, -10), (ROAD_VIEW_OFFSET + 367, 1000), 5)

    line_marking_offset = randint(0, 10)
    for l in range(1, 7):
        draw_dashed_line(
            surface,
            grey,
            (ROAD_VIEW_OFFSET + l * 50 + 15, int((speed/(MAX_SPEED * 1.0)) * -1 * line_marking_offset)),
            (ROAD_VIEW_OFFSET + l * 50 + 15, 1000),
            width=1,
            dash_length=5
        )


def draw_road_overlay_safety(surface, lane_map):
    if not config.VISUALENABLED:
        return

    # Draw on surface
    for y in range(100):
        for x in range(7):
            if lane_map[y][x] != 0:
                pygame.draw.rect(surface, yellow, (ROAD_VIEW_OFFSET + x * 50 + 15 + 1, y * 10, 49, 10))
                pygame.draw.rect(surface, grey, (ROAD_VIEW_OFFSET + x * 50 + 15 + 1, y * 10, 49, 10), 1)


def draw_road_overlay_vision(surface, subject_car):
    if not config.VISUALENABLED:
        return

    # Draw on surface
    min_x = min(max(0, subject_car.lane - VISION_W - 1), 6)
    max_x = min(max(0, subject_car.lane + VISION_W - 1), 6)

    min_y = min(max(0, int(math.ceil(subject_car.y / 10.0)) - VISION_F - 1), 100)
    max_y = min(max(0, int(math.ceil(subject_car.y / 10.0)) + VISION_B - 1), 100)

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            pygame.draw.rect(surface, orange, (ROAD_VIEW_OFFSET + x * 50 + 15 + 1, y * 10, 49, 10))
            pygame.draw.rect(surface, grey, (ROAD_VIEW_OFFSET + x * 50 + 15 + 1, y * 10, 49, 10), 1)


def control_car(target_car, keydown):
    if keydown == pygame.K_UP:
        target_car.move('A')
    elif keydown == pygame.K_DOWN:
        target_car.move('D')
    else:
        target_car.move('M')

    if keydown == pygame.K_LEFT:
        target_car.switch_lane('L')
    elif keydown == pygame.K_RIGHT:
        target_car.switch_lane('R')


def identify_free_lane(cars):
    lanes = [[n for n in range(1, 8)] for _ in range(2)]
    for car in cars:
        if -170 <= car.y <= 0:
            if car.lane in lanes[0]:
                lanes[0].remove(car.lane)
            if car.switching_lane in lanes[0]:
                lanes[0].remove(car.switching_lane)
        elif 930 <= car.y <= 1070:
            if car.lane in lanes[1]:
                lanes[1].remove(car.lane)
            if car.switching_lane in lanes[1]:
                lanes[1].remove(car.switching_lane)

    return lanes


def draw_inputs(surface, vision):
    vision_title = font_28.render("Vision:", False, (0, 0, 0))
    surface.blit(vision_title, (INPUT_VIEW_OFFSET_X - 10, INPUT_VIEW_OFFSET_Y))
    for y_i in range(len(vision)):
        for x_i, x in enumerate(vision[y_i]):
            pygame.draw.rect(surface, orange if x != 0 else white,
                             (INPUT_VIEW_OFFSET_X + x_i * 10 + 80, INPUT_VIEW_OFFSET_Y + y_i * 10, 10, 10))
            pygame.draw.rect(surface, grey,
                             (INPUT_VIEW_OFFSET_X + x_i * 10 + 1 + 80, INPUT_VIEW_OFFSET_Y + y_i * 10, 10, 10), 1)


def draw_actions(surface, action):
    action_title = font_28.render("Action:", False, (0, 0, 0))
    surface.blit(action_title, (INPUT_VIEW_OFFSET_X - 10, INPUT_VIEW_OFFSET_Y + 370))

    surface.blit(left_on if action == 'L' else left_off,
                 (INPUT_VIEW_OFFSET_X + 80, INPUT_VIEW_OFFSET_Y + 370, 34, 70))
    surface.blit(right_on if action == 'R' else right_off,
                 (INPUT_VIEW_OFFSET_X + 80 + 40, INPUT_VIEW_OFFSET_Y + 370, 34, 70))
    surface.blit(brake_on if action == 'D' else brake_off,
                 (INPUT_VIEW_OFFSET_X + 80, INPUT_VIEW_OFFSET_Y + 410, 34, 70))
    surface.blit(accelerate_on if action == 'A' else accelerate_off,
                 (INPUT_VIEW_OFFSET_X + 80 + 40, INPUT_VIEW_OFFSET_Y + 410, 34, 70))


class Score:
    def __init__(self, score=0):
        self.score = score

    def add(self):
        self.score += 1
        # self.score -= 0.1

    def subtract(self):
        self.score -= 1
        # self.score -= 0.1

    def penalty(self):
        # Penalty over time
        self.score += config.CONSTANT_PENALTY

    def brake_penalty(self):
        self.score += config.EMERGENCY_BRAKE_PENALTY

    def action_mismatch_penalty(self):
        self.score += config.MISMATCH_ACTION_PENALTY

    def switching_lane_penalty(self):
        self.score += config.SWITCHING_LANE_PENALTY


def draw_gauge(surface, speed):
    im = Image.new("RGB", (200, 200), (255, 255, 255, 0))
    g = GaugeDraw(im, 0, 110)
    g.render_simple_gauge(value=speed, major_ticks=10, minor_ticks=5, label="{}kmh".format(speed))

    gauge = pygame.image.fromstring(im.tobytes(), im.size, im.mode)

    speed_title = font_28.render("Speed:", False, (0, 0, 0))
    surface.blit(speed_title, (INPUT_VIEW_OFFSET_X - 10, 10))
    surface.blit(gauge, ((INPUT_VIEW_OFFSET_X - 10, 35), (200, 200)))


def draw_score(surface, score):
    score_title = font_28.render("Score:", False, (0, 0, 0))
    score = font_60.render(str(int(score)), False, (0, 0, 0))
    surface.blit(score_title, (INPUT_VIEW_OFFSET_X - 10, 245))
    surface.blit(score, (INPUT_VIEW_OFFSET_X - 10 + 80, 240))
