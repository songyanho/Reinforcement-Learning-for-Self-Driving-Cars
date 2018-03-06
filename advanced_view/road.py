import pygame
import os
import math
from gui_util import draw_dashed_line

ROAD_HEIGHT = 300.0


COLOR = {"white": pygame.Color(255, 255, 255),
         "opaque_white": pygame.Color(255, 255, 255, 80),
         "text": pygame.Color(172, 199, 252),
         "dark_text": pygame.Color(57, 84, 137),
         "selection": [pygame.Color(172, 199, 252), pygame.Color(100, 149, 252)],
         "sky": pygame.Color(10, 10, 10),
         "gutter": pygame.Color(100, 100, 100),
         "red": pygame.Color(204, 0, 0),
         "bonus_a": pygame.Color(255, 78, 0),
         "bonus_b": pygame.Color(255, 178, 0),
         "green": pygame.Color(0, 204, 0),
         "black": pygame.Color(0, 0, 0),
         "tunnel": pygame.Color(38, 15, 8)}


class AdvancedRoad:
    def __init__(self, surface, origin_x, origin_y, width, height, lane=6):
        self.surface = surface
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height
        self.lane = lane

        # self.road_image = pygame.image.load(os.path.join('./advanced_view/images/road_perspective_0.png'))
        # self.road_image = pygame.transform.scale(self.road_image, (2150, 515))
        #
        # self.marker_images = []
        # marker_image = pygame.image.load(os.path.join('./advanced_view/images/marker_0.png'))
        # marker_image = pygame.transform.scale(marker_image, (2150, 515))
        # self.marker_images.append(marker_image)
        # marker_image = pygame.image.load(os.path.join('./advanced_view/images/marker_1.png'))
        # marker_image = pygame.transform.scale(marker_image, (2150, 515))
        # self.marker_images.append(marker_image)

        self.subject_car_middle_image = pygame.image.load(os.path.join('./advanced_view/images/chev_rear.png'))
        # self.subject_car_middle_image = pygame.transform.scale(self.subject_car_middle_image, (165, 112))

        self.subject_car_left_image = pygame.image.load(os.path.join('./advanced_view/images/chev_left.png'))
        # self.subject_car_left_image = pygame.transform.scale(self.subject_car_left_image, (165, 112))

        self.subject_car_right_image = pygame.image.load(os.path.join('./advanced_view/images/chev_right.png'))
        # self.subject_car_right_image = pygame.transform.scale(self.subject_car_right_image, (165, 112))

        self.object_car_middle_image = pygame.image.load(os.path.join('./advanced_view/images/civic_rear.png'))
        # self.object_car_middle_image = pygame.transform.scale(self.object_car_middle_image, (165, 112))

        self.object_car_left_image = pygame.image.load(os.path.join('./advanced_view/images/civic_left.png'))
        # self.object_car_left_image = pygame.transform.scale(self.object_car_left_image, (165, 112))

        self.object_car_right_image = pygame.image.load(os.path.join('./advanced_view/images/civic_right.png'))
        # self.object_car_right_image = pygame.transform.scale(self.object_car_right_image, (165, 112))

        self.road_view = None

    def draw(self, frame, subject_car):
        lane = subject_car.lane
        while True:
            self.draw_road(frame, lane=self.lane)
            self.animate_road_marker(frame)
            self.draw_cars(subject_car)
            self.draw_subject_car(self.lane - lane)
            if self.lane != lane:
                self.lane += 0.25 if lane > self.lane else - 0.25
            if abs(self.lane - lane) < 0.1:
                break
            pygame.event.poll()
            pygame.display.flip()
        self.lane = lane

    def draw_cars(self, subject_car):
        view = pygame.Surface((1010, ROAD_HEIGHT), pygame.SRCALPHA, 32)
        view = view.convert_alpha()
        camera_lane = subject_car.lane
        cars = subject_car.get_subjective_vision()
        for car in cars:
            lane = car[0]
            y = car[1]
            relative_y = y - 42
            if relative_y < 0:
                continue

            image = self.object_car_middle_image
            ratio = 231.0 / 328.0
            if lane != camera_lane:
                if lane > camera_lane:
                    image = self.object_car_right_image
                else:
                    image = self.object_car_left_image
            pt_top_left = (lane - 1) * 100.0 / 7 + 455
            pt_top_right = lane * 100.0 / 7 + 455
            pt_bottom_left = -337.0 * camera_lane + 673.33 + (lane - 1) * 337.0
            pt_bottom_right = -337.0 * camera_lane + 673.33 + lane * 337.0
            # distance_left = math.sqrt(math.pow(pt_top_left-pt_bottom_left, 2) + math.pow(0, ROAD_HEIGHT))
            # distance_right = math.sqrt(math.pow(pt_top_right-pt_bottom_right, 2) + math.pow(0, ROAD_HEIGHT))
            target_y = ROAD_HEIGHT * relative_y / 28.0
            target_bottom_left_x = pt_top_left + target_y / ROAD_HEIGHT * (pt_bottom_left - pt_top_left)
            target_bottom_right_x = pt_top_right + target_y / ROAD_HEIGHT * (pt_bottom_right - pt_top_right)
            image_width = int(target_bottom_right_x - target_bottom_left_x - 10)
            image_width = image_width if image_width > 0 else 0
            image_height = int(image_width * ratio)
            target_top_left_x = int(target_bottom_left_x + 5)
            target_top_left_y = int(target_y - image_height)
            a = pygame.transform.scale(image, (image_width, image_height))
            view.blit(a, (target_top_left_x, target_top_left_y))
        self.surface.blit(view, ((self.origin_x, self.origin_y), (self.width, self.height)))

    def animate_road_marker(self, frame, angle=0):
        pass
        # size = self.road_image.get_rect().size
        # marker_driving_view = pygame.Surface((1010, ROAD_HEIGHT), pygame.SRCALPHA, 32)
        # marker_driving_view.blit(self.marker_images[frame % 2], (0, 0), (size[0] / 4.0, 0, 3 / 4.0 * size[0], size[1]))
        # marker_driving_view = marker_driving_view.convert_alpha()
        # self.surface.blit(marker_driving_view, ((self.origin_x, self.origin_y), (self.width, self.height)))

    def draw_road(self, frame, lane=1):
        self.road_view = pygame.Surface((1010, ROAD_HEIGHT))
        self.road_view.fill(COLOR['red'])

        pygame.draw.polygon(self.road_view, COLOR['black'],
                            [
                                (-337.0 * lane + 673.33, ROAD_HEIGHT),
                                (455, 0),
                                (555, 0),
                                (-336.67 * lane + 3032, ROAD_HEIGHT)
                            ])

        left = -337.0 * lane + 673.33
        for i in range(8):
            if i == 0 or i == 7:
                pygame.draw.line(self.road_view,
                                 COLOR['white'],
                                 (i * 100.0 / 7 + 455, 0),
                                 (left, ROAD_HEIGHT),
                                 5)
            else:
                draw_dashed_line(self.road_view,
                                 COLOR['white'],
                                 (i * 100.0 / 7 + 455, 0),
                                 (left, ROAD_HEIGHT),
                                 width=5,
                                 dash_length=7,
                                 fsnl=True)
            left += 337.0

        self.surface.blit(self.road_view, ((self.origin_x, self.origin_y), (self.width, self.height)))

    def draw_subject_car(self, direction):
        image = self.subject_car_middle_image
        if direction > 0:
            image = self.subject_car_left_image
        elif direction < 0:
            image = self.subject_car_right_image
        self.surface.blit(image, (self.origin_x + 345, self.origin_y + 70))
