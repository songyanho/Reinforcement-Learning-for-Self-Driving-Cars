import pygame
from pygame import gfxdraw
import os
from PIL import Image
from PIL import ImageDraw

from gui_util import draw_dashed_line_delay

ROAD_HEIGHT = 250.0


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
         "green": pygame.Color(32, 76, 1),
         "black": pygame.Color(0, 0, 0),
         "tunnel": pygame.Color(38, 15, 8),
         "brown": pygame.Color(124, 69, 1)}


class AdvancedRoad:
    def __init__(self, surface, origin_x, origin_y, width, height, lane=6):
        self.surface = surface
        self.sky_x = 0
        self.sky_y = 0
        self.sky_height = 800 - ROAD_HEIGHT
        self.sky_width = width
        self.origin_x = origin_x # Road origin_x
        self.origin_y = origin_y # Road origin_y
        self.width = width # Road width
        self.height = height # Road height
        self.lane = lane

        self.sky_image = pygame.image.load(os.path.join('./advanced_view/images/sky_1010x550.png'))
        self.hill_image = pygame.image.load(os.path.join('./advanced_view/images/hill.png'))
        self.field_left_image = pygame.image.load(os.path.join('./advanced_view/images/field_left.png'))
        self.field_right_image = pygame.image.load(os.path.join('./advanced_view/images/field_right.png'))
        self.field_side_left_image = pygame.image.load(os.path.join('./advanced_view/images/field_side_left.png'))
        self.field_side_right_image = pygame.image.load(os.path.join('./advanced_view/images/field_side_right.png'))
        self.field_image = Image.open(os.path.join('./advanced_view/images/field.png')).convert("RGBA")
        # self.dirt_image = pygame.image.load(os.path.join('./advanced_view/images/dirt.png'))
        self.dirt_image = Image.open(os.path.join('./advanced_view/images/dirt.png')).convert("RGBA")
        self.subject_car_middle_image = pygame.image.load(os.path.join('./advanced_view/images/chev_rear.png'))
        self.subject_car_left_image = pygame.image.load(os.path.join('./advanced_view/images/chev_left.png'))
        self.subject_car_right_image = pygame.image.load(os.path.join('./advanced_view/images/chev_right.png'))
        self.object_car_middle_image = pygame.image.load(os.path.join('./advanced_view/images/civic_rear.png'))
        self.object_car_left_image = pygame.image.load(os.path.join('./advanced_view/images/civic_left.png'))
        self.object_car_right_image = pygame.image.load(os.path.join('./advanced_view/images/civic_right.png'))

        self.road_view = None

    def draw(self, frame, subject_car):
        lane = subject_car.lane
        while True:
            self.draw_sky(frame)
            self.draw_road_side(frame, self.lane)
            self.draw_road(frame, lane=self.lane)
            self.draw_cars(subject_car)
            self.draw_subject_car(self.lane - lane)
            if self.lane != lane:
                self.lane += 0.25 if lane > self.lane else - 0.25
            if abs(self.lane - lane) < 0.1:
                break
            pygame.event.poll()
            pygame.display.flip()
        self.lane = lane

    def draw_sky(self, frame):
        view = pygame.Surface((self.sky_width, self.sky_height))
        # Resize sky
        if frame % 40 == 0:
            self.sky_image = pygame.transform.scale(self.sky_image, (self.sky_image.get_size()[0]+2, self.sky_image.get_size()[1]+1))
            self.sky_image.blit(self.sky_image, ((-1, -2), (self.sky_width, self.sky_height)))
        view.blit(self.sky_image, ((0, 0), (self.sky_width, self.sky_height)))
        view.blit(self.hill_image, ((0, self.sky_height - 49), (self.sky_width, 49)))
        self.surface.blit(view, ((self.sky_x, self.sky_y), (self.sky_width, self.sky_height)))

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

    def draw_road_side(self, frame, lane):
        polygon_left = [
            (-337.0 * lane + 673.33 - 200, ROAD_HEIGHT),
            (455 - 20, 0),
            (0, 0),
            (0, ROAD_HEIGHT)]
        polygon_right = [
            (1010, ROAD_HEIGHT),
            (1010, 0),
            (555 + 20, 0),
            (-336.67 * lane + 3032 + 200, ROAD_HEIGHT)]
        maskIm = self.dirt_image.crop((0, 500 - (frame % 20) * 25, self.width, ROAD_HEIGHT + 500 - (frame % 20) * 25))
        pdraw = ImageDraw.Draw(maskIm)
        pdraw.polygon(polygon_left, fill=(255, 255, 255, 0), outline=(255, 255, 255, 0))
        pdraw.polygon(polygon_right, fill=(255, 255, 255, 0), outline=(255, 255, 255, 0))
        side = self.field_image.crop((0, 500 - (frame % 20) * 25, self.width, ROAD_HEIGHT + 500 - (frame % 20) * 25))
        side.paste(maskIm, (0, 0), mask=maskIm)
        side_pygame = pygame.image.fromstring(side.tobytes(), side.size, side.mode)
        self.surface.blit(side_pygame, ((self.origin_x, self.origin_y), (self.width, ROAD_HEIGHT)))

    def blit_mask(source, dest, destpos, mask, maskrect):
        """
        Blit an source image to the dest surface, at destpos, with a mask, using
        only the maskrect part of the mask.
        """
        tmp = source.copy()
        tmp.blit(mask, maskrect.topleft, maskrect, special_flags=pygame.BLEND_RGBA_MULT)
        dest.blit(tmp, destpos, dest.get_rect().clip(maskrect))

    def draw_road(self, frame, lane=1):
        self.road_view = pygame.Surface((1010, ROAD_HEIGHT), pygame.SRCALPHA, 32)
        self.road_view = self.road_view.convert_alpha()

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
                gfxdraw.filled_polygon(self.road_view, (
                    (int(int(i * 100.0 / 7 + 455)), 0),
                    (int(int(i * 100.0 / 7 + 455)) + 7, 0),
                    (int(left) + 7, int(ROAD_HEIGHT)),
                    (int(left), int(ROAD_HEIGHT))
                ), COLOR['white'])
            else:
                draw_dashed_line_delay(self.road_view,
                                 COLOR['white'],
                                 (i * 100.0 / 7 + 455, 0),
                                 (left, ROAD_HEIGHT),
                                 width=5,
                                 dash_length=40,
                                 delay=frame % 3)
            left += 337.0

        self.surface.blit(self.road_view, ((self.origin_x, self.origin_y), (self.width, self.height)))

    def draw_subject_car(self, direction):
        image = self.subject_car_middle_image
        if direction > 0:
            image = self.subject_car_left_image
        elif direction < 0:
            image = self.subject_car_right_image
        self.surface.blit(image, (self.origin_x + 345, self.origin_y + 70))
