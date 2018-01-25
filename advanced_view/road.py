import pygame
import os


class AdvancedRoad:
    def __init__(self, surface, origin_x, origin_y, width, height):
        self.surface = surface
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height

        self.road_image = pygame.image.load(os.path.join('./advanced_view/images/road_perspective_0.png'))
        self.road_image = pygame.transform.scale(self.road_image, (2150, 515))

        self.marker_images = []
        marker_image = pygame.image.load(os.path.join('./advanced_view/images/marker_0.png'))
        marker_image = pygame.transform.scale(marker_image, (2150, 515))
        self.marker_images.append(marker_image)
        marker_image = pygame.image.load(os.path.join('./advanced_view/images/marker_1.png'))
        marker_image = pygame.transform.scale(marker_image, (2150, 515))
        self.marker_images.append(marker_image)

        self.subject_car_image = pygame.image.load(os.path.join('./advanced_view/images/tesla.png'))
        self.subject_car_image = pygame.transform.scale(self.subject_car_image, (165, 112))

        self.road_view = None

    def animate_road_marker(self, frame, angle=0):
        return
        size = self.road_image.get_rect().size
        marker_driving_view = pygame.Surface((1010, 500), pygame.SRCALPHA, 32)
        marker_driving_view.blit(self.marker_images[frame % 2], (0, 0), (size[0] / 4.0, 0, 3 / 4.0 * size[0], size[1]))
        marker_driving_view = marker_driving_view.convert_alpha()
        self.surface.blit(marker_driving_view, ((self.origin_x, self.origin_y), (self.width, self.height)))

    def draw_road(self, frame, redraw=False, angle=0):
        return
        if redraw or self.road_view is None:
            size = self.road_image.get_rect().size
            driving_view = pygame.Surface((1010, 500), pygame.SRCALPHA, 32)
            driving_view.blit(self.road_image, (0, 0), (size[0] / 4.0, 0, 3 / 4.0 * size[0], size[1]))
            # driving_view.blit(marker_images[frame % 2], (0, 0), (size[0]/4.0, 0, 3 / 4.0 * size[0], size[1]))
            self.road_view = driving_view.convert_alpha()
        self.surface.blit(self.road_view, ((self.origin_x, self.origin_y), (self.width, self.height)))

    def draw_subject_car(self):
        return
        self.surface.blit(self.subject_car_image, (self.origin_x + 460, self.origin_y + 350))
