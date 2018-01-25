import pygame

pygame.init()
screen = pygame.display.set_mode((300, 300))
ck = (127, 33, 33)
size = 25
while True:

    pygame.draw.circle(screen, (255, 0, 0), (size, size), size, 2)

    pygame.event.poll()
    pygame.display.flip()