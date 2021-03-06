import pygame
import sys
import random
import numpy as np
import cv2
from itertools import cycle
import pygame.surfarray as surfarray
from pygame.locals import *

# Load game resources.
def load():
    # path of player with different states
    PLAYER_PATH = (
            'assets/sprites/redbird-upflap.png',
            'assets/sprites/redbird-midflap.png',
            'assets/sprites/redbird-downflap.png'
    )

    IMAGES, HITMASKS = {}, {}

    # path of background
    BACKGROUND_PATH = 'assets/sprites/background-black.png'
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # path of pipe
    PIPE_PATH = 'assets/sprites/pipe-green.png'

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    ) 

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    IMAGES['otto'] = pygame.image.load('assets/sprites/otto1.png').convert_alpha()

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    HITMASKS['otto'] = getHitmask(IMAGES['otto'])

    return IMAGES, HITMASKS

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

# Game
def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30 ,40, 50, 60, 70, 80, 90, 100]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    create_otto = np.random.rand() < 0
    otto = None
    if create_otto:
        pipes = [
            {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPEGAPSIZE[0]},  # lower pipe
        ]
        otto = {'x':pipes[1]['x'], 'y':pipes[1]['y'] - (int)((PIPEGAPSIZE[0] + OTTO_HEIGHT)/2)}
    else:
        pipes = [
            {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPEGAPSIZE[1]},  # lower pipe
        ]

    return pipes, otto

def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes, ottos):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

        for otto in ottos:
            ottoRect = pygame.Rect(otto['x'], otto['y'], OTTO_WIDTH, OTTO_HEIGHT)
            pHitMask = HITMASKS['player'][pi]
            oHitMask = HITMASKS['otto']
            oCollide = pixelCollision(playerRect, ottoRect, pHitMask, oHitMask)

            if oCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

# Configuration
FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')
IMAGES, HITMASKS = load()
# gap between upper and lower part of pipe
PIPEGAPSIZE = [200, 200] 
# PIPEGAPSIZE = [180, 180] 
# PIPEGAPSIZE = [160, 160] 
# PIPEGAPSIZE = [140, 140] 
# PIPEGAPSIZE = [120, 120]
BASEY = SCREENHEIGHT * 0.79
PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()
OTTO_HEIGHT = IMAGES['otto'].get_height()
OTTO_WIDTH = IMAGES['otto'].get_width()
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
best_score = 0

class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1, otto1 = getRandomPipe()
        newPipe2, otto2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        self.ottos = []
        if otto1 is not None:
            otto1['x'] = SCREENWIDTH
            self.ottos.append(otto1)
        if otto2 is not None:
            otto2['x'] = SCREENWIDTH + (SCREENWIDTH / 2)
            self.ottos.append(otto2)

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  8    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps

    def frame_step(self, input_action):
        global best_score

        # input_actions = 0: do nothing
        # input_actions = 1: flap the bird
        pygame.event.pump()

        reward = 0.1
        terminate = False

        # if sum(input_actions) != 1:
        #     raise ValueError('Multiple input actions!')

        # if input_actions[1] == 1:
        if input_action == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                #SOUNDS['wing'].play()

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                if self.score > best_score:
                    best_score = self.score
                #SOUNDS['point'].play()
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        for otto in self.ottos:
            otto['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe, newOtto = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])
            if newOtto is not None:
                self.ottos.append(newOtto)

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)
            
        if len(self.ottos) > 0:
            if self.ottos[0]['x'] < -PIPE_WIDTH:
                self.ottos.pop(0)

        # check if crash here
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes, self.ottos)
        if isCrash:
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            terminate = True
            self.__init__()
            reward = -1

        reward *= (1.05 ** self.score)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        for otto in self.ottos:
            SCREEN.blit(IMAGES['otto'], (otto['x'], otto['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)

        return image_data, reward, terminate

if __name__ == '__main__':
    game = GameState()
    actions = np.zeros(2)
    actions[0] = 1
    terminate = False
    max_score = -1
    # i = 0

    for _ in range(1):
        terminate = False
        while not terminate:
            img_data, r, terminate = game.frame_step(actions)
            # img_data = cv2.cvtColor(cv2.resize(img_data, (80, 80)), cv2.COLOR_BGR2GRAY)
            # _, img_data = cv2.threshold(img_data, 1, 255, cv2.THRESH_BINARY)
            # print(img_data.shape)
            # cv2.imwrite("test%d.jpg" % i, img_data)
            # i += 1
            actions = np.zeros(2)
            actions[0] = 1
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        actions[0] = 0
                        actions[1] = 1
            if max_score < game.score:
                max_score = game.score
    print("Best score: %d" % max_score)
