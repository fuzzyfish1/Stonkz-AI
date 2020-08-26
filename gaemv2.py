import random
import sys
import Colors
from Neural import *
import json

show = True
if show:
    import pygame
    screen = pygame.display

FPS = 1000
gamerunning = False
pxrow = 15
pxcol = 15
field = []
columns = 15
rows = 15
totalapples = 0
tiks = 1
paused = False
score = 0
f = []
delay = 70
filename = "highscores.json"
keys = []
appleposition = (0,0)
highscore = 0

if show:
    import pygame

# tested works
def setup():
    global field, pxcol, pxrow, gamerunning, screen, s, columns, rows, filename, highscore,show

    field = ls.fill2D(num=0, rows=rows, cols=columns)
    if show:
        pygame.init()
    try:
        highscore = ls.deserialize(filename)
        print("ur previous highscore was " + str(highscore))
    except:
        highscore = 0
        ls.serialize(filepath=filename, obj=highscore)



    gamerunning = True

setup()

def screensetup():
    global field, pxcol, pxrow, gamerunning, screen, s, columns, rows, filename, highscore

    screen = pygame.display.set_mode([columns * pxcol, rows * pxrow], pygame.RESIZABLE)
    pygame.display.set_caption("SNAYKE")
    pygame.display.set_icon(pygame.image.load('snake.png'))

if show:
    screensetup()

def newfood():
    global totalapples, field, appleposition, rows,columns

    if totalapples < (rows * columns - score):
        while True:
            r = random.randrange(0, rows)
            c = random.randrange(0, columns)
            if field[r][c] == 0:
                field[r][c] = 2
                totalapples += 1
                appleposition = (r, c)
                break
            else:
                pass

class snake(object):

    # tested works

    def __init__(self, pos=[7, 7], think=True, color=4):
        global rows,columns

        self.color = color
        self.think = think

        self.snaykbrain = AI("conf.json")

        field[pos[0]][pos[1]] = color
        self.crashticks = 0
        self.length = 1
        self.dirnx = 0
        self.dirny = 0
        self.pos = pos
        self.lastdistance = 0

        self.snaykfield = []
        self.snaykfield = ls.fill2D(num=0,rows=rows,cols = columns)
        self.snaykfield[pos[0]][pos[1]] = 1

    def snakeframe(self):
        if self.think is True:
            self.moveself()
        elif self.think is False:
            self.usermove()
        self.snakefieldoperations()

    def usermove(self):
        global totalapples, tiks, field, score, keys

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if self.dirnx == 1 and self.dirny == 0:
                pass
            else:
                self.dirnx = -1
                self.dirny = 0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if self.dirnx == -1 and self.dirny == 0:
                pass
            else:
                self.dirnx = 1
                self.dirny = 0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            if self.dirnx == 0 and self.dirny == -1:
                pass
            else:
                self.dirny = 1
                self.dirnx = 0
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            if self.dirnx == 0 and self.dirny == 1:
                pass
            else:
                self.dirny = -1
                self.dirnx = 0
        elif keys[pygame.K_SPACE]:  # allows for pausing of the snake
            self.dirnx = 0
            self.dirny = 0

    def die(self):

        global gamerunning, filename, field, totalapples,tiks,highscore
        #gamerunning = False
        print("you died             "+" your score : " + str(score))

        tiks = 1
        totalapples = 0

        field = ls.fill2D(num=0, rows=rows, cols=columns)

        self.pos = [random.randrange(0,rows),random.randrange(0,columns)]
        field[self.pos[0]][self.pos[1]] = self.color
        self.crashticks = 0
        self.length = 1
        self.dirnx = 0
        self.dirny = 0
        self.lastdistance = 0

        self.snaykfield = ls.fill2D(num=0, rows=rows, cols=columns)
        self.snaykfield[self.pos[0]][self.pos[1]] = 1
        print("died")

        newfood()

        with open(filename, "r") as highscorefile:
            highscore = json.load(highscorefile)
            if score > highscore:
                with open(filename, "w+") as highscorefile:
                #print("you beat your last highscore ;)")
                    json.dump(score, highscorefile, indent=4)
            else:
                pass
                # print("ur last highscore was " + str(highscore))

    def snakefieldoperations(self):
        global totalapples, tiks, field, score, appleposition
        cxold = self.pos[0]
        cyold = self.pos[1]

        if self.dirnx == 0 and self.dirny == 0:  # also for stopping of the snake
            pass
        else:

            self.newpos = (cxold + self.dirnx, cyold + self.dirny)
            cx = self.newpos[0]
            cy = self.newpos[1]

            if cx not in range(len(field)) or cy not in range(len(field[0])):  # crashing into walls
                self.die()
                self.crashticks += 1
            elif self.snaykfield[cx][cy] != 0:  # crashing into self
                self.die()
            elif field[cx][cy] == 2:  # eaing an apple

                self.length += 1
                score = self.length
                totalapples += -1
                self.snaykfield[cx][cy] = tiks
                field[cx][cy] = self.color
                self.pos = [cx, cy]
                tiks += 1
                newfood()
            elif self.snaykfield[cx][cy] == 0:
                tiks += 1
                self.snaykfield[cx][cy] = tiks
                self.pos = [cx, cy]
                field[cx][cy] = self.color
                g = (tiks - self.length)
                for d in range(len(self.snaykfield)):
                    for a in range(len(self.snaykfield[d])):
                        if self.snaykfield[d][a] <= g and self.snaykfield[d][a] != 0:
                            self.snaykfield[d][a] = 0
                            field[d][a] = 0

            # field = self.snaykfield

    def moveself(self):
        global totalapples, tiks, field, score,appleposition

        g = []
        for n in range(-1,2):
            for x in range(-1,2):
                try:
                    g.append(field[self.pos[0]+n][self.pos[1]+x])
                except:
                    g.append(9)

        inputdata = ls.vectorize([self.length,appleposition[0],appleposition[1], self.dirnx,self.dirny,g])
        #print("inputdata size: "+ str(inputdata))
        predictions = self.snaykbrain.predict(data= inputdata)

        if predictions[0] >= predictions[1] and predictions[0] > predictions[2] and predictions[0] > predictions[3]:
            # left
            if self.dirnx == 1 and self.dirny == 0:
                pass
            else:
                self.dirnx = -1
                self.dirny = 0
        elif predictions[1] > predictions[0] and predictions[1] > predictions[2] and predictions[1] > predictions[3]:
            # right
            if self.dirnx == -1 and self.dirny == 0:
                pass
            else:
                self.dirnx = 1
                self.dirny = 0
        elif predictions[2] > predictions[1] and predictions[2] > predictions[0] and predictions[2] >= predictions[3]:
            #  up
            if self.dirnx == 0 and self.dirny == -1:
                pass
            else:
                self.dirny = 1
                self.dirnx = 0
        elif predictions[3] > predictions[1] and predictions[3] > predictions[2] and predictions[3] > predictions[0]:
            #down
            if self.dirnx == 0 and self.dirny == 1:
                pass
            else:
                self.dirny = -1
                self.dirnx = 0
        else:  # allows for pausing of the snake
            print(predictions)
            self.die()

        def onetarget(tdirnx, tdirny):
            cxold = self.pos[0]
            cyold = self.pos[1]
            self.newpos = (cxold + tdirnx, cyold + tdirny)
            cx = self.newpos[0]
            cy = self.newpos[1]

            if cx not in range(len(field)) or cy not in range(len(field[0])):  # crashing into walls
                reward = -1
            elif self.snaykfield[cx][cy] != 0:  # crashing into self
                reward = -1
            elif field[cx][cy] == 2:  # eaing an apple
                reward = 1
            elif self.snaykfield[cx][cy] == 0:
                steps = abs(appleposition[0] - cx) + abs(appleposition[1] - cy)
                reward = (2 - ((2 / ((columns -1)*(rows -1))) * steps)) -1
            #print(reward)
            return reward

        lefttarget = onetarget(-1, 0)
        righttarget = onetarget(1, 0)
        uptarget = onetarget(0, 1)
        downtarget = onetarget(0, -1)

        targets = [[lefttarget], [righttarget], [uptarget], [downtarget]]

        self.snaykbrain.backpropagate(input="none", output=targets)
<<<<<<< HEAD
        d ="longth: "+ str(self.length)+ "  high: "+str(highscore)+"  avgcost: "+ str(self.snaykbrain.avgcost)
=======
        #print("\r")
        d = "longth: "+ str(self.length)+" high: "+str(highscore) +"avgcost: "+str(self.snaykbrain.avgcost)
>>>>>>> 0f091f5934dbf03ffa0988efdee3d61caa84dd67
        print(d)



    '''    def displaysnaykfield(self):
        global screen, pxcol, pxrow
        screen.fill(Colors.BLACK)
        x = 0
        y = 0
        rows = len(self.snaykfield[0])
        columns = len(self.snaykfield)

        # draws the grid that
        u = 0
        l = 0
        color = Colors.WHITE
        for u in range(rows):
            for l in range(columns):
                if self.snaykfield[l][u] == 0:  # draw fo empty square
                    color = Colors.PINK
                    thiqness = 1
                elif self.snaykfield[l][u] == 1:  # draw fo snakyboi
                    color = Colors.BLUE
                    thiqness = 0
                elif self.snaykfield[l][u] == 2:  # draw fo apple
                    color = Colors.GREEN
                    thiqness = 0

                elif self.snaykfield[l][u] == 3:
                    color = Colors.DARK_BLUE
                    thiqness = 5
                elif self.snaykfield[l][u] == 4:
                    color = Colors.RED
                    thiqness = 5
                elif self.snaykfield[l][u] == 5:
                    color = Colors.YELLOW
                    thiqness = 5
                elif self.snaykfield[l][u] == 6:
                    color = Colors.ORANGE
                    thiqness = 5

                pygame.draw.rect(screen, color, (x, y, pxrow, pxcol), thiqness)
                x += pxcol
            y += pxrow
            x = 0

        pygame.display.flip()'''

s = snake()

# s.__init__()

# needs to apple the field get out of loop when it is completely appledd

def displayfield():
    global screen, pxcol, pxrow, field, rows,columns
    screen.fill(Colors.BLACK)
    x = 0
    y = 0

    # draws the grid that
    u = 0
    l = 0
    for u in range(rows):

        for l in range(columns):
            if field[l][u] == 0:  # draw fo empty square
                color = Colors.PINK
                thiqness = 1
            elif field[l][u] == 1:  # draw fo snakyboi
                color = Colors.BLUE
                thiqness = 0
            elif field[l][u] == 2:  # draw fo apple
                color = Colors.GREEN
                thiqness = 0
            elif field[l][u] == 3:
                color = Colors.DARK_BLUE
                thiqness = 0
            elif field[l][u] == 4:
                color = Colors.RED
                thiqness = 0
            elif field[l][u] == 5:
                color = Colors.YELLOW
                thiqness = 0
            elif field[l][u] == 5:
                color = Colors.ORANGE
                thiqness = 0
            else:
                color = (l * (255/columns), 25, u *(255/rows))
                thiqness = 0

            pygame.draw.rect(screen, color, (x, y, pxrow, pxcol), thiqness)
            x += pxcol
        y += pxrow
        x = 0

    pygame.display.flip()

def main():
    global FPS, gamerunning, s, field, keys,show
    newfood()
    if show:
        clock = pygame.time.Clock()
    while gamerunning:
        if show:
            for event in pygame.event.get():
                keys = pygame.key.get_pressed()
                if event.type == pygame.QUIT:
                    gamerunning = False
                    pygame.quit()
                    sys.exit()

        s.snakeframe()
        if show:
            displayfield()
            clock.tick(FPS)

main()

# init
