import numpy as np
import sys
sys.path.append("..")
import keras
from keras.utils.vis_utils import plot_model
from game2048.displays import Display,IPythonDisplay
from game2048.game import Game


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

model = keras.models.load_model('model.h5')

class MyAgent(Agent):

    def __init__(self,game,display=None):
        super().__init__(game,display)

    def step(self):

        direction = np.argmax(model.predict)
        return direction


OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i : i for i in range(1,CAND)}
map_table[0] = 0

def grid_ohe(arr):
    ret = np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,arr[r,c]] = 1
    return ret

class AnotherAgent(Agent):

    def __init__(self,game,display=None):
        super().__init__(game,display)
        self.testgame = Game(4,random=False)
        self.testgame.enable_rewrite_board = True

    def step(self):
        piece = [map_table[k] for k in self.game.board.astype(int).flatten().tolist()]
        x0 = np.array([ grid_ohe(np.array(piece).reshape(4,4)) ])
        preds = list(model.predict(x0))
        direction = np.argmax(preds[0])
        return direction
