import keras
from keras.models import Model
import numpy as np

import random
from collections import namedtuple
from game2048.game import Game
from game2048.expectimax import board_to_move,_ext
from game2048.agents import Agent,RandomAgent,ExpectiMaxAgent
from game2048.displays import Display

OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i:i for i in range(1,CAND)}
map_table[0] = 0
vmap = np.vectorize(lambda x: map_table[x])

def grid_ohe(arr):
    ret = np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,arr[r,c]] = 1
    return ret

Guide = namedtuple('Guide',('state','action'))
#生成guide字典，如guide（3，4），则guide.state=3

class Guides:
    def __init__(self,capacity):
        #初始化Guides，memoy为空数组，self指Guides自己
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self,*args):
        #*arg是一个代号，当参数不确定时代指后面所有参数
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            #append,相当于add，把元素加在列表尾
        self.memory[self.position] = Guide(*args)
        #memory存state和action两个值
        self.position = (self.position + 1) % self.capacity

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
        #从memory中提取batch_size个值

    def ready(self,batch_size):
        return len(self.memory) >= batch_size
        #memory长度比batch_size大，返回真，否则为假

    def __len__(self):
        return len(self.memory)
        #返回mrmory的长度

class ModelWrapper:

    def __init__(self,model,capacity):
        #初始化，ModelWrapper的model等值均等于输入
        self.model = model
        self.memory = Guides(capacity)
        """self.writer = SummaryWriter()"""
        self.training_step = 0

    def predict(self,board):
        return model.predict(np.expand_dims(board,axis=0))
    #expand_dims是扩展维度，如axis=0，则在第一个维度上其长度唯一，其他继承原来的

    def move(self,game):
        ohe_board = grid_ohe(vmap(game.board))
        suggest = board_to_move(game.board)
        direction = self.predict(ohe_board).argmax()
        game.move(direction)
        self.memory.push(ohe_board,suggest)
        if random.random() < 0.3:
            game.move(suggest)
        else :
            game.move(self.predict(ohe_board).argmax())

    def train(self,batch):
        if self.memory.ready(batch):
            guides = self.memory.sample(batch)
            X = []
            Y = []
            for guide in guides:
                X.append(guide.state)
                ohe_action = [0]*4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            loss,acc= self.model.train_on_batch(np.array(X),np.array(Y))
            """self.writer.add_scalar('loss',float(loss),self.training_step)
            self.writer.add_scalar('acc',float(acc),self.training_step)"""
            self.training_step+=1


MEMORY = 32768
BATCH = 1024
ARCHIEVE = 1000

model = keras.models.load_model('model.h5')
mw = ModelWrapper(model,MEMORY)

while True:
    game = Game(4,random=False)
    while not game.end:
        mw.move(game)
    print('score:',game.score,end='\t')
    mw.train(BATCH)
    if(mw.training_step%10==0):
        model.save('model.h5')
        if(mw.training_step%ARCHIEVE==0):
            model.save('model_%d.h5'%mw.training_step)
