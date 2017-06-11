import numpy as np
import math
import os
from PIL import Image
import random

def Rest_Img(im):
    return np.uint8(127.5 * (im + 1))

class Real_DB():
    def __init__(self,db_type,batch_size,db_pt='',img_size=64,db_size=64,seed=0,crop_length=108):
        self.DB_Type=db_type
        self.Batch_size=batch_size
        self.seed=seed
        self.img_size=img_size
        self.crop_length=crop_length
        self.Restart_First_Batch()
        # Calc DB Size
        if self.DB_Type==0:
            self.pt = db_pt
            self.files = [f for f in os.listdir(self.pt) if f.endswith('.jpg')]
            self.DB_size = len(self.files)

        if self.DB_Type == 1:
            self.DB_size = db_size
            self.max_objs_in_img = 4

        self.num_of_batches=math.ceil(self.DB_size/self.Batch_size)

    def Restart_First_Batch(self):
        self.DB_Idx =0
        random.seed(self.seed)

    def Get_Next_Batch(self):
        im = np.ones((self.Batch_size,self.img_size, self.img_size,3), dtype=np.float32)
        for k in range(self.DB_Idx,min(self.DB_Idx+self.Batch_size,self.DB_size)):
            if self.DB_Type == 0:
                I=Image.open(self.pt + self.files[k])
                (w,h)=I.size
                wind=np.round([(w-self.crop_length)/2,(h-self.crop_length)/2,(w+self.crop_length)/2,(h+self.crop_length)/2]).tolist()
                I=np.array(I.crop(wind).resize((self.img_size,self.img_size)),'float32') #int8')
                im[k-self.DB_Idx, :,:,:] = (I/127.5)-1 #I.resize((img_size, img_size))

            if self.DB_Type == 1:
                # The random model here for the synthetic database must not be intefered by other random model used while running
                # because this series of random images must repeat itselfs every new epoch
                # The synthetic random model is of random package only
                # The random model of the net is based on tf and the z vecrors are based on np.random
                num_of_objs = random.randint(1,self.max_objs_in_img) # np.random.randint(self.max_objs_in_img)+1
                for recs_ind in range(num_of_objs ):
                    #coor = np.random.uniform(0, self.img_size, (4))
                    coor = [random.uniform(0, self.img_size),random.uniform(0, self.img_size),random.uniform(0, self.img_size),random.uniform(0, self.img_size)]
                    color = [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)] #np.random.uniform(-1,1,(3))
                    coor[2] = min(coor[0] + coor[2] / 2, self.img_size - 1)
                    coor[3] = min(coor[1] + coor[3] / 2, self.img_size - 1)
                    coor = np.array(coor).astype(int)
                    # Filled:
                    #im[k-self.DB_Idx, coor[0]:coor[2], coor[1]:coor[3],:] = [0,0,0]

                    # Outline:
                    im[k - self.DB_Idx, coor[0], coor[1]:coor[3]+1, :] = color
                    im[k - self.DB_Idx, coor[2], coor[1]:coor[3]+1, :] = color
                    im[k - self.DB_Idx, coor[0]:coor[2], coor[1], :] = color
                    im[k - self.DB_Idx, coor[0]:coor[2], coor[3], :] = color

        # Is it end of epoch ?
        if self.DB_Idx+self.Batch_size>=self.DB_size:
                self.Restart_First_Batch()
                EndOfEpoch = True
        else:
                self.DB_Idx = self.DB_Idx+self.Batch_size + 1
                EndOfEpoch = False

        return im,EndOfEpoch