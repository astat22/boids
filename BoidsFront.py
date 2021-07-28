package Boids as bd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from mpl_toolkits.mplot3d import Axes3D
import imageio
import copy
import pandas as pd
import os


        # print([self._x[id],self._y[id],self._z[id],self._vx[id],self._vy[id],self._vz[id]])

def animation(steps = 1000):
    test = ExtendedBoids(board_size = (500,500,500),  number_of_boids = [30,20,8],noise = [0.00001],
                         neighbourhood_distance = [62,70,90], min_distance = [5], separation_weight = [0.1,0.1,0.08],
                        alignment_weight = [0.13,0.19,0.2], cohension_weight = [0.17], flat_angle = [165,160,180],
                         z_angle = [160], maximum_speed = [7.5,7.0,9.0],
                         obstacles = [[50,50,50,30], [100,100,200,40], [200,200,300,60]])
    a=Cones()
    a.mark_bird()
    filenames = []
    for i in range(steps):
        test.board_step()
        a.refresh()
        name = 'clip%d.png'%i
        filenames.append(name)
#    with imageio.get_writer('clip.gif', mode='I') as writer:
#        for filename in filenames:#
#
#
 #           image = imageio.imread(filename)
 #           writer.append_data(image)

  #      writer.close()
    length = len(a._marked)
    nei = []
    for i in a._marked:
        nei.append(i[3])
    return nei