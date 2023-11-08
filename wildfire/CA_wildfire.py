# -*- coding: utf-8 -*-

"""

import sys
import math
import random
import copy
import random
#from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation

from PIL import Image
import matplotlib as mpl

import numpy as np
from google.colab import drive
drive.mount('/content/drive')

###################################################################################
#test with real map
forest = Image.open('CA/land_data/canopy_Chimney_2016.tif')

ignition = np.loadtxt('CA/land_data/Chimney_2016_ignition_forest2.txt')

altitude = Image.open('CA/land_data/slope_Chimney_2016.tif')

density = Image.open('CA/land_data/density_Chimney_2016.tif')

np.array(forest).shape

plt.imshow(np.array(forest))



def random_ignition(dim_x,dim_y):
  field = np.ones((dim_x,dim_y))*2
  x = random.randint(30,dim_x-30)
  y = random.randint(30,dim_y-30)
  for i in range(x,x+4):
    for j in range(y,y+4):
      field[i,j] = 3
  return field

def centre_ignition(dim_x,dim_y):
  field = np.ones((dim_x,dim_y))*2
  x = round(dim_x/2)
  y = round(dim_y/2)
  for i in range(x,x+3):
    for j in range(y,y+3):
      field[i,j] = 3
  return field

forest = np.array(forest)

altitude = np.array(altitude)/np.max(altitude)

density = np.array(density)

density = np.round(density/np.max(density))

forest[forest<-999.] = 0.

forest = forest/np.max(forest)


from PIL import Image
from skimage.transform import resize

forest = resize(forest, (128, 128))
altitude = resize(altitude, (128, 128))
density = resize(density, (128, 128))


#forest = resize(forest, (128*4, 128*4))
#altitude = resize(altitude, (128*4, 128*4))
#density = resize(density, (128*4, 128*4))

density = np.round(density/np.max(density))

n_row = forest.shape[0]
n_col = forest.shape[1]

number_MC = 20
#################################################################
generation = 501
n_row = forest.shape[0]
n_col = forest.shape[1]

def colormap(i,array):
  np_array = np.array(array)
  plt.imshow(np_array, interpolation="none", cmap=cm.plasma)
  plt.title(i)
  plt.show()

def init_vegetation():
    veg_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            veg_matrix[i][j] = 1
    return veg_matrix

def init_density():
    den_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            den_matrix[i][j] = 1.0
    return den_matrix

def init_altitude():
    alt_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            alt_matrix[i][j] = 1
    return alt_matrix

def init_forest():
    forest = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            forest[i][j] = 2
    # ignite_col = int(n_col//2)
    # ignite_row = int(n_row//2)
    ignite_col = int(n_col//2)
    ignite_row = int(100)
    for row in range(ignite_row-1, ignite_row+1):
        for col in range(ignite_col-1,ignite_col+1):
            forest[row][col] = 3
    # forest[ignite_row-2:ignite_row+2][ignite_col-2:ignite_col+2] = 3
    return forest

def print_forest(forest):
    for i in range(n_row):
        for j in range(n_col):
            sys.stdout.write(str(forest[i][j]))
        sys.stdout.write("\n")

def tg(x):
    return math.degrees(math.atan(x))

def get_slope(altitude_matrix):
    slope_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for row in range(n_row):
        for col in range(n_col):
            sub_slope_matrix = [[0,0,0],[0,0,0],[0,0,0]]
            if row == 0 or row == n_row-1 or col == 0 or col == n_col-1:  # margin is flat
                slope_matrix[row][col] = sub_slope_matrix
                continue
            current_altitude = altitude_matrix[row][col]
            sub_slope_matrix[0][0] = tg((current_altitude - altitude_matrix[row-1][col-1])/1.414)
            sub_slope_matrix[0][1] = tg(current_altitude - altitude_matrix[row-1][col])
            sub_slope_matrix[0][2] = tg((current_altitude - altitude_matrix[row-1][col+1])/1.414)
            sub_slope_matrix[1][0] = tg(current_altitude - altitude_matrix[row][col-1])
            sub_slope_matrix[1][1] = 0
            sub_slope_matrix[1][2] = tg(current_altitude - altitude_matrix[row][col+1])
            sub_slope_matrix[2][0] = tg((current_altitude - altitude_matrix[row+1][col-1])/1.414)
            sub_slope_matrix[2][1] = tg(current_altitude - altitude_matrix[row+1][col])
            sub_slope_matrix[2][2] = tg((current_altitude - altitude_matrix[row+1][col+1])/1.414)
            slope_matrix[row][col] = sub_slope_matrix
    return slope_matrix

generation = 60


for index in range(0,10):
  if index%50 == 0:
    print(index)
  #ignition = random_ignition(np.array(forest).shape[0],np.array(forest).shape[1])#
  ignition = centre_ignition(np.array(forest).shape[0],np.array(forest).shape[1])#
  #np.save('drive/MyDrive/CA/VAE/ignition/ignition_Brattain_'+str(index)+'.npy',ignition)

  V = 5. # need to find the true wind data
  #p_h = 0.58
  a = 0.078
  c_1 = 0.045
  c_2 = 0.131

  p_h=random.uniform(0.20, 0.35)*1.
  #a=random.uniform(0., 0.14)*1.
  #c_1=random.uniform(0., 0.12)*1.
  #c_2=random.uniform(0., 0.40)

  ##############################################################################
  def calc_pw(theta,c_1,c_2,V):
    t = math.radians(theta)
    ft = math.exp(V*c_2*(math.cos(t)-1))
    return math.exp(c_1*V)*ft
  def get_wind():
      wind_matrix = [[0 for col in [0,1,2]] for row in [0,1,2]]

      #thetas = [[0,180,180], #need to define the exact angle
      #          [180,0,180],
      #          [180,180,0]]

      thetas = [[180,180,180], #need to define the exact angle
                [180,0,180],
                [180,180,180]]

      for row in [0,1,2]:
          for col in [0,1,2]:
              wind_matrix[row][col] = calc_pw(thetas[row][col],c_1,c_2,V)
      wind_matrix[1][1] = 0
      return wind_matrix
  def burn_or_not_burn(abs_row,abs_col,neighbour_matrix,p_h,a):
      p_veg = vegetation_matrix[abs_row][abs_col]
      p_den = {0:-0.4,1:0,2:0.3}[density_matrix[abs_row][abs_col]]
      for row in [0,1,2]:
          for col in [0,1,2]:
              if neighbour_matrix[row][col] == 3: # we only care there is a neighbour that is burning
                  # print(row,col)
                  slope = slope_matrix[abs_row][abs_col][row][col]
                  p_slope = math.exp(a * slope)
                  p_wind = wind_matrix[row][col]
                  p_burn = p_h * (0.5 + p_veg*10.) * (1 + p_den) * p_wind * p_slope
                  if p_burn > random.random():
                      return 3  #start burning
      return 2 # not burning
  def update_forest(old_forest):
      result_forest = [[1 for i in range(n_col)] for j in range(n_row)]
      for row in range(1, n_row-1):
          for col in range(1, n_col-1):

              if old_forest[row][col] == 1 or old_forest[row][col] == 4:
                  result_forest[row][col] = old_forest[row][col]  # no fuel or burnt down
              if old_forest[row][col] == 3:
                  if random.random() < 0.4:
                      result_forest[row][col] = 3  # TODO need to change back here
                  else:
                      result_forest[row][col] = 4
              if old_forest[row][col] == 2:
                  neighbours = [[row_vec[col_vec] for col_vec in range(col-1, col+2)]
                                for row_vec in old_forest[row-1:row+2]]
                  # print(neighbours)
                  result_forest[row][col] = burn_or_not_burn(row, col, neighbours,p_h,a)
      return result_forest
  #############################################################################

  fields_1_sim = np.zeros((1,100))

  vegetation_matrix = forest

  density_matrix = density.tolist()

  altitude_matrix = altitude.tolist()

  wind_matrix = get_wind()

  new_forest = ignition.tolist()


  slope_matrix = get_slope(altitude_matrix)

  ims = []

  burned_pixel = []
  ###########################################################
      # custormize colorbar

  cmap = mpl.colors.ListedColormap(['orange','yellow', 'green', 'black'])
  cmap.set_over('0.25')
  cmap.set_under('0.75')
  bounds = [1.0, 2.02, 2.27, 3.5, 5.1]
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

  ############################################################


  for i in range(generation):

      print(index, i)
      new_forest = copy.deepcopy(update_forest(new_forest))
      forest_array = np.array(new_forest)
      burned_pixel.append(np.sum(forest_array>=3))
      if i>0 and i%5 == 0:

        #np.save('drive/MyDrive/CA/VAE/ignition_Brattain_'+str(index)+'_'+str(i)+'_'+'.npy',forest_array)
        #np.save('drive/MyDrive/CA/VAE/ignition_Chimney_'+str(index)+'_'+str(i)+'_'+'.npy',forest_array)
        #np.save('drive/MyDrive/CA/Field/JChimney_'+str(index)+'_'+str(i)+'_'+'.npy',forest_array)


        plt.imshow(forest + forest_array,  cmap = cmap,norm = norm, interpolation="none" )
        plt.axis('off')
        #plt.savefig('CA_data/fire_'+ str(i) +'.png', format='png',bbox_inches='tight')
        plt.show()
        plt.close()
        print('######################################################################',p_h)
        print('burning',np.sum(forest_array==4))
      plt.show()
      plt.close()