from matplotlib import pyplot as plt
import numpy as np

def showDigit(data,row,column):
    img = np.zeros((row,column))
    aux = 0
    for i in range(0,row):
        for j in range(0,column):
            img[i,j] = data[aux]
            aux += 1
    plt.imshow(img,cmap='gray')#clim=(0.0, 0.7))
    plt.show()

# data = [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]
# row = 8
# column = 8
# showDigit(data,row,column)
