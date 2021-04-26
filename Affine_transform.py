import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

picture = Image.open('image.jpg').convert('L')
picture = np.array(picture)
picture_vec = np.transpose(np.argwhere(picture))

picture2 = Image.open('image_warping.jpg').convert('L')
picture2 = np.array(picture2)
picture2_vec = np.transpose(np.argwhere(picture2))

def scaling(x,y):
    scal = np.array([[x,0],[0,y]])
    scal_picture_vec = np.int_(scal.dot(picture_vec))
    scal_zeros = np.zeros((np.max(scal_picture_vec[0])+1,np.max(scal_picture_vec[1])+1))
    scal_zeros[scal_picture_vec[0],scal_picture_vec[1]] = picture[picture_vec[0],picture_vec[1]]
    plt.imshow(scal_zeros,'gray')
    plt.show()


def rotation(rad):
    rot = np.array([[np.cos(np.deg2rad(rad)), -np.sin(np.deg2rad(rad))],
                        [np.sin(np.deg2rad(rad)), np.cos(np.deg2rad(rad))]])
    rot_picture_vec = rot.dot(picture_vec)
    rot_picture_vec[0] = rot_picture_vec[0] - min(rot_picture_vec[0])
    rot_picture_vec = rot_picture_vec.astype(int)
    rot_zeros = np.zeros((np.max(rot_picture_vec[0])+1,np.max(rot_picture_vec[1])+1))
    rot_zeros[rot_picture_vec[0],rot_picture_vec[1]]=picture[picture_vec[0],picture_vec[1]]
    plt.imshow(rot_zeros,'gray')
    plt.show()


def trans(x,y):
    tran = np.array([[x],[y]])
    tran_picture_vec = tran + picture_vec
    tran_zeros = np.zeros((np.max(tran_picture_vec[0])+1, np.max(tran_picture_vec[1])+1))
    tran_zeros[tran_picture_vec[0],tran_picture_vec[1]] = picture[picture_vec[0],picture_vec[1]]
    plt.imshow(tran_zeros,'gray')
    plt.show()

def shear(x):
    she = np.array([[1,x],[0,1]])
    she_picture_vec = np.int_(she.dot(picture_vec))
    she_zeros = np.zeros((np.max(she_picture_vec[0])+1,np.max(she_picture_vec[1])+1))
    she_zeros[she_picture_vec[0],she_picture_vec[1]] = picture[picture_vec[0],picture_vec[1]]
    plt.imshow(she_zeros,'gray')
    plt.show()

def warping(a1,a2,a3,a4,b1,b2,b3,b4):
    war = np.array([[a1,a2,a3,a4],
                        [b1,b2,b3,b4]])
    x = picture2_vec[0]
    y = picture2_vec[1]
    x_y = x*y
    n1 = np.ones((1,np.size(x)))
    xy = np.vstack((x_y,x,y,n1))
    war_picture_vec = np.int_(war.dot(xy))
    print(war_picture_vec)
    war_picture_vec[0] = war_picture_vec[0] - min(war_picture_vec[0])
    war_picture_vec[1] = war_picture_vec[1] - min(war_picture_vec[1])
    war_zeros = np.zeros((np.max(war_picture_vec[0])+1,
                                  np.max(war_picture_vec[1])+1))
    war_zeros[war_picture_vec[0],war_picture_vec[1]] = picture2[picture2_vec[0],picture2_vec[1]]
    print(war_picture_vec.shape)
    print(picture2_vec.shape)
    print(war_zeros.shape)
    plt.imshow(war_zeros,'gray')
    plt.show()


##scaling(1,2)
##rotation(30)
#trans(100,100)
##shear(1)
##warping(-0.004, 1.2, 0.5, -30,0.002, -0.3, 0.8, 30)




#최근접 이웃 보간법

rad = 30
rot = np.array([[np.cos(np.deg2rad(rad)), -np.sin(np.deg2rad(rad))],
                    [np.sin(np.deg2rad(rad)), np.cos(np.deg2rad(rad))]])
                    
rot_picture_vec = rot.dot(picture_vec)
rot_picture_vec[0] = rot_picture_vec[0] - min(rot_picture_vec[0])
rot_picture_vec = rot_picture_vec.astype(int)
rot_zeros = np.zeros((np.max(rot_picture_vec[0])+1,np.max(rot_picture_vec[1])+1))
rot_zeros[rot_picture_vec[0],rot_picture_vec[1]]=picture[picture_vec[0],picture_vec[1]]
zeros_vec = np.where(rot_zeros == 0)

a = np.zeros((1,rot_zeros.shape[1]))
rot_zeros = np.vstack((rot_zeros,a))
b = np.zeros((rot_zeros.shape[0],1))
rot_zeros = np.hstack((rot_zeros,b))

for i in range(0,zeros_vec[0].size):
        a = rot_zeros[zeros_vec[0][i],zeros_vec[1][i]-1]
        b = rot_zeros[zeros_vec[0][i]-1,zeros_vec[1][i]-1]
        c = rot_zeros[zeros_vec[0][i]-1,zeros_vec[1][i]]
        d = rot_zeros[zeros_vec[0][i]-1,zeros_vec[1][i]+1]
        e = rot_zeros[zeros_vec[0][i],zeros_vec[1][i]+1]
        f = rot_zeros[zeros_vec[0][i]+1,zeros_vec[1][i]+1]
        g = rot_zeros[zeros_vec[0][i]+1,zeros_vec[1][i]]
        h = rot_zeros[zeros_vec[0][i]+1,zeros_vec[1][i]-1]
        p = np.array([a,b,c,d,e,f,g,h])
        p1 = np.min(p)
        rot_zeros[zeros_vec[0][i],zeros_vec[1][i]] = p1
plt.imshow(rot_zeros,'gray')
plt.show()





#다항식 보간법
"""
rad = 30
rot = np.array([[np.cos(np.deg2rad(rad)), -np.sin(np.deg2rad(rad))],
                    [np.sin(np.deg2rad(rad)), np.cos(np.deg2rad(rad))]])
rot_picture_vec = rot.dot(picture_vec)
rot_picture_vec[0] = rot_picture_vec[0] - min(rot_picture_vec[0])
rot_picture_vec = rot_picture_vec.astype(int)
rot_zeros = np.zeros((np.max(rot_picture_vec[0])+1,np.max(rot_picture_vec[1])+1))
rot_zeros[rot_picture_vec[0],rot_picture_vec[1]]=picture[picture_vec[0],picture_vec[1]]


plt.imshow(rot_zeros,'gray')
plt.show()
"""















