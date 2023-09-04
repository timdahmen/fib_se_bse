import numpy.random as rd
from scipy.spatial.transform import Rotation
import numpy as np
from BooleanModel import Render3DImage

pi = np.pi
k = 0
Cases = [[0,0,0],
         [pi/2,0,0],
         [0,pi/2,0],
         [0,0,pi/2],
         [pi/2,pi/2,0],
         [pi/2,0,pi/2],
         [0,pi/2,pi/2],
         [pi/2,pi/2,pi/2]]

Angles = Cases[k]
RotMat = Rotation.from_euler('ZXZ',Angles).as_matrix()

img = np.zeros([121,121,121],dtype=bool)

x,y,z = np.ogrid[-60:60+1,-60:60+1,-60:60+1]

xout = RotMat[0,0]*x + RotMat[1,0]*y + RotMat[2,0]*z
yout = RotMat[0,1]*x + RotMat[1,1]*y + RotMat[2,1]*z
zout = RotMat[0,2]*x + RotMat[1,2]*y + RotMat[2,2]*z

# Draw Sphere
img = np.logical_or(img,(xout**2 + yout**2 + zout**2)/2.5**2 <= 1)
# Draw Cuboid
img = np.logical_or(img,np.maximum(np.maximum(2*abs(xout)/5.,
                                              2*abs(yout)/10.),
                                              2*abs(zout-30)/50.) <= 1)
# Draw Ellipsoid
img = np.logical_or(img,(((xout-30)/25.)**2 + (yout/5.)**2 + (zout/2.5)**2) <= 1)
# Draw Cylinder
img = np.logical_or(img,np.logical_and((xout**2 + zout**2)/5**2 <= 1,
                                        2*abs(yout-30)/50 <= 1))

Render3DImage(img)