# Project: Generation of Boolean models
# Author: Niklas Rottmayer
# Date: 26.07.2023

# This project contains the framework for generating Boolean models.
# Currently supported:
# - Particle shapes: Sphere, Ellipsoid, Cube, Cuboid, Cylinder
#   -> Distribution of particle parameters: Constant, Uniform
#   -> Intervals for particle parameters: 3x2 numpy array containing the interval bounds
#                                         (for 'Constant' only the first column is considered)
# - Orientation/Rotation of particles: Fixed, Uniform, von Mises-Fisher, Schladitz
#   -> Distribution parameters: (3,) numpy array containing:
#                              Fixed - 3 Euler angles representing a rotation to be applied to all particles equally
#                            Uniform - rotations are sampled uniformly from SO3 for each particle. Parameters are redundant
#                   von Mises-Fisher - 2 Euler angles representing a preferred direction and kappa as concentration parameter
#                          Schladitz - 2 Euler angles representing a preferred direction and beta as concentration parameter
# - Edge treatment of image: Periodic, Plus Sampling
#   -> Periodic uses wrap around and is more efficient. However, it is less 'realistic' for small samples
#   -> Plus Sampling generates particles in a slightly larger volume than rendered. It is less efficient but more realistic.
# - Poisson point process (stationary)
#   -> The stationary Poisson point process is a standard for Boolean models.

# Any addition of particle shapes, orientations, edge treatments or point processes can be added with ease in the
# respective parts of this project. For example, adding another shape requires only adaptation to the file Particles.py
# and the corresponding class CParticle. Specifically, an elif statement for the new shape has to be added to the different methods.
# Note: If you want any addition to be 'tested', you should write a reasonable test for it into the main.py file.
# Further comment: The correctness of the orientations has been validated through visual inspection of selected cuboidal
# Boolean models. While the orientation is correct, it may be the wrong direction. This means that for particles which
# have no mirror symmetry with respect to their reference direction, it still needs to be checked. However, for our case
# this is sufficiently correct.

import numpy as np
import numpy.random as rd
from DirectionalDistributions import CRotationGenerator
from Particles import CParticle
from BooleanModel import CBooleanModel, Render3DImage
from scipy.spatial.transform import Rotation  # Import of rotation expressions
import vtk
from vtkmodules.util import numpy_support
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import os

def TestDirectionDistributions():
    Test = True
    tol = 10**(-2)
    # Uniform rotation - only test if it works correctly
    uniform = CRotationGenerator.create_generator(distribution='Uniform', parameters=np.array([0.0,0.0,0.0]))
    UniformAngles = uniform.generate_rotation(10)
    if UniformAngles.shape != (10,4):
        Test = False
        print('Test has failed for the option "Uniform"')

    # Fixed rotation
    fixed = CRotationGenerator.create_generator(distribution='Fixed', parameters=np.array([1.0,2.0,3.0]))
    FixedAngles = fixed.generate_rotation(100)
    uniquerows = np.unique(FixedAngles,axis=0)
    if (uniquerows.shape != (1,4)) or not (np.array_equal(uniquerows[0,:],
                                                          Rotation.from_euler('ZXZ',[1.0,2,3]).as_quat())):
        Test = False
        print('Test has failed for the option "Fixed"')

    # von Mises-Fisher distribution
    vmf = CRotationGenerator.create_generator(distribution='von Mises-Fisher', parameters=np.array([0.0,np.pi/2,1000000]))
    vmfAngles = Rotation.from_quat(vmf.generate_rotation(100)).as_euler('ZXZ')
    for i in range(100):
        if not np.amax(abs(vmfAngles[:,1]-np.pi/2)) < tol:
            Test = False
            print('Test has failed for the option "von Mises-Fisher"')
            break

    # Schladitz distribution
    Schladitz = CRotationGenerator.create_generator(distribution='Schladitz', parameters=np.array([0, 1.6, 0.0001]))
    SchladitzAngles = Rotation.from_quat(Schladitz.generate_rotation(100)).as_euler('ZXZ')
    for i in range(100):
        if not np.amax(abs(vmfAngles[:,1]-np.pi/2)) < tol:
            Test = False
            print('Test has failed for the option "Schladitz"')
            break
    return Test

def TestEulerRotations():
    if not np.allclose(Rotation.from_euler('ZXZ', [0.0,0.0,0.0]).as_matrix()[:,2],[0,0,1.0],10**(-1)):
        return False
    elif not np.allclose(Rotation.from_euler('ZXZ', [0.0,np.pi/2,0.0]).as_matrix()[:,2],[0,-1.0,0],10**(-1)):
        return False
    elif not np.allclose(Rotation.from_euler('ZXZ', [0.0,-np.pi/2,0.0]).as_matrix()[:,2],[0,1.0,0],10**(-1)):
        return False
    elif not np.allclose(Rotation.from_euler('ZXZ', [np.pi/2,np.pi/2,0.0]).as_matrix()[:,2], [1.0,0,0], 10 ** (-1)):
        return False
    elif not np.allclose(Rotation.from_euler('ZXZ', [np.pi/2,-np.pi/2,0.0]).as_matrix()[:,2], [-1.0,0,0], 10 ** (-1)):
        return False
    elif not np.allclose(Rotation.from_euler('ZXZ', [-np.pi/2,-np.pi/2,0.0]).as_matrix()[:,2], [1.0,0,0], 10 ** (-1)):
        return False
    elif not np.allclose(Rotation.from_euler('ZXZ', [np.pi/2,0.0,np.pi/2]).as_matrix()[:,2], [0,0,1.0], 10 ** (-1)):
        return False
    else:
        return True

def TestPoissonPointProcess():
    parameters = np.array([[10, 10], [10, 10], [10, 10]],dtype=float)
    # Part 1: Standard sphere model of constant radius 10, periodic
    # Test: Check if min an max of generated center points is close to possible min and max
    test_size = np.array([128,256,512])
    Model = CBooleanModel(volume_density=0.9,image_size=test_size,particle_parameters=parameters,edge_treatment='Periodic')
    centers,n = Model.Poisson_point_process()
    if (np.max(centers[:,0]) < test_size[0]-1.5) or (np.max(centers[:,0]) > test_size[0]-1):
        return False
    elif (np.min(centers[:,0]) > 0.5) or (np.min(centers[:,0]) < 0):
        return False
    if (np.max(centers[:,1]) < test_size[1]-1.5) or (np.max(centers[:,1]) > test_size[1]-1):
        return False
    elif (np.min(centers[:,2]) > 0.5) or (np.min(centers[:,1]) < 0):
        return False
    if (np.max(centers[:,2]) < test_size[2]-1.5) or (np.max(centers[:,2]) > test_size[2]-1):
        return False
    elif (np.min(centers[:,2]) > 0.5) or (np.min(centers[:,2]) < 0):
        return False
   # Part 2: Standard sphere model of constant radius 10, plus sampling
    test_size = np.array([128, 256, 512])
    Model = CBooleanModel(volume_density=0.9, image_size=test_size,edge_treatment='Plus Sampling',particle_parameters=parameters)
    centers,n = Model.Poisson_point_process()
    if (np.max(centers[:,0]) < test_size[0]+parameters[0,1]-1.5) or (np.max(centers[:,0]) > test_size[0]+parameters[0,1]-1):
        return False
    elif (np.min(centers[:,0]) > 0.5-parameters[0,1]) or (np.min(centers[:,0]) < 0-parameters[0,1]):
        return False
    if (np.max(centers[:,1]) < test_size[1]+parameters[1,1]-1.5) or (np.max(centers[:,1]) > test_size[1]+parameters[1,1]-1):
        return False
    elif (np.min(centers[:,2]) > 0.5-parameters[1,1]) or (np.min(centers[:,1]) < 0-parameters[1,1]):
        return False
    if (np.max(centers[:,2]) < test_size[2]+parameters[2,1]-1.5) or (np.max(centers[:,2]) > test_size[2]+parameters[2,1]-1):
        return False
    elif (np.min(centers[:,2]) > 0.5-parameters[2,1]) or (np.min(centers[:,2]) < 0-parameters[2,1]):
        return False
    return True

# This test checks for correctness of the drawing functions by running a drawing call for each particle type with
# specific choice of locations and parameters.
def TestDrawingImplementation():
    # 1 - Testing sphere drawing with periodic and plus sampling as edge treatment options
    Sphere1 = CParticle.create_particle(particle_shape='Sphere',
                        particle_distribution='Constant',
                        edge_treatment='Periodic')
    img = np.zeros((31,31,31),dtype=bool)
    indices = np.argwhere(Sphere1.draw_particle(img=img,center=np.array([15,15,15]),parameters=np.array([10,10,10]),rotation=np.array([0,0,0,1.0])))
    if not np.array_equal(np.min(indices,axis=0),np.array([5,5,5])) or not np.array_equal(np.max(indices,axis=0),np.array([25,25,25])):
        return False
    indices = np.argwhere(Sphere1.draw_particle(img=img,center=np.array([5,5,5]),parameters=np.array([10,10,10]),rotation=np.array([0,0,0,1.0])))
    if not np.array_equal(np.min(indices,axis=0),np.array([0,0,0])) or not np.array_equal(np.max(indices,axis=0),np.array([30,30,30])):
        return False

    Sphere2 = CParticle.create_particle(particle_shape='Sphere',
                        particle_distribution='Uniform',
                        edge_treatment='Plus Sampling')
    img = np.zeros((31,31,31),dtype=bool)
    indices = np.argwhere(Sphere2.draw_particle(img=img,center=np.array([15,15,15]),parameters=np.array([10,10,10]),rotation=np.array([0,0,0,1.0])))
    if not np.array_equal(np.min(indices,axis=0),np.array([5,5,5])) or not np.array_equal(np.max(indices,axis=0),np.array([25,25,25])):
        return False
    img = np.zeros((31,31,31),dtype=bool)
    indices = np.argwhere(Sphere2.draw_particle(img=img,center=np.array([5,5,5]),parameters=np.array([10,10,10]),rotation=np.array([0,0,0,1.0])))
    if not np.array_equal(np.min(indices,axis=0),np.array([0,0,0])) or not np.array_equal(np.max(indices,axis=0),np.array([15,15,15])):
        return False

    # 2 - Testing ellipsoid drawing with and without rotation
    Ellipsoid = CParticle.create_particle(particle_shape='Ellipsoid',
                           particle_distribution='Constant',
                           edge_treatment='Periodic')
    img = np.zeros((31,31,31),dtype=bool)
    indices = np.argwhere(Ellipsoid.draw_particle(img=img,center=np.array([15,15,15]),parameters=np.array([5,10,12]),rotation=np.array([0,0,0,1.0])))
    if not np.array_equal(np.min(indices,axis=0),np.array([10,5,3])) or not np.array_equal(np.max(indices,axis=0),np.array([20,25,27])):
        return False
    img = np.zeros((31,31,31),dtype=bool)
    # Try with rotation to rotate z->-x,x->y,y->-z
    indices = np.argwhere(Ellipsoid.draw_particle(img=img,center=np.array([15,15,15]),parameters=np.array([5,10,12]),
                                                  rotation=Rotation.from_euler('ZXZ',[np.pi/2,np.pi/2,0.0]).as_quat()))
    if not np.array_equal(np.min(indices,axis=0),np.array([3,10,5])) or not np.array_equal(np.max(indices,axis=0),np.array([27,20,25])):
        return False

    # 3 - Test cube drawing
    Cube = CParticle.create_particle(particle_shape='Cube',
                     particle_distribution='Constant',
                     edge_treatment='Periodic')
    img = np.zeros((31,31,31),dtype=bool)
    indices = np.argwhere(Cube.draw_particle(img=img,center=np.array([15,15,15]),parameters=np.array([20,20,20]),rotation=np.array([0,0,0,1.0])))
    if not np.array_equal(np.min(indices,axis=0),np.array([5,5,5])) or not np.array_equal(np.max(indices,axis=0),np.array([25,25,25])):
        return False

    # 4 - Test cuboid drawing with and without rotation
    Cuboid = CParticle.create_particle(particle_shape='Cuboid',
                       particle_distribution='Constant',
                       edge_treatment='Periodic')
    img = np.zeros((31,31,31),dtype=bool)
    indices = np.argwhere(Cuboid.draw_particle(img=img,center=np.array([15,15,15]),parameters=np.array([10,20,25]),rotation=np.array([0,0,0,1.0])))
    if not np.array_equal(np.min(indices,axis=0),np.array([10,5,3])) or not np.array_equal(np.max(indices,axis=0),np.array([20,25,27])):
        return False
    img = np.zeros((31,31,31),dtype=bool)
    # Try with rotation to rotate z->-x,x->y,y->-z
    indices = np.argwhere(Cuboid.draw_particle(img=img,center=np.array([15,15,15]),parameters=np.array([10,20,25]),
                                               rotation=Rotation.from_euler('ZXZ',[np.pi/2,np.pi/2,0.0]).as_quat()))
    if not np.array_equal(np.min(indices,axis=0),np.array([3,10,5])) or not np.array_equal(np.max(indices,axis=0),np.array([27,20,25])):
        return False

    # 5 - Test cylinder drawing with and without rotation
    Cylinder = CParticle.create_particle(particle_shape='Cylinder',
                         particle_distribution='Constant',
                         edge_treatment='Periodic')
    img = np.zeros((31,31,31),dtype=bool)
    indices = np.argwhere(Cylinder.draw_particle(img=img,center=np.array([15,15,15]),parameters=np.array([10,23,0]),rotation=np.array([0,0,0,1.0])))
    if not np.array_equal(np.min(indices,axis=0),np.array([5,5,4])) or not np.array_equal(np.max(indices,axis=0),np.array([25,25,26])):
        return False
    img = np.zeros((31,31,31),dtype=bool)
    indices = np.argwhere(Cylinder.draw_particle(img=img,center=np.array([15,15,15]),parameters=np.array([10,23,0]),
                                                 rotation=Rotation.from_euler('ZXZ',[np.pi/2,np.pi/2,0.0]).as_quat()))
    if not np.array_equal(np.min(indices,axis=0),np.array([4,5,5])) or not np.array_equal(np.max(indices,axis=0),np.array([26,25,25])):
        return False
    return True

# This test generates multiple instances of models and simply checks if they run through smoothly
def TestBooleanModel():
    Particle_Distributions = ['Constant','Uniform']
    Particle_Orientations = ['Fixed','Uniform','von Mises-Fisher','Schladitz']
    Orientation_Parameters = np.array([[0.2,0.1,0],
                                       [0,0,0],
                                       [np.pi/2,np.pi/2,20],
                                       [np.pi/2,np.pi/2,20]])
    Shapes = ['Sphere','Ellipsoid','Cube','Cuboid','Cylinder']
    Edge_Treatment = ['Periodic','Plus Sampling']
    Image_Size = np.array([64,64,64])
    Parameters = np.array([[8,11],[9,12],[10,13]])

    # 1 - Run with default
    M1 = CBooleanModel()
    M1.generate(verbose=0)

    # 2 - Run all available combinations of options on small test image
    for i in range(len(Particle_Distributions)):
        for j in range(len(Particle_Orientations)):
            for k in range(len(Shapes)):
                for l in range(len(Edge_Treatment)):
                    M = CBooleanModel(image_size=Image_Size,
                                      volume_density=0.3,
                                      particle_shape=Shapes[k],
                                      particle_parameters=Parameters,
                                      orientation=Particle_Orientations[j],
                                      orientation_parameters=Orientation_Parameters[j,:],
                                      edge_treatment=Edge_Treatment[l],
                                      particle_distribution=Particle_Distributions[i])
                    M.generate(verbose=0)
                    M.render(verbose=0)
                    # Perform some standard tests
                    # a) Image is correctly set in size
                    if M.Image is None:
                        return False
                    elif not np.array_equal(M.Image.shape,Image_Size):
                        return False
                    # b) Number of particles coincides with center, parameter and rotation arrays
                    if M._Sampled_Particle_Number < 1:
                        return False
                    elif not M._Sampled_Parameters.shape[0] == M._Sampled_Particle_Number \
                        or not M._Sampled_Centers.shape[0] == M._Sampled_Particle_Number \
                        or not M._Sampled_Rotations.shape[0] == M._Sampled_Particle_Number:
                        return False
                    # c) Check volume density - should be in a +-0.15 Interval (due to small image size)
                    if not abs(np.count_nonzero(M.Image)/np.prod(Image_Size) - M.Volume_Density) < 0.15:
                        return False
    return True

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Test #1: Correctness of Euler rotations:
    print(str(TestEulerRotations()) + ' - Correctness of Euler rotations')
    # Test #2: Correctness of directional distributions:
    print(str(TestDirectionDistributions()) + ' - Correctness of directional distributions')
    # Test #3: Correctness of Poisson point process
    print(str(TestPoissonPointProcess()) + ' - Correctness of Poisson point process')
    # Test #4: Correctness of drawing implementations
    print(str(TestDrawingImplementation()) + ' - Correctness of particle drawers')
    # Test #5: Test for overall functionality without errors
    counter = 0
    while counter < 5:
        result = TestBooleanModel()
        if result:
            break
        counter += 1
    print(str(result) + ' - Correctness of overall functionality without errors')

    #M = CBooleanModel(image_size=np.array([512,512,512]),volume_density=0.01,particle_shape='Cuboid',
    #                  particle_parameters=np.array([[30,30],[50,50],[150,150]]),
    #                  orientation_parameters=np.array([np.pi/2,np.pi/2,0]))
    #M.generate()
    #M.render()

    # Test functionality on Cube example
    np.random.seed(123)
    M = CBooleanModel(image_size=np.array([128,128,128]),volume_density=0.1,particle_shape='Cube',
                      particle_parameters=np.array([[30,30],[50,50],[150,150]]),
                      orientation_parameters=np.array([0,0,0]),
                      orientation='Uniform',edge_treatment='Plus Sampling')

    M.generate()
    #M._Sampled_Centers[0,:] = M.Image_Size.reshape((1, -1)) // 2
    M.render()

    #cwd = os.getcwd()
    #M.save_image(cwd)
    #M.save_configuration(cwd)
    #M.save_image(cwd)
    #M.save_configuration(cwd)
    #M.save_ITWM_configuration(cwd)
    Render3DImage(M.Image)

    # Test of euler angle of any vector and rotation
    #Rot = Rotation.from_euler('ZXZ',(0,np.pi/2,0))
    #vec = np.array([0,0,1])
    #print(Rot.as_matrix())
    #print(Rot.apply(vec))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
