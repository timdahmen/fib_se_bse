# Author:   Niklas Rottmayer
# Date:     25.07.2023
# Version:  1.1

# This file utilizes the Boolean Model class to generate an entire collection of geometric data from a txt-file.
# The txt-file should be of the following format:
# 1 Volume Density
# 2-4 Image size
# 5 Particle Shape (Sphere, Cube, Cylinder, Ellipsoid, Cuboid)
# 6 Parameter distribution (Uniform, Constant)
# 7-12 Particle Parameters 1-6
# 13 Orientation distribution (Constant, von Mises-Fisher, Schladitz, Uniform)
# 14-16 Orientation parameters (e.g. Euler angles or distribution parameters)
# 17 Edge treatment (Plus Sampling, Periodic)

from BooleanModel import CBooleanModel
import numpy as np
import os


if __name__ == '__main__':
    # File path to txt-file
    file_path = 'C:/Users/Rottmayer/Seafile/Promotion/Matlab/poST/Geometries/Datensatz-08-2023/Zylinder/Zylinder-Konfigurationen.txt'
    # Destination of storing generated data
    target_folder = 'C:/Users/Rottmayer/Seafile/Promotion/Matlab/poST/Geometries/Datensatz-08-2023/Zylinder'

    if target_folder and not target_folder.endswith('/'):
        target_folder += '/'

    if not os.path.exists(target_folder + 'Images'):
        print(target_folder + 'Images')
        os.makedirs(target_folder + 'Images')
        print('Creates directory Images.')
    if not os.path.exists(target_folder + 'Configs'):
        os.makedirs(target_folder + 'Configs')
        print('Created directory Configs')
    if not os.path.exists(target_folder + 'Configs_ITWM'):
        os.makedirs(target_folder + 'Configs_ITWM')
        print('Created directory Configs_ITWM')
    # File parsing
    with open(file_path,'r') as file:
        counter = 1
        for line in file:
            print(f'Generating {counter}')
            counter += 1
            # Split the line by commas
            parameters = line.strip().split(",")
            assert len(parameters) == 17, f"The number of given parameters is wrong."
            Model = CBooleanModel(volume_density = float(parameters[0]),
                                  image_size = np.array([int(num) for num in parameters[1:4]]),
                                  particle_shape = parameters[4],
                                  particle_distribution = parameters[5],
                                  particle_parameters = np.reshape([float(num) for num in parameters[6:12]],(3,2)),
                                  orientation = parameters[12],
                                  orientation_parameters = np.array([float(num) for num in parameters[13:16]]),
                                  edge_treatment = parameters[16])
            Model.generate(verbose=False)
            Model.render(verbose=False)
            # Check for the approriate name - no duplicates
            number = 1
            while os.path.exists(target_folder + 'Images/' + Model._Name + '_Num' + str(number) + '.tif') or \
                    os.path.exists(target_folder + 'Configs/' + Model._Name + '_Num' + str(number) + '.txt'):
                number += 1
            Model.save_image(target_folder+'Images/',number = number)
            Model.save_configuration(target_folder+'Configs/',number = number)
            Model.save_ITWM_configuration(target_folder+'Configs_ITWM/',number = number)
            # Temporary:
            if not os.path.exists(target_folder + 'Configs_ITWM_Euler'):
                os.makedirs(target_folder + 'Configs_ITWM_Euler')
            Model.save_ITWM_Euler_configuration(target_folder+'Configs_ITWM_Euler/',number = number)