import sys
sys.path.append(r"/app")
# for local testing
sys.path.append(r"D:/Kratos_general/Kratos_build_version/Kratos")

import KratosMultiphysics as Kratos
import KratosMultiphysics.GeoMechanicsApplication as KratosGeo
import KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis as analysis_geo

from shetpilegenerator.calculation import run_multistage_calculation
from shetpilegenerator.output_utils import post_process
import pickle
import os 

if __name__ == '__main__':
    directory = f'output'
    # get all folders in output directory
    folders_to_run = os.listdir(directory)
    for folder in folders_to_run:
        sub_directory = os.path.join(directory, folder)
        run_multistage_calculation(sub_directory, 3)
        # open pickle file
        model_file = os.path.join(sub_directory, "model.p")
        gmsh_to_kratos = pickle.load(open(model_file, "rb"))
        post_process(3, 3.0, gmsh_to_kratos, save=True, directory=sub_directory, file_name="stage_3.png")

