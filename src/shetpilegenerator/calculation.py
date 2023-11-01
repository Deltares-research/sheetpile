import os
import json
import sys
import numpy as np

from shetpilegenerator.output_utils import plot_nodal_results
from shetpilegenerator.material_modifications import DICT_NORMALIZATION

sys.path.append(r"D:\Kratos_general\Kratos_build_version\KratosGeoMechanics")

import KratosMultiphysics as Kratos
import KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis as analysis_geo
import KratosMultiphysics.GeoMechanicsApplication as KratosGeo


def find_local_id_node(node, x_loc, y_loc):
    for counter, x in enumerate(x_loc):
        if node.X == x and node.Y == y_loc[counter]:
            return counter
    return None


def plot_RFs_set_in_last_stage(last_stage, directory):
    # collect model parts
    model_parts = [output_process.model_part for output_process in last_stage._list_of_output_processes if "RF" in output_process.base_file_name ]
    x_nodes = [Node.X for model_part in model_parts for Node in model_part.Nodes]
    y_nodes = [Node.Y for model_part in model_parts for Node in model_part.Nodes]
    phi = []
    youngs_modulus = []
    connectivities = []
    for model_part in model_parts:
        elements = [element for element in model_part.Elements]
      


        for element in elements:
            values = element.CalculateOnIntegrationPoints(KratosGeo.UMAT_PARAMETERS, model_part.ProcessInfo)
            nodes = element.GetNodes()
            # create connectivity
            connectivities.append([find_local_id_node(node, x_nodes, y_nodes) + 1 for node in nodes])
            phi.append(values[0][3])
            youngs_modulus.append(values[0][0])
    
    x = np.array(x_nodes)
    y = np.array(y_nodes)
    phi = np.array(phi)
    youngs_modulus = np.array(youngs_modulus)
    # normalize the values
    phi = phi / DICT_NORMALIZATION["FRICTION_ANGLE"]
    youngs_modulus = youngs_modulus / DICT_NORMALIZATION["YOUNG_MODULUS"]

    plot_nodal_results(
        x,
        y,
        phi,
        connectivities,
        save=True,
        file_name="FRICTION_ANGLE.png",
        directory=directory
    )
    plot_nodal_results(
        x,
        y,
        youngs_modulus,
        connectivities,
        save=True,
        file_name="YOUNGS_MODULUS.png",
        directory=directory
    )
    return None



def get_stages(project_path, n_stages):

    parameter_file_names = ['Project_Parameters_' + str(i + 1) + '.json' for i in range(n_stages)]
    # set stage parameters
    parameters_stages = [None] * n_stages
    os.chdir(project_path)
    for idx, parameter_file_name in enumerate(parameter_file_names):
        with open(parameter_file_name, 'r') as parameter_file:
            parameters_stages[idx] = Kratos.Parameters(parameter_file.read())

    model = Kratos.Model()
    stages = [analysis_geo.GeoMechanicsAnalysis(model, stage_parameters) for stage_parameters in parameters_stages]
    for stage in stages:
        stage.Run()
    return stages, model


def run_multistage_calculation(file_path, stages_number):
    cwd = os.getcwd()
    stages, model = get_stages(file_path, stages_number)
    os.chdir(cwd)
    plot_RFs_set_in_last_stage(stages[-1], file_path)
    model.Reset()
    return