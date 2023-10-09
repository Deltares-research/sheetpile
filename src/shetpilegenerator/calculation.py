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


def plot_RFs_set_in_last_stage(last_stage, directory):
    # collect data
    model_part = last_stage._list_of_output_processes[0].model_part
    elements = model_part.Elements

    x = []
    y = []
    x_nodes = [Node.X for Node in model_part.Nodes]
    y_nodes = [Node.Y for Node in model_part.Nodes]
    phi = []
    youngs_modulus = []
    connectivities = []
    for element in elements:
        values = element.CalculateOnIntegrationPoints(KratosGeo.UMAT_PARAMETERS, model_part.ProcessInfo)
        points = element.GetIntegrationPoints()
        nodes = element.GetNodes()
        # create connectivity
        connectivities.append([node.Id for node in nodes])
        for counter, umat_vector in enumerate(values):
            x.append(points[counter][0])
            y.append(points[counter][1])
            phi.append(umat_vector[3])
            youngs_modulus.append(umat_vector[0])
    # interpolate from the integration points to the nodes
    x = np.array(x)
    y = np.array(y)
    phi = np.array(phi)
    youngs_modulus = np.array(youngs_modulus)
    import scipy.interpolate as interpolate
    phi = interpolate.griddata((x, y), phi, (x_nodes, y_nodes), method='nearest')
    youngs_modulus = interpolate.griddata((x, y), youngs_modulus, (x_nodes, y_nodes), method='nearest')
    # normalize the values
    phi = phi / DICT_NORMALIZATION["FRICTION_ANGLE"]
    youngs_modulus = youngs_modulus / DICT_NORMALIZATION["YOUNG_MODULUS"]

    plot_nodal_results(
        x_nodes,
        y_nodes,
        phi,
        np.array(connectivities),
        save=True,
        file_name="FRICTION_ANGLE.png",
        directory=directory
    )
    plot_nodal_results(
        x_nodes,
        y_nodes,
        youngs_modulus,
        np.array(connectivities),
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
    os.chdir('..\..')
    plot_RFs_set_in_last_stage(stages[-1], file_path)
    model.Reset()
    return