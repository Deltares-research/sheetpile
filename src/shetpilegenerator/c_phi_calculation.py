import os
import json

from shetpilegenerator.calculation import run_multistage_calculation
from shetpilegenerator.output_utils import post_process


def reduce_phi(RF, phi_init):
    import math
    return math.degrees(math.atan(math.tan(math.radians(phi_init)) / RF))


def modify_material_parameters_c_phi_reduction(project_path, stage_number,  RF, c_init, phi_init):
    # modify the parameters
    material_parameters_file = os.path.join(project_path, f'MaterialParameters_{stage_number}.json')
    # open json file
    with open(material_parameters_file, 'r') as parameter_file:
        material_parameters = json.load(parameter_file)
    # modify the parameters
    for counter, key in enumerate(material_parameters['properties']):
        material_parameters['properties'][counter]['Material']['Variables']['UMAT_PARAMETERS'][2] = c_init / RF
        material_parameters['properties'][counter]['Material']['Variables']['UMAT_PARAMETERS'][3] = reduce_phi(RF,
                                                                                                               phi_init)
    # write the modified parameters
    with open(material_parameters_file, 'w') as parameter_file:
        json.dump(material_parameters, parameter_file, indent=4)


def get_initial_c_phi_parameters(project_path, stage_number):
    # modify the parameters
    material_parameters_file = os.path.join(project_path, f'MaterialParameters_{stage_number}.json')
    # open json file
    with open(material_parameters_file, 'r') as parameter_file:
        material_parameters = json.load(parameter_file)
    # modify the parameters
    for counter, key in enumerate(material_parameters['properties']):
        c_init = material_parameters['properties'][counter]['Material']['Variables']['UMAT_PARAMETERS'][2]
        phi_init = material_parameters['properties'][counter]['Material']['Variables']['UMAT_PARAMETERS'][3]
    return c_init, phi_init


def run_c_phi_reduction(project_path, stage_number, RF_min, RF_max, gmsh_to_kratos, step=0.05, ):
    RF = RF_min
    c_init, phi_init = get_initial_c_phi_parameters(project_path, stage_number)
    while RF < RF_max:
        try:
            print("RF = ", RF)
            modify_material_parameters_c_phi_reduction(project_path, stage_number, RF, c_init, phi_init)
            # run the simulation
            run_multistage_calculation("kratos_write_test", 2)
            RF += step
        except Exception as e:
            if str(e) == "The maximum number of cycles is reached without convergence!":
                print("C-phi reduction finished! At RF = ", RF)
                # rerun the simulation with the last RF
                os.chdir("..")
                modify_material_parameters_c_phi_reduction(project_path, stage_number, RF - step, c_init, phi_init)
                run_multistage_calculation("kratos_write_test", 2)
                post_process(2, 2.0, gmsh_to_kratos)
                break
            else:
                raise e