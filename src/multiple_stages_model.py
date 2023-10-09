from shetpilegenerator.gmsh_to_kratos import GmshToKratos
from shetpilegenerator.material_library import *
from shetpilegenerator.rf_generator import generate_jsons_for_material
from shetpilegenerator.material_modifications import write_materials_dict, modify_MC_parameters
from shetpilegenerator.water_pressures_modifications import define_water_boundaries, define_water_line_based_on_outside_head
from shetpilegenerator.project_parameters_modifications import add_and_save_water_and_rfs, modify_project_parameters
from shetpilegenerator.output_utils import plot_geometry, post_process
from shetpilegenerator.calculation import run_multistage_calculation


import sqlite3
import matplotlib.pyplot as plt

from stem.load import LineLoad
from stem.IO.kratos_loads_io import KratosLoadsIO
from gmsh_utils.gmsh_IO import GmshIO


NUMBER_OF_SAMPLES = 1000  # number of samples to generate


def define_geometry_from_gmsh(input_points, name_labels):
    dimension = 2
    gmsh_io = GmshIO()
    gmsh_io.generate_geometry(input_points,
                              [0, 0, 0],
                              dimension,
                              "mesh_dike_2d",
                              name_labels,
                              5)

    physical_groups = gmsh_io.generate_extract_mesh(dimension, "mesh_dike_2d", ".", False, True)
    geo_data = gmsh_io.geo_data
    mesh_data = gmsh_io.mesh_data
    mesh_data['physical_groups'] = physical_groups
    total_dict = {'geo_data': geo_data, 'mesh_data': mesh_data}
    return total_dict





def get_field_process_dict(field_process):
    return {
        "python_module": "set_parameter_field_process",
        "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
        "process_name": "SetParameterFieldProcess",
        "Parameters": {
            "model_part_name": f"PorousDomain.{field_process['name']}",
            "variable_name": field_process['variable'],
            "func_type": field_process['function_type'],
            "function": field_process['function'],
            "dataset": "dummy",
            "dataset_file_name": field_process['dataset_file_name'],
        }
    }


def set_stage(stage_number, start_time, end_time, project_path, constrains_on_surfaces, soils, head_level, soil_model,
              data, loads):
    write_materials_dict(project_path + f"/MaterialParameters_{stage_number}.json",
                         soils,
                         soil_model=soil_model)
    # define water conditions stage 1
    TOP_1_WL, TOP_2_WL, TOP_3_WL, WL2, WL3 = define_water_line_based_on_outside_head(head_level)
    dictionary_water_boundaries_stage3 = define_water_boundaries(TOP_1_WL, TOP_2_WL, TOP_3_WL, WL2, WL3)
    load_names = []
    # add the loads
    if loads is not None:
        for load in loads:
            io_load = KratosLoadsIO("PorousDomain")
            dictionary_water_boundaries_stage3.append(io_load.create_load_dict(part_name=load['name'],
                                                                               parameters=load['load'])
                                                      )
            load_names.append(load['name'])
    # add the field processes
    sub_model_names_field_process = []
    for field_process in constrains_on_surfaces['field_processes']:
        sub_model_names_field_process.append(field_process['name'])
        dictionary_water_boundaries_stage3.append(get_field_process_dict(field_process))
    # define stage 1 parameters
    add_and_save_water_and_rfs("Project_Parameters_template.json",
                               project_path + f"/Project_Parameters_{stage_number}.json",
                               {"loads_process_list": dictionary_water_boundaries_stage3})
    # create the object
    gmsh_to_kratos = GmshToKratos(data)
    gmsh_to_kratos.read_gmsh_to_kratos(property_list=list(range(1, len(constrains_on_surfaces['surfaces']) + 1)),
                                       mpda_file=project_path + f"/test_multistage_{stage_number}.mdpa",
                                       constrains_on_surfaces=constrains_on_surfaces,
                                       top_load=bool(loads is not None), )
    lists_of_all_parts = {
        "problem_domain_sub_model_part_list": constrains_on_surfaces['surfaces'],
        "body_domain_sub_model_part_list": constrains_on_surfaces['surfaces'],
        "processes_sub_model_part_list": ["bottom_disp", "side_disp", "gravity"] +
                                         constrains_on_surfaces['names'] +
                                         load_names +
                                         sub_model_names_field_process
    }

    modify_project_parameters(project_path + f"/Project_Parameters_{stage_number}.json",
                              f"test_multistage_{stage_number}",
                              f"MaterialParameters_{stage_number}.json",
                              lists_of_all_parts,
                              time_start=start_time,
                              time_end=end_time)
    return gmsh_to_kratos


def create_model(directory, input_values):
    constrains_on_surfaces = {
        "names": ["top_water_boundary_1", "top_water_boundary_2", "top_water_boundary_3",
                  "middle_water_boundary", "bottom_water_boundary"],
        "surfaces": ["TOP_1", "TOP_2", "TOP_3", "MIDDLE", "BOTTOM"],
        "material_per_surface": ["TOP_1", "TOP_2", "TOP_3", "MIDDLE", "BOTTOM"],
        "field_processes": [
            {"name": "TOP_1_RF", "variable": "YOUNG_MODULUS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "TOP_1_le_RF.json"},
            {"name": "TOP_2_RF", "variable": "YOUNG_MODULUS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "TOP_2_le_RF.json"},
            {"name": "TOP_3_RF", "variable": "YOUNG_MODULUS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "TOP_3_le_RF.json"},
            {"name": "MIDDLE_RF", "variable": "YOUNG_MODULUS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "MIDDLE_le_RF.json"},
            {"name": "BOTTOM_RF", "variable": "YOUNG_MODULUS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "BOTTOM_le_RF.json"}],
    }

    # set default loads
    # define the line load
    line_load = LineLoad(active=[True, True, False], value=[0.0, 0.0, 0.0])
    line_load_input = {"name": "dike_load", "load": line_load}

    layers = define_layers()
    plot_geometry(layers, save=True, file_name="geometry.png", directory=directory)
    data = define_geometry_from_gmsh(layers, constrains_on_surfaces['surfaces'])
    # set all random field json files
    generate_jsons_for_material(input_values["INDEX"], directory, data['mesh_data']['physical_groups'],
                                data['mesh_data']['nodes']['coordinates'])
    # make stage 1
    # create materials
    TOP_LE_1 = TOP_LE.copy()
    TOP_LE_1["name"] = "PorousDomain.TOP_1"
    TOP_LE_2 = TOP_LE.copy()
    TOP_LE_2["name"] = "PorousDomain.TOP_2"
    TOP_LE_3 = TOP_LE.copy()
    TOP_LE_3["name"] = "PorousDomain.TOP_3"
    gmsh_to_kratos = set_stage(1,
                               0.0,
                               1.0,
                               directory,
                               constrains_on_surfaces,
                               [TOP_LE_1, TOP_LE_2, TOP_LE_3, MIDDLE_LE, BOTTOM_LE],
                               3.08,
                               "linear_elastic",
                               data,
                               [line_load_input])

    # make stage 2
    constrains_on_surfaces = {
        "names": ["top_water_boundary_1", "top_water_boundary_2", "top_water_boundary_3", "middle_water_boundary",
                  "bottom_water_boundary"],
        "surfaces": ["TOP_1", "TOP_2", "TOP_3", "MIDDLE", "BOTTOM"],
        "material_per_surface": ["TOP_1", "TOP_2", "TOP_3", "MIDDLE", "BOTTOM"],
        "field_processes": [
            {"name": "TOP_1_RF", "variable": "UMAT_PARAMETERS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "TOP_1_mc_RF.json"},
            {"name": "TOP_2_RF", "variable": "UMAT_PARAMETERS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "TOP_2_mc_RF.json"},
            {"name": "TOP_3_RF", "variable": "UMAT_PARAMETERS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "TOP_3_mc_RF.json"},
            {"name": "MIDDLE_RF", "variable": "UMAT_PARAMETERS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "MIDDLE_mc_RF.json"},
            {"name": "BOTTOM_RF", "variable": "UMAT_PARAMETERS", "function": "dummy", "function_type": "json_file",
             "dataset_file_name": "BOTTOM_mc_RF.json"}],
    }
    # create materials
    TOP_MC_1 = TOP_MC.copy()
    TOP_MC_1["name"] = "PorousDomain.TOP_1"
    TOP_MC_2 = TOP_MC.copy()
    TOP_MC_2["name"] = "PorousDomain.TOP_2"
    TOP_MC_3 = TOP_MC.copy()
    TOP_MC_3["name"] = "PorousDomain.TOP_3"
    modify_MC_parameters([TOP_MC_1, TOP_MC_2, TOP_MC_3, MIDDLE_MC, BOTTOM_MC], input_values)
    gmsh_to_kratos = set_stage(2, 1.0, 2.0,
                               directory,
                               constrains_on_surfaces,
                               [TOP_MC_1, TOP_MC_2, TOP_MC_3, MIDDLE_MC, BOTTOM_MC],
                               input_values['HEAD'],
                               "mohr_coulomb",
                               data,
                               [line_load_input])

    return gmsh_to_kratos


if __name__ == '__main__':
    # open sqlite database and loop over the values
    conn = sqlite3.connect('inputs_outputs.db')
    c = conn.cursor()
    c.execute("SELECT * FROM inputs")
    results = c.fetchall()
    for result in results[:1]:
        input_values = {
            "INDEX": result[0],
            "YOUNG_MODULUS_TOP": result[2],
            "YOUNG_MODULUS_MIDDLE": result[5],
            "YOUNG_MODULUS_BOTTOM": result[8],
            "FRICTION_ANGLE_TOP": result[4],
            "FRICTION_ANGLE_MIDDLE": result[7],
            "FRICTION_ANGLE_BOTTOM": result[10],
            "HEAD": result[11],
            "LOAD": 0.0
        }
        directory = f"results_RF/{result[0]}"
        gmsh_to_kratos = create_model(directory, input_values)
        plt.close("all")
    conn.close()
    failed = []
    for result in results[:100]:
        try:
            directory = f"results_RF/{result[0]}"
            run_multistage_calculation(directory, 2)
        except:
            print(f"failed to run {result[0]}")
            failed.append(result[0])
            pass
    for result in results[:100]:
        try:
            directory = f"results_RF/{result[0]}"
            post_process(2, 2.0, gmsh_to_kratos, save=True, directory=directory, file_name="stage_3.png")
        except:
            print(f"failed to post process {result[0]}")
            pass
    print(failed)
