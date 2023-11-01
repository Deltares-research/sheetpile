from shetpilegenerator.mdpa_file_modify import MdpaFileModify
from shetpilegenerator.material_library import *
from shetpilegenerator.rf_generator import generate_jsons_for_material
from shetpilegenerator.material_modifications import write_materials_dict
from shetpilegenerator.water_pressures_modifications import define_water_boundaries, define_water_line_based_on_outside_head
from shetpilegenerator.project_parameters_modifications import add_and_save_water_and_rfs
from shetpilegenerator.output_utils import plot_geometry
from shetpilegenerator.layers_definition import define_layers, define_layers_with_dike_geometry, test_define_layers_with_dike_geometry
from shetpilegenerator.sheetpile_setup import add_line_by_coordinates, SHEETPILE_MATERIALS


from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.boundary import DisplacementConstraint
from stem.load import LineLoad
from stem.model import Model
from stem.IO.kratos_io import KratosIO
from stem.output import NodalOutput, GiDOutputParameters, Output, GaussPointOutput, JsonOutputParameters
from stem.model_part import BodyModelPart
from stem.structural_material import StructuralMaterial, EulerBeam
from stem.additional_processes import Excavation
import pickle
import numpy as np


NUMBER_OF_SAMPLES = 10  # number of samples to generate


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


def get_all_lines(model):
    lines_list = [part.geometry.lines for part in model.body_model_parts]
    lines = {}
    for dictionary in lines_list:
        lines.update(dictionary)
    return lines

def get_all_points(model):
    points_list = [part.geometry.points for part in model.body_model_parts]
    points = {}
    for dictionary in points_list:
        points.update(dictionary)
    return points

def get_ids_from_coordinates(model, coordinates):
    start_coordinate = coordinates[0]
    end_coordinate = coordinates[1]
    lines = get_all_lines(model)
    points = get_all_points(model)

    # check that x coordinates are the same
    if start_coordinate[0] != end_coordinate[0]:
        raise ValueError("x coordinates are not the same sheetpile is not vertical")
    
    line_ids = []
    for id, line in lines.items():
        x_coords = [points[point_id].coordinates[0] for point_id in line.point_ids]
        y_coords = [points[point_id].coordinates[1] for point_id in line.point_ids]
        x_coordinates_are_the_same = x_coords[0] == start_coordinate[0] and x_coords[1] == end_coordinate[0]
        y_coordinates_are_between_0 = y_coords[0] <= start_coordinate[1] and y_coords[0] >= end_coordinate[1]
        y_coordinates_are_between_1 = y_coords[1] <= start_coordinate[1] and y_coords[1] >= end_coordinate[1]
        if x_coordinates_are_the_same and y_coordinates_are_between_0 and y_coordinates_are_between_1:
            line_ids.append(id)
    return line_ids


def get_geometry_ids_boundaries(model, left_boundary_coord, right_boundary_coord, bottom_boundary_coord):
    # Method to get all vertical lines on the left and right boundary and horizontal lines at the bottom boundary
    line_ids_fixed, line_ids_roller = list(), list()

    lines = get_all_lines(model)
    points = get_all_points(model)

    for id, line in lines.items():
        x_coords = [points[point_id].coordinates[0] for point_id in line.point_ids]
        y_coords = [points[point_id].coordinates[1] for point_id in line.point_ids]

        if (x_coords[0] == left_boundary_coord and x_coords[1] == left_boundary_coord) or \
                (x_coords[0] == right_boundary_coord and x_coords[1] == right_boundary_coord):
            line_ids_roller.append(id)

        elif (y_coords[0] == bottom_boundary_coord and y_coords[1] == bottom_boundary_coord):
            line_ids_fixed.append(id)

        else:
            pass

    if len(line_ids_fixed) < 1:
        print("No line found on the bottom boundary")
    if len(line_ids_roller) < 2:
        print("Not enough lines found on left or right boundary (less than 2) ")

    return line_ids_fixed, line_ids_roller

def set_up_solver(start_time, end_time):
    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=start_time, end_time=end_time, delta_time=0.25, reduction_factor=1.0,
                                       increase_factor=1.0, max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-2,
                                                            displacement_absolute_tolerance=1.0e-4)
    strategy_type = NewtonRaphsonStrategy(min_iterations=6, max_iterations=50, number_cycles=100)
    scheme_type = NewmarkScheme(newmark_beta=0.25, newmark_gamma=0.5, newmark_theta=0.5)
    linear_solver_settings = Amgcl(tolerance=1e-8, max_iteration=500, scaling=True)
    stress_initialisation_type = StressInitialisationType.GRAVITY_LOADING
    solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=False, are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=strategy_type, scheme=scheme_type,
                                     linear_solver_settings=linear_solver_settings, rayleigh_k=0.0,reset_displacements=True,
                                     rayleigh_m=0.0)
    return solver_settings


def set_up_output_settings(nodal_results, gauss_point_results, stage_number):
    # Define the output process
    porous_media = ["TOP_1_RF",
                    "TOP_2_RF",
                    "TOP_3_RF",
                    "MIDDLE_RF",
                    "BOTTOM_RF"]
    outputs_soils = []
    for porous_medium in porous_media:
        outputs_soils.append( Output(
            part_name=porous_medium,
            output_name=f"gid_output_{stage_number}_{porous_medium}",
            output_dir="output",
            output_parameters=GiDOutputParameters(
                file_format="ascii",
                output_interval=1,
                nodal_results=nodal_results,
                gauss_point_results=gauss_point_results,
                output_control_type="step"
            )
        )
        )
    gid_output_sheetpile = Output(
        part_name="sheetpile",
        output_name=f"gid_output_{stage_number}_sheetpile",
        output_dir="output",
        output_parameters=GiDOutputParameters(
            file_format="ascii",
            output_interval=1,
            nodal_results=[NodalOutput.DISPLACEMENT],
            gauss_point_results=[GaussPointOutput.MOMENT, GaussPointOutput.FORCE],
            output_control_type="step"
        )
    )
    json_output = Output(
        part_name="gravity_load_2d",
        output_name=f"json_output_{stage_number}",
        output_dir="output",
        output_parameters=JsonOutputParameters(
            time_frequency=0.1,
            nodal_results=nodal_results,
            gauss_point_results=gauss_point_results,
        )
    )

    return outputs_soils + [json_output, gid_output_sheetpile]


def write_input_files(model, outputs, output_folder, mdpa_file, materials_file, project_file_name):
    kratos_io = KratosIO(ndim=model.ndim)
    # Write project settings to ProjectParameters.json file
    kratos_io.write_project_parameters_json(
        model=model,
        outputs=outputs,
        mesh_file_name=mdpa_file,
        materials_file_name=materials_file,
        output_folder=output_folder,
        project_file_name=project_file_name
    )

    # Write mesh to .mdpa file
    kratos_io.write_mesh_to_mdpa(
        model=model,
        mesh_file_name=mdpa_file,
        output_folder=output_folder
    )

    # Write materials to MaterialParameters.json file
    kratos_io.write_material_parameters_json(
        model=model,
        output_folder=output_folder,
        materials_file_name=materials_file
    )


def set_up_sheetpile(model, line_sheetpile, sheetpile_active):
    beam = EulerBeam(ndim=2, YOUNG_MODULUS=SHEETPILE_MATERIALS['YOUNG_MODULUS'], DENSITY=5, CROSS_AREA=1, I33=200, POISSON_RATIO=0.3)
    material_sheetpile = StructuralMaterial(name="sheetpile", material_parameters=beam)
    add_line_by_coordinates(model, line_sheetpile, material_sheetpile, "sheetpile")
    excavation = Excavation(deactivate_body_model_part=not(sheetpile_active))
    lines = get_ids_from_coordinates(model, line_sheetpile)
    model.add_boundary_condition_by_geometry_ids(
        ndim_boundary=1,
        geometry_ids=lines,
        boundary_parameters=excavation,
        name="sheetpile_excavation"
    )


def set_stage(index, stage_number, start_time, end_time, project_path, constrains_on_surfaces, soils, head_level, soil_model,
              layers, loads, sheetpile_active, sheetpile_coordinates):
    materials_collection = write_materials_dict(soils, soil_model=soil_model)
    # create Model
    model = Model(2)
    for counter, layer in enumerate(layers):
        model.add_soil_layer_by_coordinates(layer, materials_collection[counter], soils[counter]['name'])
    # define water conditions stage 1
    TOP_1_WL, TOP_2_WL, TOP_3_WL, WL2, WL3 = define_water_line_based_on_outside_head(head_level)
    dictionary_water_boundaries_stage3 = define_water_boundaries(TOP_1_WL, TOP_2_WL, TOP_3_WL, WL2, WL3)
    # add the loads
    if loads is not None:
        for load in loads:
            model.add_load_by_coordinates([[0., 1., 0.], [0., 2., 0.]], load['load'], load['name'])
    # Detect boundary conditions
    line_ids_fixed, line_ids_roller = get_geometry_ids_boundaries(model,
                                                                  left_boundary_coord=LEFT_BOUND,
                                                                  right_boundary_coord=RIGHT_BOUND,
                                                                  bottom_boundary_coord=BOTTOM_BOUND)

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, False],
                                                            value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(1, line_ids_fixed, no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(1, line_ids_roller, roller_displacement_parameters, "roller_fixed")
    # set the sheetpile
    set_up_sheetpile(model, sheetpile_coordinates, sheetpile_active)
    # Set mesh size and generate mesh
    # set up solver
    solver_settings = set_up_solver(start_time, end_time)
    # Set up problem data
    problem = Problem(problem_name=f"analysis_{stage_number}", number_of_threads=1, settings=solver_settings)
    model.project_parameters = problem
    # set outputs
    outputs = set_up_output_settings(nodal_results=nodal_results, gauss_point_results=gauss_point_results, stage_number=stage_number)
    model.post_setup()
    #model.show_geometry(show_line_ids=True, show_point_ids=True)
    model.mesh_settings.element_size = -1
    model.generate_mesh()

    write_input_files(model, outputs, project_path, f"analysis_{stage_number}", f"MaterialParameters_{stage_number}.json", f"Project_Parameters_{stage_number}.json")

    # write the random field json files
    generate_jsons_for_material(index, project_path, model)

    # add the field processes
    sub_model_names_field_process = []
    for field_process in constrains_on_surfaces['field_processes']:
        sub_model_names_field_process.append(field_process['name'])
        dictionary_water_boundaries_stage3.append(get_field_process_dict(field_process))
    # define stage 1 parameters
    add_and_save_water_and_rfs(project_path + f"/Project_Parameters_{stage_number}.json",
                               project_path + f"/Project_Parameters_{stage_number}.json",
                               {"loads_process_list": dictionary_water_boundaries_stage3})
    # Add the field processes to the mdpa file
    modifier = MdpaFileModify(model)
    ids_model_parts = list(range(0, len(model.body_model_parts) - 1 )) + list(range(0, len(model.body_model_parts) - 1))
    modifier.add_missing_boundaries(project_path + f"/analysis_{stage_number}.mdpa", dictionary_water_boundaries_stage3, ids_model_parts)

    return model


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
    sheetpile_coordinates = [[-12, input_values['HEIGHT'], 0], [-12, 0, 0]]
    # set default loads
    # define the line load
    line_load = LineLoad(active=[True, False, False], value=[0.0, 0.0, 0.0])
    line_load_input = {"name": "dike_load", "load": line_load}
    # create materials
    # create materials
    TOP_LE_1 = TOP_LE.copy()
    TOP_LE_1["name"] = "TOP_1"
    TOP_LE_2 = TOP_LE.copy()
    TOP_LE_2["name"] = "TOP_2"
    TOP_LE_3 = TOP_LE.copy()
    TOP_LE_3["name"] = "TOP_3"
    material_list = [TOP_LE_1, TOP_LE_2, TOP_LE_3, MIDDLE_LE, BOTTOM_LE]
    # define layer points
    layers = define_layers_with_dike_geometry(input_values['HEIGHT'], input_values['WIDTH'], input_values['ANGLE'])
    plot_geometry(layers, save=True, directory=directory)
    # make stage 1
    gmsh_to_kratos = set_stage(index=input_values['INDEX'],
                               stage_number=1,
                               start_time=0.0,
                               end_time=1.0,
                               project_path=directory,
                               constrains_on_surfaces=constrains_on_surfaces,
                               soils=material_list,
                               head_level=3.08,
                               soil_model="linear_elastic",
                               layers=layers,
                               loads=[line_load_input],
                               sheetpile_active=False,
                               sheetpile_coordinates=sheetpile_coordinates)

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
    TOP_MC_1["name"] = "TOP_1"
    TOP_MC_2 = TOP_MC.copy()
    TOP_MC_2["name"] = "TOP_2"
    TOP_MC_3 = TOP_MC.copy()
    TOP_MC_3["name"] = "TOP_3"
    material_list = [TOP_MC_1, TOP_MC_2, TOP_MC_3, MIDDLE_MC, BOTTOM_MC]
    gmsh_to_kratos = set_stage(index=input_values['INDEX'],
                               stage_number=2,
                               start_time=1.0,
                               end_time=2.0,
                               project_path=directory,
                               constrains_on_surfaces=constrains_on_surfaces,
                               soils=material_list,
                               head_level=input_values['HEAD'],
                               soil_model="mohr_coulomb",
                               layers=layers,
                               loads=[line_load_input],
                               sheetpile_active=False,
                               sheetpile_coordinates=sheetpile_coordinates)
    
    # make stage 3
    gmsh_to_kratos = set_stage(index=input_values['INDEX'],
                               stage_number=3,
                               start_time=2.0,
                               end_time=3.0,
                               project_path=directory,
                               constrains_on_surfaces=constrains_on_surfaces,
                               soils=material_list,
                               head_level=input_values['HEAD'],
                               soil_model="mohr_coulomb",
                               layers=layers,
                               loads=[line_load_input],
                               sheetpile_active=True,
                               sheetpile_coordinates=sheetpile_coordinates)

    return gmsh_to_kratos

def pick_values_from_normal_distribution(mean, std, size):
    return np.random.normal(mean, std, size)

def pick_values_from_uniform_distribution(min, max, size):
    return np.random.uniform(min, max, size)


if __name__ == '__main__':
    # open sqlite database and loop over the values
    total_models = 1000
    default_value = 100
    LEFT_BOUND = -80.0
    RIGHT_BOUND = 80.0
    BOTTOM_BOUND = -15.0
    nodal_results = [NodalOutput.DISPLACEMENT,
                     NodalOutput.TOTAL_DISPLACEMENT,
                     NodalOutput.WATER_PRESSURE

                     ]
    # Gauss point results
    gauss_point_results = [GaussPointOutput.CAUCHY_STRESS_VECTOR,
                           GaussPointOutput.GREEN_LAGRANGE_STRAIN_VECTOR]
    # sensitivity analysis dike geometry
    #import matplotlib.pyplot as plt
    #angles = np.linspace(30, 60, 1000)
    #fig, ax = plt.subplots()
    #cmap = plt.get_cmap('viridis')
    #for angle in angles:
    #    point_list = test_define_layers_with_dike_geometry(8, 5, angle)
    #    point_list = np.array(point_list)
    #    # color of the line is the angle
    #    normalized_angle = angle / 50
    #    color = cmap(normalized_angle)
    #    ax.plot(point_list[:, 0], point_list[:, 1], color=color)
    ## Create a ScalarMappable to map values to colors for the colorbar
    #sm = plt.cm.ScalarMappable(cmap=cmap)
    #sm.set_array([])  # Set an empty array to the ScalarMappable
    ## Add a colorbar
    #cbar = plt.colorbar(sm, ax=ax)
    #cbar.set_label('Angle')
    #ax.set_xlabel('x [m]')
    #ax.set_ylabel('y [m]')
    ## set title
    #ax.set_title('Dike geometry sensitivity analysis for the angle')
    #plt.show()
    # define the head values
    head_values = pick_values_from_normal_distribution(15, 1, total_models)
    angles = pick_values_from_uniform_distribution(30, 60, total_models)
    for id in range(1, total_models):
        input_values = {
            "INDEX": id,
            "YOUNG_MODULUS_TOP": default_value,
            "YOUNG_MODULUS_MIDDLE": default_value,
            "YOUNG_MODULUS_BOTTOM": default_value,
            "FRICTION_ANGLE_TOP": default_value,
            "FRICTION_ANGLE_MIDDLE": default_value,
            "FRICTION_ANGLE_BOTTOM": default_value,
            "HEAD": head_values[id],
            "LOAD": 0.0,
            "ANGLE": angles[id],
            "HEIGHT": 8.0,
            "WIDTH": 5.0,
        }
        directory = f"output/{id}"
        gmsh_to_kratos = create_model(directory, input_values)
        # dump the model part to pickle
        pickle.dump(gmsh_to_kratos, open(f"{directory}/model.p", "wb"))
    # add folder to bucket