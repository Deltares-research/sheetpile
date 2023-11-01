from gstools import SRF, Gaussian
import numpy as np
import json
from .material_library import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os


MATERIAL_VALUES = {
    "TOP" : {
        "YOUNG_MODULUS": {"m": 125e6, "s": 15e6},
        "UNIT_WEIGHT": {"m": 22, "s": 2},
        "FRICION_ANGLE": {"m": 39.8, "s": 2},
    },
    "MIDDLE": {
        "YOUNG_MODULUS": {"m": 6500000, "s": 2e6},
        "UNIT_WEIGHT": {"m": 15, "s": 2},
        "FRICION_ANGLE": {"m": 25.0, "s": 2},
    },
    "BOTTOM": {
        "YOUNG_MODULUS": {"m": 50e6, "s": 10e6},
        "UNIT_WEIGHT": {"m": 18, "s": 2},
        "FRICION_ANGLE": {"m": 37, "s": 1},
    }
}

ELEMENTS = "TRIANGLE_3N"


def generate_field(mean, std_value, aniso_x, aniso_y, coordinates, seed):

    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]
    # set random field parameters
    len_scale = np.array([aniso_x, aniso_y])
    var = std_value ** 2

    model = Gaussian(dim=2, var=var, len_scale=len_scale, seed=seed)
    srf = SRF(model, mean=mean, seed=seed)
    new_values = srf((x, y), mesh_type="unstructured", seed=seed)
    # if the values are negative, set them to mean
    new_values[new_values < 0] = mean
    return new_values.tolist()


def generate_jsons_for_le(sample_number, directory, stem_model):
    # set up all the parameters
    le_files = [{"json_names": f"{directory}/TOP_1_le_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_1"},
                {"json_names": f"{directory}/TOP_2_le_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_2"},
                {"json_names": f"{directory}/TOP_3_le_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_3"},
                {"json_names": f"{directory}/MIDDLE_le_RF.json", "material_parameters": MATERIAL_VALUES['MIDDLE'], "physical_group": "MIDDLE"},
                {"json_names": f"{directory}/BOTTOM_le_RF.json", "material_parameters": MATERIAL_VALUES['BOTTOM'], "physical_group": "BOTTOM"}]
    young_modulus = []
    for counter, file in enumerate(le_files):
        mean = file["material_parameters"]["YOUNG_MODULUS"]["m"]
        std_value = file["material_parameters"]["YOUNG_MODULUS"]["s"]
        aniso_x = 50
        aniso_y = 8
        seed = sample_number + counter + 761993
        elements_pg = stem_model.body_model_parts[counter].mesh.elements
        connectivities_pg = [value.node_ids for key, value in elements_pg.items()]
        global_nodes = stem_model.gmsh_io.mesh_data['nodes']
        # get element coordinates
        elements_coordinates = []
        for node in connectivities_pg:
            # get the center of the element
            x = (global_nodes[int(node[0])][0] + global_nodes[int(node[1])][0] + global_nodes[int(node[2])][0]) / 3
            y = (global_nodes[int(node[0])][1] + global_nodes[int(node[1])][1] + global_nodes[int(node[2])][1]) / 3
            elements_coordinates.append([x, y])
        new_values = generate_field(mean, std_value, aniso_x, aniso_y, elements_coordinates, seed)
        # check that the length of the values is the same as the number of elements
        assert len(new_values) == len(elements_coordinates)
        young_modulus.append(new_values)
        dict_values = {"values": new_values}
        # write the new values in the json file
        with open(file["json_names"], "w") as json_file:
            json.dump(dict_values, json_file, indent=4)
    return young_modulus


def generate_jsons_for_mc(sample_number, directory, stem_model, young_modulus):
    total_results = []
    # set up all the parameters
    le_files = [{"json_names": f"{directory}/TOP_1_mc_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_1", "material": TOP_MC},
                {"json_names": f"{directory}/TOP_2_mc_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_2", "material": TOP_MC},
                {"json_names": f"{directory}/TOP_3_mc_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_3", "material": TOP_MC},
                {"json_names": f"{directory}/MIDDLE_mc_RF.json", "material_parameters": MATERIAL_VALUES['MIDDLE'], "physical_group": "MIDDLE", "material": MIDDLE_MC},
                {"json_names": f"{directory}/BOTTOM_mc_RF.json", "material_parameters": MATERIAL_VALUES['BOTTOM'], "physical_group": "BOTTOM", "material": BOTTOM_MC}]
    for counter, file in enumerate(le_files):
        aniso_x = 50
        aniso_y = 8
        seed = sample_number + counter + 761993
        elements_pg = stem_model.body_model_parts[counter].mesh.elements
        connectivities_pg = [value.node_ids for key, value in elements_pg.items()]
        global_nodes = stem_model.gmsh_io.mesh_data['nodes']
        # get element coordinates
        elements_coordinates = []
        for node in connectivities_pg:
            # get the center of the element
            x = (global_nodes[int(node[0])][0] + global_nodes[int(node[1])][0] + global_nodes[int(node[2])][0]) / 3
            y = (global_nodes[int(node[0])][1] + global_nodes[int(node[1])][1] + global_nodes[int(node[2])][1]) / 3
            elements_coordinates.append([x, y])
        # sample young modulus
        youngs_modulus = young_modulus[counter]
        # sample the friction angle
        mean = file["material_parameters"]["FRICION_ANGLE"]["m"]
        std_value = file["material_parameters"]["FRICION_ANGLE"]["s"]
        friction_angle = generate_field(mean, std_value, aniso_x, aniso_y, elements_coordinates, seed)
        # get everything else in list format
        values = []
        for i in range(len(youngs_modulus)):
            values.append([youngs_modulus[i],
                           file["material"]["UMAT_PARAMETERS"]["POISSON_RATIO"],
                           file["material"]["UMAT_PARAMETERS"]["COHESION"],
                           friction_angle[i],
                           file["material"]["UMAT_PARAMETERS"]["DILATANCY_ANGLE"],
                           file["material"]["UMAT_PARAMETERS"]["CUTOFF_STRENGTH"],
                           file["material"]["UMAT_PARAMETERS"]["YIELD_FUNCTION_TYPE"],
                           file["material"]["UMAT_PARAMETERS"]["UNDRAINED_POISSON_RATIO"]])

        dict_values = {"values": values}
        # write the new values in the json file
        with open(file["json_names"], "w") as json_file:
            json.dump(dict_values, json_file, indent=4)
        total_results.append(values)
    return total_results


def generate_jsons_for_material(sample_number, directory, stem_model):
    young_modulus = generate_jsons_for_le(sample_number, directory, stem_model)
    results_rd_mc = generate_jsons_for_mc(sample_number, directory, stem_model, young_modulus)



