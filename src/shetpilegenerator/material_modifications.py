import os
import json

from stem.soil_material import TwoPhaseSoil, SmallStrainUdsmLaw, SaturatedBelowPhreaticLevelLaw, SoilMaterial, LinearElasticSoil
from stem.IO.kratos_material_io import KratosMaterialIO
from shetpilegenerator.material_library import *

DICT_NORMALIZATION = {
    "YOUNG_MODULUS": 1e9,
    "FRICTION_ANGLE": 90,
}


def unit_weight_to_density_solid(unit_weight, porosity, gravity, density_water=1000):
    return (1 / (1 - porosity)) * ((unit_weight * 1000 / gravity) - (porosity * density_water))


def create_linear_elastic_model(values_dict):
    density_solid = unit_weight_to_density_solid(values_dict.get("unit_weight", 18),
                                                 values_dict.get("POROSITY", 0.3),
                                                 values_dict.get("gravity", 9.81)
                                                 )
    # define soil 1
    soil_1_gen = TwoPhaseSoil(ndim=2,
                              DENSITY_SOLID=density_solid,
                              POROSITY=values_dict.get("POROSITY", 0.3),
                              BULK_MODULUS_SOLID=values_dict.get("BULK_MODULUS_SOLID", 1E9),
                              PERMEABILITY_XX=values_dict.get("PERMEABILITY_XX", 1E-15),
                              PERMEABILITY_YY=values_dict.get("PERMEABILITY_YY", 1E-15),
                              PERMEABILITY_XY=values_dict.get("PERMEABILITY_XY", 1E-15),

                              )
    # Define umat constitutive law parameters
    umat_constitutive_parameters = LinearElasticSoil(YOUNG_MODULUS=values_dict.get("YOUNG_MODULUS", 1E9),
                                                     POISSON_RATIO=values_dict.get("POISSON_RATIO", 0.3),
                                                     )
    soil_1_water = SaturatedBelowPhreaticLevelLaw()
    soil_1 = SoilMaterial(name=values_dict.get("name", "soil_1"),
                          soil_formulation=soil_1_gen,
                          constitutive_law=umat_constitutive_parameters,
                          retention_parameters=soil_1_water)
    return soil_1



def mohr_coulomb_parameters_to_list(umat_parameters):
    return [umat_parameters["YOUNG_MODULUS"],
            umat_parameters["POISSON_RATIO"],
            umat_parameters["COHESION"],
            umat_parameters["FRICTION_ANGLE"],
            umat_parameters["DILATANCY_ANGLE"],
            umat_parameters["CUTOFF_STRENGTH"],
            umat_parameters["YIELD_FUNCTION_TYPE"],
            umat_parameters["UNDRAINED_POISSON_RATIO"],
            ]


def create_mohr_coloumb_model(values_dict):
    density_solid = unit_weight_to_density_solid(values_dict.get("unit_weight", 18),
                                                 values_dict.get("POROSITY", 0.3),
                                                 values_dict.get("gravity", 9.81)
                                                 )
    umat_parameters = mohr_coulomb_parameters_to_list(values_dict.get("UMAT_PARAMETERS", {}))
    # define soil 1
    soil_1_gen = TwoPhaseSoil(ndim=2,
                              DENSITY_SOLID=density_solid,
                              POROSITY=values_dict.get("POROSITY", 0.3),
                              BULK_MODULUS_SOLID=values_dict.get("BULK_MODULUS_SOLID", 1E9),
                              PERMEABILITY_XX=values_dict.get("PERMEABILITY_XX", 1E-15),
                              PERMEABILITY_YY=values_dict.get("PERMEABILITY_YY", 1E-15),
                              PERMEABILITY_XY=values_dict.get("PERMEABILITY_XY", 1E-15),
                              )
    # Define umat constitutive law parameters
    umat_constitutive_parameters = SmallStrainUdsmLaw(UDSM_PARAMETERS=umat_parameters,
                                                      UDSM_NAME=values_dict.get("UDSM_NAME", "D:/SheetPileGenerator/test/kratos_write_test/MohrCoulomb64.dll"),
                                                      UDSM_NUMBER=values_dict.get("UDSM_NUMBER", 1),
                                                      IS_FORTRAN_UDSM=True,
                                                      )
    soil_1_water = SaturatedBelowPhreaticLevelLaw()
    soil_1 = SoilMaterial(name=values_dict.get("name", "soil_1"),
                          soil_formulation=soil_1_gen,
                          constitutive_law=umat_constitutive_parameters,
                          retention_parameters=soil_1_water)
    return soil_1


def write_materials_dict(materials, soil_model="mohr_coulomb"):
    if soil_model == "mohr_coulomb":
        all_materials = [create_mohr_coloumb_model(values_dict) for values_dict in materials]
    elif soil_model == "linear_elastic":
        all_materials = [create_linear_elastic_model(values_dict) for values_dict in materials]
    return all_materials

def modify_MC_parameters(materials, mc_initial_parameters):
    materials[0]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_TOP']
    materials[1]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_TOP']
    materials[2]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_TOP']
    materials[3]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_MIDDLE']
    materials[4]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_BOTTOM']
    materials[0]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_TOP']
    materials[1]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_TOP']
    materials[2]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_TOP']
    materials[3]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_MIDDLE']
    materials[4]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_BOTTOM']
    return materials


def plot_materials(materials, layers, material_names, save=False, file_name="geometry.png", directory="."):
    for material_name in material_names:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        fig, ax = plt.subplots(figsize=(10, 10))
        for counter, layer in enumerate(layers):
            material = materials[counter]
            # get the values for the material
            value = material['UMAT_PARAMETERS'][material_name]
            # normalize the value
            norm_value = DICT_NORMALIZATION[material_name]
            my_cmap = cm.get_cmap('Greys')  # or any other one
            norm = colors.Normalize(0, norm_value)
            color_i = my_cmap(norm(value))  # returns an rgba value
            # plot the value with the color
            points_2d = [point[:2] for point in layer]
            # plot patches with the value as color
            ax.add_patch(patches.Polygon(points_2d, color=color_i))
        ax.autoscale()
        # turn off the axis
        ax.axis('off')
        if save:
            # save the figure in the directory and create the directory if it does not exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(os.path.join(directory, f"{material_name}_{file_name}"))
        else:
            plt.show()
