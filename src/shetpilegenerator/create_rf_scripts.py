from jinja2 import Template, DebugUndefined

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

NUMBER_OF_SAMPLES = 1000

for sample_number in range(1, NUMBER_OF_SAMPLES + 1):

    le_files = {f"TOP_1_le_RF_{sample_number}.py": MATERIAL_VALUES['TOP'],
                f"TOP_2_le_RF_{sample_number}.py": MATERIAL_VALUES['TOP'],
                f"TOP_3_le_RF_{sample_number}.py": MATERIAL_VALUES['TOP'],
                f"MIDDLE_le_RF_{sample_number}.py": MATERIAL_VALUES['MIDDLE'],
                f"BOTTOM_le_RF_{sample_number}.py": MATERIAL_VALUES['BOTTOM']}
    # import the txt files
    with open("../test/kratos_write_test/template_linear_elastic.txt", "r") as f:
        template_le = f.read()
    j2_template = Template(template_le)
    for file_name, values in le_files.items():
        with open("D:/Kratos_general/Kratos_build_version/KratosGeoMechanics/KratosMultiphysics/GeoMechanicsApplication/user_defined_scripts/" + file_name, "w") as f:
            f.write(j2_template.render(mean=values['YOUNG_MODULUS']['m'],
                                       std_value=values['YOUNG_MODULUS']['s'],
                                       aniso_x=40,
                                       aniso_y=10,
                                       seed=sample_number + 671993)
                    )

    mc_files = {f"TOP_1_mc_RF_{sample_number}.py": MATERIAL_VALUES['TOP'],
                f"TOP_2_mc_RF_{sample_number}.py": MATERIAL_VALUES['TOP'],
                f"TOP_3_mc_RF_{sample_number}.py": MATERIAL_VALUES['TOP'],
                f"MIDDLE_mc_RF_{sample_number}.py": MATERIAL_VALUES['MIDDLE'],
                f"BOTTOM_mc_RF_{sample_number}.py": MATERIAL_VALUES['BOTTOM']}
    with open("../test/kratos_write_test/template_mc.txt", "r") as f:
        template_mc = f.read()
    j2_template = Template(template_mc)
    for file_name, values in mc_files.items():
        with open("D:/Kratos_general/Kratos_build_version/KratosGeoMechanics/KratosMultiphysics/GeoMechanicsApplication/user_defined_scripts/" + file_name, "w") as f:
            f.write(j2_template.render(mean=values['YOUNG_MODULUS']['m'],
                                       std_value=values['YOUNG_MODULUS']['s'],
                                       aniso_x=40,
                                       aniso_y=10,
                                       seed=sample_number + 671993)
                    )

