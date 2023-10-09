import json

DEFAULT_GRAVITY = {
                "python_module": "apply_vector_constraint_table_process",
                "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
                "process_name": "ApplyVectorConstraintTableProcess",
                "Parameters": {
                    "model_part_name": "PorousDomain.gravity",
                    "variable_name": "VOLUME_ACCELERATION",
                    "active": [
                        False,
                        True,
                        False
                    ],
                    "value": [
                        0.0,
                        -9.81,
                        0.0
                    ],
                    "table": [
                        0,
                        0,
                        0
                    ]
                }
            }


def add_and_save_water_and_rfs(template_file, output_file, extra_parameters):
    with open(template_file, 'r') as json_file:
        data = json.load(json_file)
    # change the nodal smoothing part otherwise it fails in the umat part
    data["solver_settings"]['nodal_smoothing'] = False
    for key, value in extra_parameters.items():
        data["processes"][key] += value
    # add gravity
    data["processes"]["loads_process_list"].append(DEFAULT_GRAVITY)
    for value in extra_parameters["loads_process_list"]:
        name = value["Parameters"]["model_part_name"].split(".")[-1]
        data['solver_settings']['processes_sub_model_part_list'].append(name)
    # add the gravity model part
    data['solver_settings']['processes_sub_model_part_list'].append("gravity")
    with open(output_file, 'w') as json_file:
        json.dump(data,
                  json_file,
                  indent=4)