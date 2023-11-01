import json


def add_and_save_water_and_rfs(template_file, output_file, extra_parameters):
    with open(template_file, 'r') as json_file:
        data = json.load(json_file)
    # pop start time in solver settings if it exists
    if "start_time" in data["solver_settings"]:
        data["solver_settings"].pop("start_time")
    # change the nodal smoothing part otherwise it fails in the umat part
    data["solver_settings"]['nodal_smoothing'] = False
    for key, value in extra_parameters.items():
        data["processes"][key] += value
    for value in extra_parameters["loads_process_list"]:
        name = value["Parameters"]["model_part_name"].split(".")[-1]
        data['solver_settings']['processes_sub_model_part_list'].append(name)
    with open(output_file, 'w') as json_file:
        json.dump(data,
                  json_file,
                  indent=4)