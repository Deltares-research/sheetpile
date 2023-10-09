import json


class OutputProcessJsonReader:

    def __init__(self, json_file_name):
        self.json_file_name = json_file_name
        self.get_output_variables()

    def get_output_variables(self):
        with open(self.json_file_name) as json_file:
            self.data = json.load(json_file)

    def get_values_in_timestep(self, timestep):
        # get timestep index
        timestep_index = self.data['TIME'].index(timestep)
        # get values for nodes
        water_pressure, total_displacement = [], []
        for key, value in self.data.items():
            if "NODE" in key:
                total_displacement.append(value['TOTAL_DISPLACEMENT_Y'][timestep_index])
                water_pressure.append(value['WATER_PRESSURE'][timestep_index])
        return water_pressure, total_displacement