from typing import List, Dict
import numpy as np


GMSH_TO_KRATOS_ELEMENTS = {
    "TRIANGLE_3N" : "UPwSmallStrainElement2D3N",
}



class MdpaFileModify:

    def __init__(self, model_stem):
        self.model = model_stem

    def add_missing_boundaries(self, mpda_file: str, constrains_on_surfaces: Dict[str, str], model_parts_ids: List[int]):
        # create mdpa file
        file = open(mpda_file, "a")
        file = self.write_field_processes(file, constrains_on_surfaces, model_parts_ids)
        file = self.add_gravity(file)
        file.close()
        return

    def add_gravity(self, file):
        # get all the nodes
        nodes = list(self.model.gmsh_io.mesh_data["nodes"].keys())
        # get all the elements
        elements = []
        for key, value in self.model.gmsh_io.mesh_data['physical_groups'].items():
            if value['element_type'] == 'TRIANGLE_3N':
                elements.append(value['element_ids'])
        elements = [item for sublist in elements for item in sublist]
        # write the nodes
        file.write(f"Begin SubModelPart gravity\n")
        file.write(f"  Begin SubModelPartTables\n")
        file.write(f"  End SubModelPartTables\n")
        file.write(f"  Begin SubModelPartNodes\n")
        for node in nodes:
            file.write(f"    {node}\n")
        file.write(f"  End SubModelPartNodes\n")
        # write the elements
        file.write(f"  Begin SubModelPartElements\n")
        for element in elements:
            file.write(f"    {element}\n")
        file.write(f"  End SubModelPartElements\n")
        file.write(f"  Begin SubModelPartProperties\n")
        file.write(f"  End SubModelPartProperties\n")
        file.write(f"End SubModelPart\n\n")
        return file


    def write_field_processes(self, file, field_processes, model_parts_ids):
        for counter, field_process in enumerate(field_processes):
            name = field_process['Parameters']['model_part_name'].split('.')[-1]
            file.write(f"Begin SubModelPart {name}\n")
            file.write(f"  Begin SubModelPartTables\n")
            file.write(f"  End SubModelPartTables\n")
            file.write(f"  Begin SubModelPartNodes\n")
            # get all the nodes from the physical groups also sorted
            nodes_dict = self.model.body_model_parts[model_parts_ids[counter]].mesh.nodes
            nodes_list = np.array(list(nodes_dict.keys()))
            # sort the nodes
            nodes = np.sort(nodes_list)
            for node in nodes:
                file.write(f"    {node}\n")
            file.write(f"  End SubModelPartNodes\n")
            file.write(f"  Begin SubModelPartElements\n")
            elements_dict = self.model.body_model_parts[model_parts_ids[counter]].mesh.elements
            elements_list = np.array(list(elements_dict.keys()))
            # sort the elements
            elements = np.sort(elements_list)
            for element in elements:
                file.write(f"    {element}\n")
            file.write(f"  End SubModelPartElements\n")
            file.write(f"  Begin SubModelPartProperties\n")
            file.write(f"  End SubModelPartProperties\n")
            file.write(f"End SubModelPart\n\n")
        return file


