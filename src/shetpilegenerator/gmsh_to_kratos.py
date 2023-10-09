from typing import List, Dict
import numpy as np


GMSH_TO_KRATOS_ELEMENTS = {
    "TRIANGLE_3N" : "UPwSmallStrainElement2D3N",
}



class GmshToKratos:

    def __init__(self, mesh_dictionary: dict):
        self.gmsh_dict = mesh_dictionary
        self.default_element_type = "TRIANGLE_3N"

    def read_gmsh_to_kratos(self, property_list: List[int], mpda_file: str, constrains_on_surfaces: Dict[str, str], top_load=False ):
        # create mdpa file
        with open(mpda_file, "w") as file:
            self.write_properties(file, property_list)
            self.write_nodes(self.gmsh_dict["mesh_data"]["nodes"] , file)
            self.write_elements(self.gmsh_dict["mesh_data"], file)
            self.write_materials(constrains_on_surfaces,
                                 list(self.gmsh_dict["mesh_data"]["physical_groups"].keys()),
                                 self.gmsh_dict["mesh_data"]["physical_groups"],
                                 file)
            self.write_field_processes(file, constrains_on_surfaces['field_processes'])
            self.write_gravity(file)
            self.write_boundary_conditions(file, constrains_on_surfaces, top_load)
            self.write_output_processes(file)
        return

    def write_field_processes(self, file, field_processes):
        for field_process in field_processes:
            file.write(f"Begin SubModelPart {field_process['name']}\n")
            file.write(f"  Begin SubModelPartTables\n")
            file.write(f"  End SubModelPartTables\n")
            file.write(f"  Begin SubModelPartNodes\n")
            # get all the nodes from the physical groups also sorted
            nodes = self.gmsh_dict["mesh_data"]["physical_groups"][field_process['name'].split("_RF")[0]][self.default_element_type]["connectivities"].flatten()
            nodes = np.unique(nodes)
            # sort the nodes
            nodes = np.sort(nodes)
            for node in nodes:
                file.write(f"    {node}\n")
            file.write(f"  End SubModelPartNodes\n")
            file.write(f"  Begin SubModelPartElements\n")
            elements = self.gmsh_dict["mesh_data"]["physical_groups"][field_process['name'].split("_RF")[0]][self.default_element_type]["element_ids"]
            for element in elements:
                file.write(f"    {element}\n")
            file.write(f"  End SubModelPartElements\n")
            file.write(f"  Begin SubModelPartProperties\n")
            file.write(f"  End SubModelPartProperties\n")
            file.write(f"End SubModelPart\n\n")

    def write_properties(self, file, property_list: List[int]):
        for property in property_list:
            file.write(f"Begin Properties {property}\n")
            file.write("End Properties\n\n")

    def write_nodes(self, nodes, file):
        file.write("Begin Nodes\n")
        for i in range(len(nodes['coordinates'])):
            file.write(f"  {int(nodes['ids'][i])}  {nodes['coordinates'][i][0]} {nodes['coordinates'][i][1]} {0.0}\n")
        file.write("End Nodes\n\n")

    def reorder_nodes_for_kratos(self, nodes_ids, nodes_coordinates):
        bottomest_leftest_node = np.argmin(nodes_coordinates[:, 1])
        bottomest_leftest_node_id = nodes_ids[bottomest_leftest_node]
        # get the coordinates of the bottomest rightest node

    def write_elements(self, mesh, file):
        counter_material = 1
        for meshed_layer, geometry in mesh['physical_groups'].items():
            # TODO the material id should be set in a better way not so hardcoded
            surface = geometry[list(geometry.keys())[0]]
            file.write(f"Begin Elements {GMSH_TO_KRATOS_ELEMENTS[list(geometry.keys())[0]]}\n")
            for counter, element in enumerate(surface['element_ids']):
                nodes = surface['connectivities'][counter]
                line = f"  {int(element)}  {counter_material}  {' '.join(str(n) for n in nodes)}"
                file.write(line + "\n")
            counter_material += 1
            file.write("End Elements\n\n")


    def write_materials(self, material_names: Dict, surface_names: List[str], physical_groups, file):
        #TODO exent with more materials this is only for soils now

        # group surfaces with the same material
        material_groups = {}
        for counter, material in enumerate(material_names['material_per_surface']):
            if material not in material_groups.keys():
                material_groups[material] = [material_names['surfaces'][counter]]
            else:
                material_groups[material].append(material_names['surfaces'][counter])

        for counter, material in enumerate(material_groups):
            file.write(f"Begin SubModelPart {material}\n")
            file.write(f"  Begin SubModelPartTables\n")
            file.write(f"  End SubModelPartTables\n")
            elements = []
            nodes = []
            # get physical group
            for surface in material_groups[material]:

                physical_group = physical_groups[surface][list(physical_groups[surface_names[counter]].keys())[0]]
                # get all the elements
                elements += list(physical_group['element_ids'])
                # get all the nodes
                nodes += list(physical_group['connectivities'].flatten())
            # write the nodes
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


    def write_gravity(self, file):
        # get all the nodes
        nodes = self.gmsh_dict["mesh_data"]["nodes"]["ids"]
        # get all the elements
        elements = []
        for key, value in self.gmsh_dict['mesh_data']['physical_groups'].items():
            elements.append(list(value[list(value.keys())[0]]['element_ids']))
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

    def get_bottom_boundary_nodes(self):
        node_coordinates_y = self.gmsh_dict["mesh_data"]["nodes"]["coordinates"].T[1]
        min_y = min(node_coordinates_y)
        mask = np.ma.masked_equal(node_coordinates_y, min_y)
        return self.gmsh_dict["mesh_data"]["nodes"]['ids'][np.where(mask.mask)[0]]
    def get_top_boundary_nodes(self):
        node_coordinates_y = self.gmsh_dict["mesh_data"]["nodes"]["coordinates"].T[1]
        min_y = max(node_coordinates_y)
        mask = np.ma.masked_equal(node_coordinates_y, min_y)
        return self.gmsh_dict["mesh_data"]["nodes"]['ids'][np.where(mask.mask)[0]]


    def get_side_nodes(self):
        node_coordinates_x = self.gmsh_dict["mesh_data"]["nodes"]["coordinates"].T[0]
        max_x = max(node_coordinates_x)
        mask_max = np.ma.masked_equal(node_coordinates_x, max_x)
        min_x = min(node_coordinates_x)
        mask_min = np.ma.masked_equal(node_coordinates_x, min_x)
        # merge masks
        mask = np.ma.mask_or(mask_max.mask, mask_min.mask)
        return self.gmsh_dict["mesh_data"]["nodes"]['ids'][mask]

    def write_boundary_condition(self, file, nodes, condition_name, elements=[]):
        file.write(f"Begin SubModelPart {condition_name}\n")
        file.write(f"  Begin SubModelPartTables\n")
        file.write(f"  End SubModelPartTables\n")
        file.write(f"  Begin SubModelPartNodes\n")
        for node in nodes:
            file.write(f"  {node}\n")
        file.write(f"  End SubModelPartNodes\n")
        file.write(f"  Begin SubModelPartElements\n")
        for element in elements:
            file.write(f"  {element}\n")
        file.write(f"  End SubModelPartElements\n")
        file.write(f"  Begin SubModelPartConditions\n")
        file.write(f"  End SubModelPartConditions\n")
        file.write(f"End SubModelPart\n\n")

    def write_load_condition(self, file, nodes, condition_name, elements=[]):
        file.write(f"Begin SubModelPart {condition_name}\n")
        file.write(f"  Begin SubModelPartTables\n")
        file.write(f"  End SubModelPartTables\n")
        file.write(f"  Begin SubModelPartNodes\n")
        for node in nodes:
            file.write(f"  {node}\n")
        file.write(f"  End SubModelPartNodes\n")
        file.write(f"  Begin SubModelPartElements\n")
        file.write(f"  End SubModelPartElements\n")
        file.write(f"  Begin SubModelPartConditions\n")
        file.write(f"  End SubModelPartConditions\n")
        file.write(f"End SubModelPart\n\n")


    def write_bottom_boundary_conditions(self, file):
        node_idx = self.get_bottom_boundary_nodes()
        self.write_boundary_condition(file, node_idx, "bottom_disp")

    def write_side_boundary_conditions(self, file):
        node_idx = self.get_side_nodes()
        self.write_boundary_condition(file, node_idx, "side_disp")

    def write_top_load(self, file):
        node_idx = self.get_top_boundary_nodes()
        self.write_boundary_condition(file, node_idx, "dike_load")

    def write_boundary_conditions(self, file, extra_conditions=None, top_load=False):
        self.write_bottom_boundary_conditions(file)
        self.write_side_boundary_conditions(file)
        if top_load:
            self.write_top_load(file)
        if extra_conditions is not None:
            for counter, surface in enumerate(extra_conditions['surfaces']):
                # get elements in surface
                elements = self.gmsh_dict['mesh_data']['physical_groups'][surface][self.default_element_type]['element_ids']
                # get nodes in surface
                nodes = self.gmsh_dict['mesh_data']['physical_groups'][surface][self.default_element_type]['connectivities'].flatten()
                nodes = np.unique(nodes)
                nodes.sort()
                self.write_boundary_condition(file, nodes, extra_conditions['names'][counter], elements)

    def write_output_processes(self, file):
        # in this case we only have one output process which is applied to all the nodes
        file.write(f"Begin SubModelPart OutputProcess\n")
        file.write(f"  Begin SubModelPartTables\n")
        file.write(f"  End SubModelPartTables\n")
        file.write(f"  Begin SubModelPartNodes\n")
        for node in self.gmsh_dict["mesh_data"]["nodes"]["ids"]:
            file.write(f"  {node}\n")
        file.write(f"  End SubModelPartNodes\n")
        file.write(f"  Begin SubModelPartElements\n")
        file.write(f"  End SubModelPartElements\n")
        file.write(f"  Begin SubModelPartConditions\n")
        file.write(f"  End SubModelPartConditions\n")
        file.write(f"End SubModelPart\n\n")


