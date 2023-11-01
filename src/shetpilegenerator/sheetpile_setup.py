from stem.model_part import BodyModelPart


SHEETPILE_MATERIALS = {
    "YOUNG_MODULUS": 100000, 
    "DENSITY": 5, 
    "CROSS_AREA": 1, 
    "I33": 200, 
    "POISSON_RATIO": 0.3
}


def add_line_by_coordinates(model, coordinates,material_parameters, name):
    """
    Adds a soil layer to the model by giving a sequence of 2D coordinates. In 3D the 2D geometry is extruded in
    the direction of the extrusion_length

    Args:
        - coordinates (Sequence[Sequence[float]]): The plane coordinates of the soil layer.
        - material_parameters (Union[:class:`stem.soil_material.SoilMaterial`, \
            :class:`stem.structural_material.StructuralMaterial`]): The material parameters of the soil layer.
        - name (str): The name of the soil layer.

    Raises:
        - ValueError: if extrusion_length is not specified in 3D.
    """


    gmsh_input = {name: {"coordinates": coordinates, "ndim": 1}}
    # check if extrusion length is specified in 3D
    if model.ndim == 3:
        if model.extrusion_length is None:
            raise ValueError("Extrusion length must be specified for 3D models")

        gmsh_input[name]["extrusion_length"] = model.extrusion_length

    # todo check if this function in gmsh io can be improved
    model.gmsh_io.generate_geometry(gmsh_input, "")

    # create body model part
    body_model_part = BodyModelPart(name)
    body_model_part.material = material_parameters

    # set the geometry of the body model part
    body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, name)

    model.body_model_parts.append(body_model_part)