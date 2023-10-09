from stem.IO.kratos_material_io import KratosMaterialIO
from stem.soil_material import *
from stem.structural_material import *


class MaterialGlobalDefinition:

    def __int__(self, soil_material: SoilMaterial, assigned_surface: str):
        self.soil_material = soil_material
        self.assigned_surface = assigned_surface


