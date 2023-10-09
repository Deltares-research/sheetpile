
from shetpilegenerator.IO.kratos_water_boundaries_io import KratosWaterBoundariesIO
from shetpilegenerator.water_boundaries import WaterBoundary, InterpolateLineBoundary, PhreaticLine


def define_water_boundaries(water_top_1, water_top_2, water_top_3,water_middle, water_bottom):
    # top layer boundary conditions multiline
    water_boundaries_top = []
    for counter, boundary in enumerate([water_top_1, water_top_2, water_top_3]):
        water_line_top_parameters = PhreaticLine(
            is_fixed=True,
            gravity_direction=1,
            out_of_plane_direction=2,
            specific_weight=9.81,
            value=0,
            first_reference_coordinate=water_top_1[0],
            second_reference_coordinate=water_top_1[1],
            surfaces_assigment=water_top_1[2],
        )
        water_boundaries_top.append(WaterBoundary(water_line_top_parameters, name=f"top_water_boundary_{counter + 1}"))
    # bottom layer boundary conditions multiline
    water_line_bottom_parameters = PhreaticLine(
        is_fixed=True,
        gravity_direction=1,
        out_of_plane_direction=2,
        value=0,
        first_reference_coordinate=water_bottom[0],
        second_reference_coordinate=water_bottom[1],
        specific_weight=9.81,
        surfaces_assigment=water_bottom[2],
    )
    water_boundary_bottom = WaterBoundary(water_line_bottom_parameters, name="bottom_water_boundary")
    # middle layer boundary conditions
    interpolation_type = InterpolateLineBoundary(
        is_fixed=True,
        out_of_plane_direction=2,
        gravity_direction=1,
        surfaces_assigment=water_middle,
    )
    water_boundary_interpolate = WaterBoundary(interpolation_type, name="middle_water_boundary")

    kratos_io = KratosWaterBoundariesIO(domain="PorousDomain")
    return [kratos_io.create_water_boundary_dict(boundary) for boundary in water_boundaries_top + [water_boundary_bottom, water_boundary_interpolate]]


def define_water_line_based_on_outside_head(head):
    TOP_1_WL = [[-80., head, 0.], [-20., head, 0.], ['TOP_1']]
    TOP_2_WL = [[-20., head, 0.], [25., 5.58, 0.], ['TOP_2']]
    TOP_3_WL = [[25., 2, 0.], [80., 2., 0.], ['TOP_3']]
    # water line bottom
    WL3 = [[-80., head, 0.], [80.,  5.31, 0.], ["BOTTOM"]]
    # water line middle
    WL2 = "MIDDLE"
    return TOP_1_WL, TOP_2_WL, TOP_3_WL, WL2, WL3