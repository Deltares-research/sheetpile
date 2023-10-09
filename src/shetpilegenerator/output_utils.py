import os
import numpy as np
from shetpilegenerator.output_process import OutputProcessJsonReader

def plot_nodal_results(x, y, values, connectivity, save=False, file_name="geometry.png", directory="."):
    # plot the results contour plot
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    colormap = plt.cm.get_cmap('Greys')
    sm = plt.cm.ScalarMappable(cmap=colormap)
    sm.set_clim(vmin=min(values), vmax=max(values))
    trianges = connectivity - 1
    triang = mtri.Triangulation(x, y, trianges)
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    tcf = ax1.tricontourf(triang, values, cmap=colormap)
    #ax1.triplot(triang, 'ko-', alpha=0)
    # add colorbar with min and max values
    #cbar = fig1.colorbar(sm, ax=ax1)
    # turn off the axis
    ax1.axis('off')
    if save:
        # save the figure if the directory exists ortherwise make the directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        # save the figure in gray scale
        plt.savefig(os.path.join(directory, file_name))
    else:
        plt.show()
    plt.close()
    # close all
    plt.close('all')


def post_process(stage_index, timestep, gmsh_to_kratos, save=False, file_name="geometry.png", directory="."):
    path_to_results = os.path.join(f"{directory}/output/json_output_{stage_index}.json")
    post_process = OutputProcessJsonReader(path_to_results)
    water_pressure, total_displacement = post_process.get_values_in_timestep(timestep)
    # get the coordinates of the nodes
    x_coordinates = [node[0] for node in gmsh_to_kratos.gmsh_io.mesh_data['nodes'].values()]
    y_coordinates = [node[1] for node in gmsh_to_kratos.gmsh_io.mesh_data['nodes'].values()]
    connectivity = np.array(list(gmsh_to_kratos.gmsh_io.mesh_data['elements']['TRIANGLE_3N'].values()))
    plot_nodal_results(x_coordinates,
                       y_coordinates,
                       water_pressure,
                       connectivity,
                       save=save,
                       file_name="water_pressure_" + file_name,
                       directory=directory)
    plot_nodal_results(x_coordinates,
                       y_coordinates,
                       total_displacement,
                       connectivity,
                       save=save,
                       file_name="total_displacement_" + file_name,
                       directory=directory)


def plot_geometry(layers, save=False, file_name="geometry.png", directory="."):
    # plot the geometry with filled in polygons
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(10, 10))
    for layer in layers:
        points_2d = [point[:2] for point in layer]
        # plot with only the boundaries shown as black lines
        ax.plot(*zip(*(points_2d + [points_2d[0]])), color='black')
    # set the limits automatically based on the geometry
    ax.autoscale()
    # turn off the axis
    ax.axis('off')
    if save:
        # save the figure in the directory and create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, file_name))
    else:
        plt.show()
    plt.close()
    # close all the figures
    plt.close('all')