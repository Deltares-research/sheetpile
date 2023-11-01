# BEGIN: 7f7d8z5v5f3c
import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

if __name__ == '__main__':
    directory = f'output/1'
    # open pickle file
    model_file = os.path.join(directory, "test.p")
    test_parameters = pickle.load(open(model_file, "rb"))
    x = test_parameters[0]
    y = test_parameters[1]
    phi = test_parameters[2]
    youngs_modulus = test_parameters[3]
    connectivity = np.array(test_parameters[4])

    values = phi

    colormap = plt.cm.get_cmap('Greys')
    sm = plt.cm.ScalarMappable(cmap=colormap)
    sm.set_clim(vmin=min(values), vmax=max(values))
    trianges = connectivity - 1
    triang = mtri.Triangulation(x, y, trianges)
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    tcf = ax1.tripcolor(triang, values, cmap=colormap)
    # add colorbar with min and max values
    cbar = fig1.colorbar(sm, ax=ax1)
    # turn off the axis
    ax1.axis('off')
    plt.show()
    plt.close()
    # close all
    plt.close('all')
# END: 7f7d8z5v5f3c
