# Visualizer script
# WNixalo - 2017-Nov-09 21:37

def visualize(float_matrix=None, shape=None):
    """
        Pars: float_matrix: (ndarray or listolists) of floats.
                     shape: shape to resize ndarray to, if needed.
        Visualizes a matrix of floats by normalizing, and scaling to
        256-grayscale. For Statoil Iceberg Competition.
    """
    if float_matrix == None:
        print("Must enter a matrix")
        return

    import numpy as np
    import matplotlib.pyplot as plt

    if shape != None:
        float_matrix = np.reshape(float_matrix, shape)

    # try:
    #     mat = np.copy(float_matrix)
    # except NameError:
    #     import numpy as np
    #     mat = np.copy(float_matrix)

    mat = np.copy(float_matrix)

    mat = mat - np.min(mat) if np.min(mat) < 0 else mat
    mat = np.round(255 * mat / np.max(mat)).astype(int)

    plt.imshow(mat, cmap='gray')

    # try:
    #     plt.imshow(mat, cmap='gray')
    # except NameError:
    #     print("Matplotlib not imported: importing. NOTE: if image doesn't appear,"
    #           " enter `%matplotlib inline` in Jupyter Notebook")
    #     import matplotlib.pyplot as plt
    #     plt.imshow(mat, cmap='gray')


# NOTE: module imports in the program calling this module do not carry into
#       this one's namespace.. so the exceptions get triggered every run.
#       May tidy that when I learn more about this kind of Python.
