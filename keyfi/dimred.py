import numpy as np
import pandas as pd
import pyvista as pv

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Sequence, Tuple, Type, Callable


def import_csv_data(path: str = '') -> pd.DataFrame:
    '''
    Creates a pandas dataframe from path to a csv data file.
    '''
    if not path:
        path = input('Enter the path of your csv data file: ')
    return pd.read_csv(path)


def import_vtk_data(path: str = '') -> pd.DataFrame:
    '''
    Creates a pandas dataframe from path to a vtk data file.
    Also returns mesh pyvista object.
    '''
    if not path:
        path = input('Enter the path of your vtk data file: ')

    mesh = pv.read(path)

    vector_names = []

    # Detect which variables are vectors
    for var_name in mesh.array_names:
        if np.size(mesh.get_array(var_name)) != mesh.n_points:
            vector_names.append(var_name)

    # Make a dataframe from only scalar mesh arrays (i.e. exclude vectors)
    var_names = [name for name in mesh.array_names if name not in vector_names]
    var_arrays = np.transpose([mesh.get_array(var_name) for var_name in var_names])

    if var_names:
        df = pd.DataFrame(var_arrays, columns=var_names)
    else:
        df = pd.DataFrame(index=range(mesh.n_points))

    # Add the vectors back with one row per component
    for vector_name in vector_names:
        # Get dimension of data e.g., 1D or 2D
        data_dim = mesh.get_array(vector_name).ndim

        if data_dim == 1:
            pass
        else:
            # Get dimension (number of columns) of typical vector
            dim = mesh.get_array(vector_name).shape[1]
            # split data using dim insteady of hard coding
            df[[vector_name + ':' + str(i) for i in range(dim)]] = mesh.get_array(vector_name)

    return df, mesh


def export_vtk_data(mesh: Type, path: str = '', cluster_labels: np.ndarray = None):
    '''
    Exports vtk file with mesh. If cluster labels are passed it
    will include them in a new variable
    '''
    if cluster_labels is not None:
        mesh['clusters'] = cluster_labels
    mesh.save(path)

def clean_data(data: pd.DataFrame, dim: int = 2, vars_to_drop: Sequence[str] = None, vars_to_keep: Sequence[str] = None) -> pd.DataFrame:
    '''
    Removes ghost cells (if present) and other data columns that
    are not relevant for the dimensionality reduction (i.e. spatial
    coordinates) from the original data.
    '''
    if dim not in [2, 3]:
        raise ValueError(
            'dim can only be 2 or 3. Use 2 for 2D-plane data and 3 for 3D-volume data')

    cols_to_drop = []

    if 'Points:0' in data.columns:
        cols_to_drop.append(['Points:0', 'Points:1', 'Points:2'])

    if 'vtkGhostType' in data.columns:
        data.drop(data[data.vtkGhostType == 2].index, inplace=True)
        cols_to_drop.append('vtkGhostType')

    if vars_to_keep is not None:
        # Return cleaned data based on preferred var
        cleaned_data = data[["{}".format(var) for var in vars_to_keep]]
    else:
        # drop undesired variables based on 'dim' and 'var_to_drop'
        if 'U:0' in data.columns and dim == 2:
            cols_to_drop.append('U:2')

        if vars_to_drop is not None:
            cols_to_drop.extend(vars_to_drop)

        cleaned_data = data.drop(columns=cols_to_drop, axis=1)
        cleaned_data.reset_index(drop=True, inplace=True)

    return cleaned_data


def scale_data(
    data: pd.DataFrame,
    features: [[str, ...], ...],
    scalers: [Callable, ...]
    ) -> pd.DataFrame():
    """
    data is entire dataframe with data to be scaled.
    features should be a sequence of columns and sequences of columns in the
    dataframe which are to be scaled.
    The same scaler object will be applied to each column in each
    value in features.
    scalers has to be the same length as features. the nth sequence or column in
    features will be scaled using the nth scaler.
    Each element in scalers has to be a callable object which returns the
    scaler object itself.
    Scalers must have .fit(data) and .transform(data) methods.
    You can for example pass sklearn.preprocessing.StandardScaler
    but NOT sklearn.preprocessing.StandardScaler().
    Not sure why I made this choice, but that is how it is now.
    If features are ommitted from they will not be scaled,
    but still returned.
    """

    if len(features) != len(scalers):
        raise ValueError(("scale_data() arguments features and scalers must be of equal length"))

    scaled_data = data.copy()

    #for backwards compatibility with python 3.8.8 zip(strict=True) is not used
    #instead ValueError is raised above if features and scalers are of
    #unequal length
    for features, scaler in zip(features, scalers):
        scaler = scaler()

        if isinstance(features, str):
            # fit_transform is not used because scalers have to implement fit
            # and transform anyway for the other case, so requiring
            # fit_transform is unnecessary
            values = data[features].values.reshape(-1, 1)
            scaler.fit(values)
            scaled_data[features] = scaler.transform(values)
        elif len(features) == 1:
            #for compatibility with tuples, lists etc
            values = data[features[0]].values.reshape(-1, 1)
            scaler.fit(values)
            scaled_data[features[0]] = scaler.transform(values)
        else:
            #compatibility with tuples, lists etc
            features = [feature for feature in features]

            scaler.fit(data[features].values.reshape(-1, 1))
            for feature in features:
                scaled_data[feature] = scaler.transform(
                    data[feature].values.reshape(-1,1)
                    )

    return scaled_data

def embed_data(data: pd.DataFrame, algorithm, scale: bool = True, **params) -> Tuple[np.ndarray, Type]:
    '''
    Applies either UMAP or t-SNE dimensionality reduction algorithm
    to the input data and returns the embedding array.
    Also accepts specific and optional algorithm parameters.

    The scale parameter exists for backwards compatibility,
    the scale_data function can be used before embed_data with
    scale set to False for more options.
    '''
    algorithms = [UMAP, TSNE]
    if algorithm not in algorithms:
        raise ValueError(
            'invalid algorithm. Expected one of: %s' % algorithms)

    if scale:
        data = StandardScaler().fit_transform(data)

    reducer = algorithm(**params)

    if algorithm == UMAP:
        mapper = reducer.fit(data)
        embedding = mapper.transform(data)
    elif algorithm == TSNE:
        mapper = None
        embedding = reducer.fit_transform(data)

    return embedding, mapper
