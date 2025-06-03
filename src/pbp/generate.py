import numpy as np
import logging
from bitarray import bitarray
from bitarray.util import ba2int, int2ba
import pandas as pd

BIT_ORDER="little"
logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG )
logger = logging.getLogger()
sub_s = "0123456789" # substring to use for the variable names for up to 9 variables. Useful for debugging.

# Function to encode the binary variables
def bin_encoder(v, l):
    """
    Encode a Boolean variable into an integer.

    Args:
        v (int): The Boolean variable to encode.
        l (int): The length of the binary representation.
    Returns:
        int: The encoded integer.
    """
    y = bitarray(l, endian=BIT_ORDER)
    y[v] = 1
    return ba2int(y)
# Function to create П matrix
def create_perm(C: np.array):
    """
    Generate a permutation indicator matrix based on the column-wise sorting of a given numpy array.

    Parameters:
        C (numpy.ndarray): The input numpy array.

    Returns:
        numpy.ndarray: The permutation indicator matrix.

    """
    perm = C.argsort(kind='quick', axis=0)
    return perm


def calculate_degree(y):
    """
    Calculate the degree of a Boolean variable.

    Args:
        y (int): The Boolean variable to calculate the degree of.

    Returns:
        int: The degree of the Boolean variable.
    """
    y_bin = int2ba(y, endian=BIT_ORDER)
    degree = y_bin.count()
    return degree
# Function to create the coefficients using the П matrix
def create_coeffs_matrix(C: np.array, perm: np.array):
    """
    Create a coefficients matrix based on the given matrix C and permutation indicator matrix perm.

    Args:
        C (numpy.ndarray): The input matrix C.
        perm (numpy.ndarray): The permutation indicator matrix.

    Returns:
        numpy.ndarray: The coefficients matrix.

    This function takes a matrix C and a permutation indicator matrix perm as input. It sorts the columns of C based on the permutation indicator matrix and creates a coefficients matrix. The coefficients matrix is obtained by subtracting the sorted matrix from its transpose. The last row and column of the matrix are removed. The function returns the resulting coefficients matrix.

    Example:
        >>> C = np.array([[7, 8, 2, 10, 3], [4, 12, 1, 8, 4], [5, 3, 0, 6, 9], [9, 6, 7, 1, 5]])
        >>> perm = np.array([[1, 2, 2, 3, 0],
                        [2, 3, 1, 2, 1],
                        [0, 0, 0, 1, 3],
                        [3, 1, 3, 0, 2]])
        >>> create_coeffs_matrix(C, perm)
        np.array([[4, 3, 0, 1, 3],
                [1, 3, 1, 5, 1],
                [2, 2, 1, 2, 1],
                [2, 4, 5, 2, 4]])
    """
    sorted_c = np.take_along_axis(C, perm, axis=0)
    zs = np.zeros((sorted_c.shape[1],), dtype=sorted_c.dtype)
    coeffs_c = np.vstack((sorted_c, zs)) - np.vstack((zs, sorted_c))
    coeffs_c = coeffs_c[:-1]
    return coeffs_c
# Function to create the Boolean variables
def create_variable_matrix(C: np.array, perm: np.array):
    """
    Create a variable matrix based on the given matrix C and permutation indicator matrix perm.

    Args:
        C (numpy.ndarray): The input matrix C.
        perm (numpy.ndarray): The permutation indicator matrix.

    Returns:
        numpy.ndarray: The variable matrix.

    This function takes a matrix C and a permutation indicator matrix perm as input. It generates a variable matrix by applying the bin_encoder function to the elements of the perm array. The bin_encoder function encodes each element of the perm array into a Boolean variable. The resulting Boolean variables are then cumulatively summed along the rows of the perm array to generate the variable matrix.

    Example:
        >>> C = np.array([[7, 8, 2, 10, 3], [4, 12, 1, 8, 4], [5, 3, 0, 6, 9], [9, 6, 7, 1, 5]])
        >>> perm = np.array([[1, 2, 2, 3, 0],
                    [2, 3, 1, 2, 1],
                    [0, 0, 0, 1, 3],
                    [3, 1, 3, 0, 2]])
        >>> create_variable_matrix(C, perm)
        np.array([[2, 4, 4, 8, 1],
                [6, 12, 6, 12, 3],
                [7, 13, 7, 14, 11]])
    """
    y = perm[:-1]
    y_maker = np.frompyfunc(bin_encoder, 2, 1)
    y = y_maker(y, perm.shape[0])
    y = y.cumsum(axis=0)
    return y
# Function to calculate the term's degree
def calculate_degree(y):
    """
    Calculate the degree of a Boolean variable.

    Args:
        y (int): The Boolean variable to calculate the degree for.

    Returns:
        int: The degree of the Boolean variable.
    """
    if isinstance(y, bitarray):
        y_bin = y
    else:
        y_bin = int2ba(y, endian=BIT_ORDER)
    degree = y_bin.count()
    return degree
# Function to reduce the polynomial
def reduce_pbp_pandas(coeffs: np.array, variables: np.array, w_str=False):
    """
    Reduces a polynomial basis representation using pandas DataFrame operations.

    Args:
        coeffs (numpy.ndarray): An array of coefficients representing the polynomial basis.
        variables (numpy.ndarray): An array of variables representing the polynomial basis.

    Returns:
        pandas.DataFrame: A DataFrame containing the reduced polynomial basis representation.
            The DataFrame has the following columns:
            - 'y': The encoded variables.
            - 'coeffs': The sum of coefficients corresponding to each variable.
            - 'y_str': The decoded variables.
            - 'degree': The degree of each variable.

    This function takes a polynomial basis representation represented by arrays of coefficients and variables as input.
    It creates a pandas DataFrame by stacking the coefficients and variables arrays and grouping the DataFrame by the variables.
    The resulting DataFrame is then filtered to remove rows with zero coefficients.
    The variables are decoded using the `decode_var` function and the degree of each variable is calculated using the `calculate_degree` function.
    The DataFrame is sorted by degree and the index is reset to a range index.
    The resulting DataFrame is returned.
    """
    zero_vars = np.zeros((1, variables.shape[1]), dtype=int)
    var_flat = np.vstack([zero_vars, variables]).ravel()
    df = pd.DataFrame()
    df["y"] = var_flat
    df["coeffs"] = coeffs.ravel()
    df = df.groupby(['y'], as_index=False).agg({'y': 'first', 'coeffs': 'sum' })
    zero_coeffs = df["coeffs"] == 0
    df = df.loc[~zero_coeffs]

    
    if w_str:
        df["y_str"] = df["y"].apply(decode_var)

    df["degree"] = df["y"].apply(calculate_degree)
    df.sort_values(by=['degree'], inplace=True)
    blankIndex=[''] * len(df)
    df.index=blankIndex
    return df
# Function to decode Boolean variables
def decode_var(y):
    """
    Decode a variable from its binary representation.

    Parameters:
        y (int): The binary representation of the variable.

    Returns:
        str: The decoded variable as a string.

    This function takes a binary representation of a variable and decodes it into a string. The binary representation is first converted to a bitarray using the `int2ba` function. The bitarray is then converted to a numpy array of boolean values using the `unpack` method. The indices of the non-zero values in the numpy array are obtained using the `nonzero` function. If there are no non-zero values, an empty string is returned. Otherwise, the decoded variable is constructed by concatenating the letter "y" with the corresponding subscript from the `sub_s` list.

    Example:
        >>> decode_var(10)
        'y2y4'
    """
    try:
        if isinstance(y, list):
            y_arr = np.array(y)
        else:
            y_arr = np.array(y.tolist())
    except:
        bin_indices = int2ba(y, endian=BIT_ORDER)
        y_arr  = np.frombuffer(bin_indices.unpack(), dtype=bool)

    
    indices = np.nonzero(y_arr)[0]
    if indices.size == 0:
        return ""
    
    return ["y" + get_sub_s(i+1) for i in indices]


def get_sub_s(n):
    """
    Generate a substring representation of a number using the sub_s mapping.
    Recursively handles numbers of any size.

    Args:
        n (int): The number to convert to substring representation.

    Returns:
        str: The substring representation of the number.
    """
    if n < 10:
        return sub_s[n]
    return get_sub_s(n // 10) + get_sub_s(n % 10)
    

# Driver function to create a whole pBp
def create_pbp(c: np.array, verbose=False):
    """
    Create a polynomial basis representation (pBp) based on the given matrix C.

    Args:
        c (numpy.ndarray): The input matrix C.

    Returns:
        pandas.DataFrame: A DataFrame containing the polynomial basis representation.
            The DataFrame has the following columns:
            - 'y': The encoded variables.
            - 'coeffs': The sum of coefficients corresponding to each variable.
            - 'y_str': The decoded variables.
            - 'degree': The degree of each variable.

    This function takes a matrix C as input and generates a polynomial basis representation (pBp) based on the column-wise sorting of C.
    It first creates a permutation indicator matrix based on the column-wise sorting of C using the `create_perm` function.
    Then, it creates a coefficients matrix based on the permutation indicator matrix and the original matrix C using the `create_coeffs_matrix` function.
    Next, it creates a variable matrix based on the permutation indicator matrix and the original matrix C using the `create_variable_matrix` function.
    Finally, it reduces the polynomial basis representation using pandas DataFrame operations using the `reduce_pbp_pandas` function.
    The resulting DataFrame is returned.

    Example:
        >>> C = np.array([[7, 8, 2, 10, 3], [4, 12, 1, 8, 4], [5, 3, 0, 6, 9], [9, 6, 7, 1, 5]])
        >>> print(create_pbp(C))
        y  coeffs   y_str  degree
        0      11               0
        1       1      y1       1
        2       1      y2       1
        4       4      y3       1
        8       5      y4       1
        3       1    y1y2       2
        6       3    y2y3       2
        12       4    y3y4       2
        7       7  y1y2y3       3
        11       4  y1y2y4       3
        13       4  y1y3y4       3
        14       2  y2y3y4       3
    """
    assert len(c.shape) == 2
    perm_c = create_perm(c)
    coeffs_c = create_coeffs_matrix(c, perm_c)
    y = create_variable_matrix(c, perm_c)
    pBp = reduce_pbp_pandas(coeffs_c, y, w_str=verbose)
    return pBp


def cap_to_pbp_df(c: np.array, f: np.array, verbose: bool = False):
    f_df = pd.DataFrame({'y': np.power(2, np.arange(len(f)), dtype=int), 'coeffs': -f, 'degree': 1})
    
    pbp = create_pbp(c, verbose=verbose)
   
    pbp.loc[pbp['degree'] == 0, 'coeffs'] += np.sum(f)
    pbp = pd.concat([pbp, f_df]).groupby(['y', 'degree']).agg({'coeffs': 'sum'}).reset_index()
    pbp.sort_values(by=['degree'], inplace=True)
    
    pbp = pbp[pbp['coeffs'] != 0]
    pbp.reset_index(inplace=True)
    
    if verbose:
        pbp['y_str'] = pbp['y'].apply(decode_var)

    pbp['y'] = pbp['y'].apply(lambda l: int2ba(l, endian=BIT_ORDER))

    return pbp

# Function to truncate a pBp by a given p value
def truncate_pBp(pBp, c, p):
    """
    Truncates the polynomial basis representation (pBp) based on the given parameters.

    Parameters:
        pBp (pandas.DataFrame): The original polynomial basis representation.
        c (numpy.array): The input matrix C used for truncation.
        p (int): The cutoff value for truncation.

    Returns:
        pandas.DataFrame: The truncated polynomial basis representation.
    """
    if 'degree' not in pBp.columns:
        pBp['degree'] = pBp['y'].apply(calculate_degree)

    cutoff = c.shape[0] - p + 1
    truncated_pBp = pBp.loc[pBp['degree'] < cutoff]
    
    return truncated_pBp


def get_dep_list(num_vars, pbp: pd.DataFrame):
    terms = 2**np.arange(num_vars)
    dep_list = [np.where(y & pbp['y'])[0] for y in terms]

    return dep_list

def count_linear_terms(pbp: pd.DataFrame):
    return pbp[pbp['degree'] == 1]['coeffs'].count()
def count_quadratic_terms(pbp: pd.DataFrame):
    return pbp[pbp['degree'] == 2]['coeffs'].count()
def count_cubic_terms(pbp: pd.DataFrame):
    return pbp[pbp['degree'] == 3]['coeffs'].count()
def count_non_linear_terms(pbp: pd.DataFrame, degree: int):
    degree = max(degree, 2)
    return pbp[(pbp['degree'] > 1) & (pbp['degree'] <= degree)]['coeffs'].count()

# Examples
if __name__ == "__main__":
    f = np.array([1000, 1200, 800, 1500, 900, 1100, 1300])
    c = np.array([
        [50, 60, 40, 70, 45, 55, 65],
        [45, 55, 35, 65, 40, 50, 60],
        [60, 70, 50, 80, 55, 65, 75],
        [40, 50, 30, 60, 35, 45, 55],
        [55, 65, 45, 75, 50, 60, 70]
    ])

    cap_pbp_form = cap_to_pbp_df(c, f, verbose=True)
    print("cap_pbp_form")
    print(cap_pbp_form)

    print("B1: linear terms")
    print(count_linear_terms(cap_pbp_form))
    print("B2: quadratic terms")
    print(count_quadratic_terms(cap_pbp_form))
    print("B3: cubic terms")
    print(count_cubic_terms(cap_pbp_form))
    print("B4: non-linear terms")
    
    for degree in range(2, c.shape[0] + 1):
        print(f"Degree {degree}: {count_non_linear_terms(cap_pbp_form, degree)}")
    



