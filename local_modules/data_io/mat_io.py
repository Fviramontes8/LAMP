import scipy.io as scio

"""
@brief A helper function to read the .mat files in the Data/ folder.
    Typically when the file is read, it is in the form of a dictionary 
    with the following keys:
        ['__header__', '__version__', '__globals__', 'sample_CXRTL3']
    The key 'sample_CXRTL3' contains the interesting data. The 'CX' part
    describes the configuration. The X can be 1, 2, 3, or 4. The 'RTL3" means
    that the data is coming from relay 3.

@param path -- A string describing the absolute or relative path to 
    the .mat file.

@returns A dictionary containing the contents of the .mat file.
"""
def open_mat_file(path: str) -> dict:
    return scio.loadmat(path);



