import hashlib
from os.path import isfile, isdir, exists
from multiprocessing import Pool

__all__ = ['get_md5', 'multiprocessing_md5']


def get_md5(fullpath: str, verbose: bool=True) :
    """
    receives a path as string and evaluate MD5 using hashlib
    returns a string of the MD5 hexadecimal value of that file.
    """
    md5_hash = hashlib.md5()
    if isfile(fullpath):
        with open(fullpath, "rb") as file:
            content = file.read()
            md5_hash.update(content)
            digest = md5_hash.hexdigest()
        if verbose:
            print(f"MD5 {digest}: file {fullpath}")
    elif isdir(fullpath):
        digest = 'folder'
        if verbose:
            print(f"MD5 {digest}: {fullpath}")
    elif not exists(fullpath):
        if verbose:
            print(f"doesn't exists: {fullpath}")
        return None
    else:
        digest = 'NaN'
        if verbose:
            print(f"something missing! MD5 {digest}: {fullpath}")
    return digest


def get_path_md5(fullpath: str, verbose: bool=True):
    return {fullpath: get_md5(fullpath, verbose=verbose)}


def multiprocessing_md5(list_of_fullpath: list, n_jobs=None, verbose: bool=True):
    get_md5_ = lambda fullpath: get_md5(fullpath, verbose=verbose)
    with Pool(n_jobs) as pool:
        digest_dict = pool.map(get_md5_, list_of_fullpath)
    return digest_dict
