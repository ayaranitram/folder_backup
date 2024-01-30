import hashlib
from os.path import isfile, isdir, exists
from multiprocessing import Pool

__all__ = ['get_md5', 'multiprocessing_md5']


def get_md5(fullpath: str) :
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
    elif isdir(fullpath):
        digest = 'folder'
    elif not exists(fullpath):
        return None
    else:
        digest = 'NaN'
    return digest


def get_path_md5(fullpath: str):
    return {fullpath: get_md5(fullpath)}


def multiprocessing_md5(list_of_fullpath: list, n_jobs=None):
    with Pool(n_jobs) as pool:
        digest_dict = pool.map(get_md5, list_of_fullpath)
    return digest_dict
