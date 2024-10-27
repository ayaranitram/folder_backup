import hashlib
import os.path
from glob import iglob
from fnmatch import fnmatch
from os.path import exists, isdir, isfile, getsize

def nas_get_md5(fullpath: str, verbose: bool=True, recall_md5:bool=None) :
    """
    receives a path as string and evaluate MD5 using hashlib
    returns a string of the MD5 hexadecimal value of that file.
    """
    md5_hash = hashlib.md5()
    md5_file = f"{fullpath}.md5"
    if recall_md5 and exists(md5_file):
        return None
    else:
        if isfile(fullpath):
            with open(fullpath, "rb") as file:
                try:
                    content = file.read()
                    md5_hash.update(content)
                    digest = md5_hash.hexdigest()
                    if verbose:
                        print(f"MD5 {digest}: file {fullpath}")
                except:
                    digest = 'NaN'
                    print(f"hash failed! MD5 {digest}: {fullpath}")
            if recall_md5 in (True, None) and digest != 'NaN':
                with open(md5_file, 'w') as f:
                    f.write(digest)
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

def md5_on_nas(path, md5: bool=True, exclude=None, recursive: bool=True, verbose: bool=False, recall_md5: bool=True,
               max_file_size_mb:float=512):
    print("listing files...")
    if type(path) is str:
        path = [path]
    path = [each.replace('\\', '/') for each in path]
    path = [f"{each}{'' if each.endswith('/') else '/'}{'**/' * recursive}*" if isdir(each) else each
            for each in path]
    if exclude is not None:
        if type(exclude) is str:
            exclude = [exclude]

    for eachpath in path:
        print(f"calaculating MD5 in {eachpath}")
        for fullpath in iglob(eachpath, recursive=recursive):
            # fullpath = [filepath for list_of_paths in fullpath for filepath in list_of_paths]
            if fullpath.endswith('.md5'):  # ignore .md5 files
                continue
            fullpath = fullpath.replace('\\', '/')
            if '/_folder_backup_/' in  fullpath:  # ignore log folders from this script
                continue
            if '/@eaDir/' in fullpath or fullpath.endswith('/@eaDir'):
                continue
            if (getsize(fullpath) / 1024 / 1024) > max_file_size_mb:
                print(f"skipping {fullpath}")
                continue

            continue_ = False
            if exclude is not None:
                exclude_ = list(exclude)
                while len(exclude_) > 0:
                    pat = exclude_.pop()
                    if fnmatch(fullpath, pat):
                        continue_ = True
                        break
            if continue_:
                print(f"excluded file {fullpath}")
                continue

            _ = nas_get_md5(fullpath, verbose=verbose, recall_md5=recall_md5)


if __name__ == '__main__':
    from os import getcwd
    md5_on_nas(getcwd(), md5=True, recall_md5=True, verbose=False)