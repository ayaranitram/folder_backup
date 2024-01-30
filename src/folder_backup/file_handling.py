from os.path import isfile, exists
from os import remove, rename, makedirs, rmdir, removedirs, listdir
from shutil import copy2
from stringthings import extension
from .md5 import get_md5


__all__ = ['copy3', 'cp', 'mv', 'rm']

def new_name(path):
    i = 1
    new_path = path
    ext, file, folder, _ = extension(path)
    while exists(new_path) and i < 10000:
        new_path = f"{folder}{file} {i}{ext}"
    return new_path


def copy3(src, dst, *, follow_symlinks=True):
    """
    Wrapper of the Python shutil.copy2() but copy3() will create any folder required to copy the file.

    *The follwing lines are taken and modified from shutil.copy() and shutil.copy2():*

    Copies the file src to the file or directory dst. src and dst should be path-like objects or strings. If dst specifies a directory, the file will be copied into dst using the base filename from src. If dst specifies a file that already exists, it will be replaced. Returns the path to the newly created file.
    If follow_symlinks is false, and src is a symbolic link, dst will be created as a symbolic link. If follow_symlinks is true and src is a symbolic link, dst will be a copy of the file src refers to.
    copy3() copies the file data and the file’s permission mode (see os.chmod()), like copy() does, and attempts to preserve file metadata as copy2().
    """
    dst_folder = extension(dst)[2]
    makedirs(dst_folder, exist_ok=True)
    return copy2(src, dst, follow_symlinks=follow_symlinks)


def rm(path, attempts=3):
    i = 0
    while exists(path) and i < attempts:
        remove(path)
        i += 1
    return f"{exists(path) * 'not '}deleted"


def cp(source, destination, md5=True, source_md5=None, if_exists='both', attempts=3, *,
       return_details=False):
    def _return():
        if return_details:
            return output
        else:
            return output['msg']

    if if_exists not in ['stop', 'both', 'overwrite']:
        raise ValueError(f"`if_exists` must one of the strings: 'stop', 'both' or 'overwrite', not {if_exists}.")

    if md5 and source_md5 is None:
        source_md5 = get_md5(source)

    if return_details:
        output = {'source': source,
                  'destination': destination,
                  'source_md5': source_md5,
                  'destination_md5': None,
                  'successful': None,
                  'msg': '',
                  }
    else:
        output = {'msg': ''}

    if exists(destination):
        if if_exists == 'stop':
            if md5:
                destination_md5 = get_md5(destination)
                if source_md5 == destination_md5:
                    output['msg'] = f"file already in destination (md5: {source_md5})"
                    output['successful'] = True
                else:
                    output['msg'] = f"other file exists in destination (md5: {destination_md5})"
                    output['successful'] = False
                output['destination_md5'] = destination_md5
            else:
                output['msg'] = f"other file exists in destination"
                output['successful'] = False
            return _return()

        elif if_exists == 'overwrite':
            output['msg'] = f"existing file in destination '{extension(destination)[1]}{extension(destination)[0]}' rm(destination)"
            pass

        elif if_exists == 'both':
            if md5:
                destination_md5 = get_md5(destination)
                if source_md5 == destination_md5:
                    output['msg'] = f"file already in destination (md5: {source_md5})"
                    output['destination_md5'] = destination_md5
                    output['successful'] = True
                    return _return()
                else:
                    new_destination = new_name(destination)
            else:
                new_destination = new_name(destination)
                destination = new_destination
            output['msg'] = f", file renamed from '{extension(destination)[1]}{extension(destination)[0]}' to '{extension(new_destination)[1]}{extension(new_destination)[0]}'"
            output['destination'] = new_destination

    if md5:
        destination_md5 = ''
        i = 0
        while destination_md5 != source_md5 and i < attempts:
            _ = copy3(source, destination)
            destination_md5 = get_md5(destination)
            i += 1
        if not exists(destination):
            output['msg'] = f"failed to copy{output['msg']}"
            output['successful'] = False
        elif source_md5 == destination_md5:
            output['msg'] = f"successfully copied (md5: {source_md5}){output['msg']}"
            output['successful'] = True
        else:
            output['msg'] = f"copy doesn't match (md5: {source_md5} != {destination_md5}){output['msg']}"
            output['successful'] = False
    else:
        i = 0
        while not exists(destination) and i < attempts:
            _ = copy3(source, destination)
            i += 1
        if not exists(destination):
            output['msg'] = f"failed to copy{output['msg']}"
            output['successful'] = False
        else:
            output['msg'] = f"copied{output['msg']}"
            output['successful'] = True
    return _return()


def mv(source, destination, md5=True, source_md5=None, if_exists='both', attempts=3):
    if md5 and source_md5 is None:
        source_md5 = get_md5(source)

    if ':' in source and ':' in destination \
            and source[:source.index(':')] != destination[:destination.index(':')]:
        if md5:
            destination_md5 = None
            i = 0
            while source_md5 != destination_md5 and i < attempts:
                output = cp(source, destination, md5=md5, source_md5=source_md5, if_exists=if_exists, attempts=attempts,
                            return_details=True)
                if output['destination_md5'] is None:
                    destination_md5 = get_md5(destination)
                else:
                    destination_md5 = output['destination_md5']

            if source_md5 is None:
                source_md5 = get_md5(source)
            if destination_md5 is None:
                destination_md5 = get_md5(destination)
            if source_md5 == destination_md5:
                return f"source and destination in different systems, {output['msg']}, source file {rm(source)}"
            else:
                return f"source and destination in different systems, failed to copy, {output['msg']}"
        else:
            i = 0
            successful = False
            while not successful and i < attempts:
                output = cp(source, destination, md5=md5, source_md5=source_md5, if_exists=if_exists, attempts=attempts,
                            return_details=True)
                successful = output['successful']
            if successful:
                return f"source and destination in different systems, {output['msg']}, source file {rm(source)}"
            else:
                return f"source and destination in different systems, failed to copy, {output['msg']}"

    # else:  # in the same system
    if exists(destination):
        if if_exists == 'stop':
            if md5:
                destination_md5 = get_md5(destination)
                if source_md5 == destination_md5:
                    return f"file already in destination (md5: {source_md5}), source file {rm(source)}"
                else:
                    return f"other file exists in destination (md5: {destination_md5})"
            else:
                return f"other file exists in destination"
        elif if_exists == 'overwrite':
            pass
        elif if_exists == 'both':
            if md5:
                destination_md5 = get_md5(destination)
                if source_md5 == destination_md5:
                    return f"file already in destination (md5: {source_md5}), source file {rm(source)}"
                else:
                    new_destination = new_name(destination)
            else:
                new_destination = new_name(destination)
            destination_folder = extension(new_destination)[2]
            makedirs(destination_folder, exist_ok=True)
            rename(source, new_destination)
            return f"moved, file renamed from '{extension(destination)[1]}{extension(destination)[0]}' to '{extension(new_destination)[1]}{extension(new_destination)[0]}'"

    destination_folder = extension(destination)[2]
    makedirs(destination_folder, exist_ok=True)
    i = 0
    while exists(source) and i < attempts:
        rename(source, destination)
        i += 1
    return f"{exists(source) * 'not '}moved"
