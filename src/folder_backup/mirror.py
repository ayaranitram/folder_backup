from stringthings import extension
from glob import glob
from .md5 import get_md5, multiprocessing_md5
from .file_handling import rm, mv, cp
import pandas as pd
import numpy as np
from fnmatch import fnmatch
from multiprocessing import Pool
from os.path import exists, isdir
import datetime


__all__ = ['run_backup', 'mirror', 'make_actions', 'execute_actions']


def listing_files(path, md5: bool=True, exclude=None, recursive: bool=True, n_jobs=None, verbose: bool=True):
    print("listing files...")
    if type(path) is str:
        path = [path]
    path = [each.replace('\\', '/') for each in path]
    path = [f"{each}{'' if each.endswith('/') else '/'}{'**/' * recursive}*" if isdir(each) else each
            for each in path]
    fullpath = [glob(each, recursive=recursive) for each in path]
    fullpath = [filepath for list_of_paths in fullpath for filepath in list_of_paths]

    if exclude is not None:
        if type(exclude) is str:
            exclude = [exclude]
        while len(exclude) > 0:
            pat = exclude.pop()
            fullpath = [each for each in fullpath if not fnmatch(each, pat)]
    if verbose:
        print(f"{len(fullpath)} files listed...")
    if len(fullpath) == 0:
        print("no files found in the path:\n" + "\n ".join(fullpath))

    if verbose:
        print("getting filename and folder...")
    filename_folder = [extension(each) for each in fullpath]
    filename = [f"{each[1]}{each[0]}" for each in filename_folder]
    folder = [each[2] for each in filename_folder]
    fullpath = [each[3] for each in filename_folder]

    if verbose:
        print("creating dataframe")
    df = pd.DataFrame(
        data={'path': fullpath,
              'folder': folder,
              'filename': filename,
              'md5': None,
              }
    )
    if md5:
        if n_jobs == 1:
            print("calculating MD5 hash")
            df['md5'] = [get_md5(each, verbose=verbose)
                         for each in df['path'].values]
        else:
            print("calculating MD5 hash")
            df['md5'] = multiprocessing_md5(df['path'].to_list(), n_jobs=n_jobs, verbose=verbose)
    return df


def calculate_missing_md5(df, path_column, md5_column, verbose: bool=True):
    return [get_md5(df.loc[i, path_column], verbose=verbose).values[0]
            if df.loc[i, md5_column].values[0] is None
            else df.loc[i, md5_column].values[0]
            for i in df.index]


def mirror(source, destination, md5: bool=True, file_pattern: str='*', exclude=None,
           recursive: bool=True, log_file=None,
           n_jobs=None, *, pass_log_folder: bool=False, report: bool=False, verbose: bool=True):

    # check input parameters
    if type(source) is str:
        source = source.replace('\\', '/')
        if not exists(source):
            raise ValueError(f"`source` directory '{source}' doesn't exist.")
        log_file = f"{source}{'' if source.endswith('/') else '/'}mirror.xlsx" if log_file is None else log_file
        source = f"{source}{'' if source.endswith('/') else '/'}{'**/' * recursive}{file_pattern}"
    elif isinstance(source, pd.DataFrame) and (
            'path' in source.columns and
            'folder' in source.columns and
            'filename' in source.columns and
            'md5' in source.columns):
        print("SOURCE data provided")
        source_df = source
    elif hasattr(source, '__iter__'):
        for each in source:
            if type(each) is not str:
                raise TypeError("`source` must be a string, list of strings, or frame with columns 'path', 'folder', 'filename' and 'md5'.")
            elif not exists(each):
                ValueError(f"`source` directory '{each}' doesn't exist.")
        source = [each.replace('\\', '/') for each in source]
        source = [f"{each}{'' if each.endswith('/') else '/'}{'**/' * recursive}{file_pattern}" for each in source]
    else:
        raise TypeError("`source` must be a string, list of strings, or frame with columns 'path', 'folder', 'filename' and 'md5'.")

    if type(destination) is str:
        destination = destination.replace('\\', '/')
        if not exists(destination):
            if ':' in destination:
                if not exists(destination[:destination.index(':')+1]):
                    raise ValueError(f"`destination` root directory '{destination[:destination.index(':')+1]}' doesn't exist.")
            elif destination.startswith('/'):
                pass  # pending to check network or linux folders
        destination = f"{destination}{'' if destination.endswith('/') else '/'}{'**/' * recursive}{file_pattern}"
    elif isinstance(destination, pd.DataFrame) and (
            'path' in destination.columns and
            'folder' in destination.columns and
            'filename' in destination.columns and
            'md5' in destination.columns):
        print("DESTINATION data provided")
        destination_df = destination
    elif hasattr(destination, '__iter__'):
        if False in [type(each) is str for each in destination]:
            raise TypeError("`destination` must be a string, list of strings, or frame with columns 'path', 'folder', 'filename' and 'md5'.")
        destination = [each.replace('\\', '/') for each in destination]
        destination = [f"{each}{'' if each.endswith('/') else '/'}{'**/' * recursive}{file_pattern}" for each in destination]
    else:
        raise TypeError("`destination` must be a string, list of strings, or frame with columns 'path', 'folder', 'filename' and 'md5'.")

    # list files if needed
    if not isinstance(source, pd.DataFrame):
        print("extracting data from SOURCE")
        source_df = listing_files(source, md5=md5, exclude=exclude, recursive=recursive, n_jobs=n_jobs)

    if md5 and source_df['md5'].isna().sum() > 0:
        print("calculating missing MD5 hash")
        source_df['md5'] = calculate_missing_md5(source_df, 'path', 'md5')

    if not isinstance(destination, pd.DataFrame):
        print("extracting data from DESTINATION")
        destination_df = listing_files(destination, md5=md5, exclude=exclude, recursive=recursive, n_jobs=n_jobs)

    if md5 and destination_df['md5'].isna().sum() > 0:
        print("calculating missing MD5 hash")
        destination_df['md5'] = calculate_missing_md5(destination_df, 'path', 'md5')

    # find common relative path, starting from the shortest path
    source_common = source_df.loc[source_df.folder.str.len().sort_values().index[0]].folder
    common_relative = source_common[2:] if ':' in source_common[:2] else source_common

    i = 0
    while False in [common_relative.split('/')[i] in each for each in source_df.folder.values]\
            and i < len(common_relative.split('/')):
        i += 1
    f = len(common_relative.split('/'))
    while False in ['/'.join(common_relative.split('/')[i:f]) in each for each in source_df.folder.values]\
            and f > i:
        f -= 1
    common_relative = common_relative[i: f] + '' if common_relative[i: f].endswith('/') else '/'

    if len(common_relative) > 0 and common_relative in source_common and \
        exists(source_common[:source_common.index(common_relative) + len(common_relative)]):
        source_common = source_common[:source_common.index(common_relative) + len(common_relative)]

    if log_file is None:
        log_file = source_common  # the shortest common folder in the source path

    i, f = 0, len(common_relative)
    while False in [common_relative in each for each in destination_df.folder.values] and f > i:
        i += 1
    common_relative = common_relative[i: f] + '' if common_relative[i: f].endswith('/') else '/'

    destination_common = destination_df.loc[destination_df.folder.str.len().sort_values().index[0]].folder
    if len(common_relative) > 0 and common_relative in destination_common and \
            exists(destination_common[:destination_common.index(common_relative) + len(common_relative)]):
        destination_common = destination_common[:destination_common.index(common_relative)+len(common_relative)]

    # write the relative-common path for each file
    source_root = source_common[:source_common.index(common_relative)]
    source_df['relative'] = [each[len(source_root):] for each in source_df.path.values]

    destination_root = destination_common[:destination_common.index(common_relative)]
    destination_df['relative'] = [each[len(destination_root):] for each in destination_df.path.values]

    # found files from source already in destination
    if md5:
        merge_md5 = source_df[(source_df.md5 != 'folder') & source_df.md5.notna()].merge(
                destination_df[(destination_df.md5 != 'folder') & destination_df.md5.notna()],
                on='md5', how='outer', suffixes=('_source', '_destination'))
        merge_relative = source_df[(source_df.md5 == 'folder') | source_df.md5.isna()].merge(
                destination_df[(destination_df.md5 == 'folder') | destination_df.md5.isna()].drop(columns='md5'),
                on='relative', how='outer', suffixes=('_source', '_destination'))
        merge_relative['relative_destination'] = merge_relative.relative
        merge_relative.rename(columns={'relative': 'relative_source'}, inplace=True)
        relative_df = pd.concat([merge_md5, merge_relative], axis=0)
        del merge_md5
        del merge_relative
    else:
        relative_df = source_df.drop(columns='md5').merge(
            destination_df.drop(columns='md5'), on='relative', how='outer', suffixes=('_source', '_destination'))

    relative_df.reset_index(drop=True, inplace=True)
    relative_df['source_root'] = source_root
    relative_df['destination_root'] = destination_root

    if report:
        print("writing excel mirror report files")
        xlsx = pd.ExcelWriter(log_file)
        source_df.to_excel(xlsx, sheet_name='source', index=False)
        destination_df.to_excel(xlsx, sheet_name='destination', index=False)
        relative_df.to_excel(xlsx, sheet_name='relative', index=False)
        xlsx.close()

    print("mirror done!")

    if pass_log_folder:
        return relative_df, extension(log_file)[2]
    return relative_df

def make_actions(relative_df, md5: bool=True, delete: bool=True, verbose: bool=True):
    # files and folders already in destination
    to_ignore = relative_df.md5.notna() & \
                (relative_df.relative_source == relative_df.relative_destination)
    relative_df.loc[to_ignore, 'action'] = 'done'

    # files that are not in source, to be deleted from destination
    to_delete = relative_df.path_source.isna()
    if delete:
        relative_df.loc[to_delete, 'action'] = 'delete'
    else:
        relative_df.loc[to_delete, 'action'] = 'ignore'

    # files existing in source and destination, but in different relative paths
    if md5:
        to_move = relative_df.md5.notna() & (relative_df.md5 != 'folder') & \
                  (relative_df.path_source.notna()) & \
                  (relative_df.relative_source != relative_df.relative_destination)
        relative_df.loc[to_move, 'action'] = 'move'

    # files missing in destination to copy
    to_copy = relative_df.path_destination.isna()
    relative_df.loc[to_copy, 'action'] = 'copy'

    return relative_df

def mirror_delete(relative_df, n_jobs=None, simulate: bool=True, verbose: bool=True):
    to_delete = relative_df.loc[relative_df.action == 'delete'].index
    print(f"{len(to_delete)} files to delete.")
    if simulate:
        relative_df.loc[to_delete, 'simulation'] = [
            f"rm({each})" for each in relative_df.loc[to_delete].path_destination.to_list()]
        return relative_df

    with Pool(n_jobs) as pool:
        digest = pool.map(rm, relative_df.loc[to_delete].path_destination.to_list())
    relative_df.loc[to_delete, 'execution'] = digest
    return relative_df

def mirror_move(relative_df, md5: bool=True, attempts: int=3, if_exists: str='both', n_jobs=None, simulate: bool=True,
                verbose: bool=True):
    def _mv(s: pd.Series):
        return mv(s.path_destination,
                  (s['path_destination']).replace(s['relative_destination'], s['relative_source']),
                  md5=md5,
                  source_md5=s['md5'],
                  attempts=attempts,
                  if_exists=if_exists,
                  )
    if if_exists not in ['stop', 'both', 'overwrite']:
        raise ValueError(f"`if_exists` must one of the strings: 'stop', 'both' or 'overwrite', not {if_exists}.")

    to_move = relative_df.loc[relative_df.action == 'move'].index
    print(f"{len(to_move)} files to move.")
    if simulate:
        relative_df.loc[to_move, 'simulation'] = [f"cp({s.path_destination}, \n{(s['path_destination']).replace(s['relative_destination'], s['relative_source'])}, \nmd5={md5}, source_md5={s['md5']}, attempts={attempts}, if_exists={if_exists},)"
                                 for s in relative_df.loc[to_move].T]
        return relative_df

    with Pool(n_jobs) as pool:
        digest = pool.map(_mv, relative_df.loc[to_move].T)
    relative_df.loc[to_move, 'execution'] = digest
    return relative_df

def mirror_copy(relative_df, md5: bool=True, attempts: int=3, if_exists: str='both', n_jobs=None,
                simulate: bool=True, verbose: bool=True):
    def _cp(s_: pd.Series):
        return cp(s_.path_destination,
                  f"{s_['destination_root']}{s_['relative_source']}",
                  md5=md5,
                  source_md5=s_['md5'],
                  attempts=attempts,
                  if_exists=if_exists,
                  )

    if if_exists not in ['stop', 'both', 'overwrite']:
        raise ValueError(f"`if_exists` must one of the strings: 'stop', 'both' or 'overwrite', not {if_exists}.")

    to_copy = relative_df.loc[relative_df.action == 'copy'].index
    print(f"{len(to_copy)} files to copy.")
    if simulate:
        if len(to_copy) == 1:
            s = relative_df.loc[to_copy].T
            relative_df.loc[to_copy, 'simulation'] = f"cp({s.path_source}, \n{s['destination_root']}{s['relative_source']}, \nmd5={md5}, source_md5={s['md5']}, attempts={attempts}, if_exists={if_exists},)"
            return relative_df.loc[to_copy]
        else:
            relative_df.loc[to_copy, 'simulation'] = [f"cp({s.path_source}, \n{s['destination_root']}{s['relative_source']}, \nmd5={md5}, source_md5={s['md5']}, attempts={attempts}, if_exists={if_exists},)"
                                                      for s in relative_df.loc[to_copy].T]
        return relative_df

    if len(to_copy) == 1:
        s = relative_df.loc[to_copy].T
        digest = _cp(s)
    else:
        with Pool(n_jobs) as pool:
            digest = pool.map(_cp, relative_df.loc[to_copy].T)
    relative_df.loc[to_copy, 'execution'] = digest
    return relative_df

def execute_actions(relative_df, delete: bool=True, md5: bool=True, attempts: int=3, if_exists: str='both',
                    n_jobs=None, simulate: bool=True, verbose: bool=True):
    if if_exists not in ['stop', 'both', 'overwrite']:
        raise ValueError(f"`if_exists` must one of the strings: 'stop', 'both' or 'overwrite', not {if_exists}.")

    if type(relative_df) is str:
        relative_df = relative_df.replace('\\', '/')
        if exists(relative_df):
            if simulate:
                raise ValueError(f"When providing report of actions to execute, `simulate` should be False.")
            print(f"reading actions report from the file '{relative_df}'")
            relative_df = pd.read_excel(relative_df)
        else:
            raise ValueError(f"The file '{relative_df}' doesn't exist.")

    if delete:
        relative_df = mirror_delete(relative_df, simulate=simulate)
    relative_df = mirror_move(relative_df, md5=md5, attempts=attempts, if_exists=if_exists, n_jobs=n_jobs,
                              simulate=simulate)
    relative_df = mirror_copy(relative_df, md5=md5, attempts=attempts, if_exists=if_exists, n_jobs=n_jobs,
                              simulate=simulate)
    return relative_df

def run_backup(source, destination, file_pattern: str='*', *, exclude=None, md5: bool=True, if_exists: str='both',
               recursive: bool=True, log_file=None, delete: bool=True, attempts: int=3, n_jobs: int=None,
               simulate: bool=True, log_folder: str=None, verbose: bool=True):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if if_exists not in ['stop', 'both', 'overwrite']:
        raise ValueError(f"`if_exists` must one of the strings: 'stop', 'both' or 'overwrite', not {if_exists}.")

    if log_folder in [False, 'False', 'false', 'FALSE', '0', 0]:
        pass_log_folder = False
        log_folder = False
    elif log_folder is None:
        pass_log_folder = True
    elif type(log_folder) is str:
        log_folder = log_folder.replace('\\', '/')
        if not exists(log_folder):
            raise ValueError(f"The log folder '{log_folder}' doesn't exist.")
        pass_log_folder = False
    else:
        raise TypeError("`log_folder` must be a string.")

    df = mirror(source, destination, md5=md5, file_pattern=file_pattern, exclude=exclude,
                recursive=recursive, log_file=log_file,
                n_jobs=n_jobs, pass_log_folder=pass_log_folder, verbose=verbose)
    if pass_log_folder:
        df, log_folder = df[0], df[1]
    if log_folder is False:  # is False to be explicit
        pass
    else:
        log_folder = f"{log_folder}{'' if log_folder.endswith('/') else '/'}"

    if bool(log_folder):
        df.to_excel(f'{log_folder}run_backup_mirror {now}.xlsx', sheet_name=now, index=False)
    mk = make_actions(df, md5=md5, delete=delete, verbose=verbose)
    if bool(log_folder):
        mk.to_excel(f'{log_folder}run_backup_make {now}.xlsx', sheet_name=now, index=False)
    ex = execute_actions(mk, delete=delete, md5=md5, attempts=attempts, if_exists=if_exists,
                         n_jobs=n_jobs, simulate=simulate, verbose=verbose)
    if bool(log_folder):
        ex.to_excel(f'{log_folder}run_backup_execute {now}.xlsx', sheet_name=now, index=False)
    return ex