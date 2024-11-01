# `folder_backup`

Script to back up one or several folders into other folders:
- replicating the source folders structure
- not copying files already in destination (better if `md5=True`)
- moving (not copying) files already present in the destination path (requires `md5=True`)
- delete (`delete=True`), or not (`delete=False`) files from destination that are not present in source
- copy new files from source into destination verifying the copy (requires `md5=True`)
- if a file with the same name (full path) already exists in destination, the parameter `if_exists` choose how to proceed:
  - **'stop'** will not copy or move the file (skip)
  - **'both'** will keep both files, adding a numeric suffix to the new file
  - **'overwrite'** will overwrite the file in destination

To save time calculating MD5 checksum everytime, setting the parameter `recall_md5=True` will write an accompanying _.md5_ file for each read file.
Next time the script will run, the md5 will be recovered from the _.md5_ file instead of recalculating at every run of the script.
  
It is possible to prepare a _simulation_ of the actions to be done, using the parameter `simulate=True`.  
This mode only create the report in Excel format, that can be reviewed and later executed using the function `execute_actions`.
  
Further documentation and examples will come later...  
  