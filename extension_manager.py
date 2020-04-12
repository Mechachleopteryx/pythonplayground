#!/usr/bin/env python
# coding: utf-8



# Author: Mathias Godwin
# Email: godwinsaint6@gmail.com
# Date: 9-Apr-20
# Authenticity : under-construction 

import os
import re
import warnings

class extension_manager:
    """
       Provides with functions to manipulate on files in a dir(s).
       ==========================================================
       ==========================================================
       --- version:: loading...
       
       Parameters:
       ----------
       Dir : directory (of folders) to a file extension or file name.
       Example:
       -------
             "C:\\" or "C:\\Gwin\\"
       name_or_extension : name or extension of file to work with.
              name:
              ----
              When provided with name, set "by_extension to False".
              The function would lookup or
              search for whatever matches that name in the Directory
              and all sub-directories.
              extension:
              ---------
              When provided with extension, set "by_extension to "True".
              It then perform same logic as above.   
       Example:
       --------
              name:
                   'python', 'music', 'text', ..etc
              extension:
                   '.html', '.mp3', '.mp4', ..etc
       by_extension: bool, default True
             "False" when searching file names rather than extension.
       strategy : action to perform with the file if found
             Available: (copy, move, delete)
        

    """
    def __init__(self, Dir="C:\\", name_or_extension=None, *, by_extension=True, strategy='copy'):
        self.Dir = Dir
        self.name_or_extension = name_or_extension
        self.by_extension = by_extension
        self.strategy = strategy.lower()
        self.en_list = []
        
    def extension_search(self):
        if self.name_or_extension is None:
            pass
        elif self.by_extension:
            for root, dirs, file in os.walk(self.Dir):
                get_files_tolist = [os.path.join(root, name) for name in file]
                for file in get_files_tolist:
                    search_ = re.search(self.name_or_extension, file)
                    if ((search_ is not None) and (file[-len(self.name_or_extension):] == self.name_or_extension)):
                        self.en_list.append(file)
        else:
            for root, dirs, files in os.walk(self.Dir):
                get_files_tolist = [os.path.join(root, name) for name in files]
                for file in get_files_tolist:
                    search_ = re.search(self.name_or_extension, file)
                    if (search_ is not None) and search_.span()[0] == (search_.span()[1] - len(self.name_or_extension)):
                        self.en_list.append(file)
        return self.en_list
    
    def group_extension(self, extensions_dir='C:\\', 
                        names_of_extensions=[], *, dir_target_names=[]):
        if extensions_dir != self.Dir:
            warnings.warn(f"\ndidn't provide directory\n***** using default: {extensions_dir} *****")
            self.Dir = extensions_dir
        
        if len(names_of_extensions) == 0:
            raise ValueError('you must provide a dirs to copy from')
        if not len(names_of_extensions) == len(dir_target_names):
            warnings.warn('\ntarget directory name not provided \n************over-riding dir target names************')
            dir_target_names = []
            searched_ex = []
            for ext in names_of_extensions:
                if ext.startswith('.'):
                    _, dir_target_n = ext.split('.')
                    dir_target_names.append(dir_target_n)
                else:
                    dir_target_names.append(ext)  
        #***********************************************#
                searched_ex.append(extension_manager(self.Dir, ext, 
                                                     strategy=self.strategy, 
                                                     by_extension=self.by_extension).extension_search())
        #***********************************************#
            for dir_target, extension_ in zip(dir_target_names, searched_ex):
                for dr in extension_:
                    if dir_target not in set(os.listdir(self.Dir)):
                        os.chdir(self.Dir)
                        os.mkdir(dir_target)
                    if self.strategy == 'copy':
                        warnings.warn('\nabout to #copy a file to '+ self.Dir + dir_target)
                        os.chdir(self.Dir + dir_target)
                        os.system(f'copy "{dr}"')
                    if self.strategy == 'move':
                        warnings.warn('\nabout to #move a file to '+ self.Dir + dir_target)
                        os.chdir(self.Dir + dir_target)
                        os.system(f'move "{dr}"')
                    if self.strategy == 'delete':
                        warnings.warn('\nabout to #delete a file from  '+ self.Dir + dir_target)
                        os.chdir(self.Dir + dir_target)
                        os.system(f'del "{dr}"')
                
                print(f"{len(extension_)} {dir_target} file(s) in {self.Dir}")

            return f'{self.strategy} completed!'

