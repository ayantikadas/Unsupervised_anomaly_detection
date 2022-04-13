import os
def check_dir_existance(path_):
    if not os.path.exists(path_):
        print('Directory does not exist')
        print('Making directory')
        os.mkdir(path_)
    else:
        print('Exists')