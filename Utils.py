import os

def ifNoneCreateDirs(filePath):
    if not os.path.exists(filePath):
        os.makedirs(filePath)
