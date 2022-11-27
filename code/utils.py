"""
Class which contains all common utility funstions
"""



import os
import shutil

import numpy as np


def removeDir(path):
    shutil.rmtree(path, ignore_errors=True)


def createDirIfDoesntExists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def createDir(path, exist_ok = False):
    if os.path.exists(path) and not exist_ok:
        print(f"Directory already exists : {path}")
        print(f"Type 'yes' to override...")
    
        if input() == 'yes':
            print(f'Removing dir {path}')
            removeDir(path)
            
    os.makedirs(path, exist_ok = exist_ok)


import json


def readJson(path):

    with open(path, 'r') as json_file:
        json_dict = json.load(json_file)

    return json_dict


def createNloadJson(path):

    json_dict = {}
    
    #Load existing dict if exists
    if os.path.exists(path):
        json_dict = readJson(path)

    return json_dict


def writeJson(json_dict, path):

    #Save Dict
    json_file = json.dumps(json_dict, indent=4)
    f = open(path,"w")
    f.write(json_file)
    f.close()


"""
Class to record text to file as well as print to cmd.
"""
class Logger():

    def __init__(self, filename) -> None:
    
        self.filename = filename

        self.file = open(self.filename, 'w')

    def log(self, txt):

        print(txt)
        self.file.write(f"\n{txt}\n")

    #Print all of the file contents
    def printAll(self):


        print(f"{self.file}\n\n")

        with open(self.filename, 'r') as f:
            print(f.read())


    def close(self,):

        self.file.close()


