#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:17:21 2017

@author: pure
"""

try:
    print("Use cPickle")
    import _pickle as pickle
except:
    print("Use pickle")
    import pickle

class PickleHelper(object):
    @classmethod
    def validation_check(cls, _path, _name):
        assert (_path or _name is not None), "Error: set corret path and name."
                
        if not _path.endswith("/"):
            return _path + "/"
            
        else:
            return _path
    
    @classmethod
    def save_to_pickle(cls, path = None, name = None, data = None):
        path = cls.validation_check(path, name)
        
        with open(path+name, "wb") as f:
            print("\t => Save '{0}' to '{1}'".format(name, path))
            pickle.dump(data, f)
        
    @classmethod        
    def load_pickle(cls, path = None, name = None):
        path = cls.validation_check(path, name)
        
        with open(path+name, "rb") as f:
            print("\t=> Load '{0}' to '{1}'".format(name, path))
            data = pickle.load(f)
        
        return data
            
if __name__ == "__main__":
    test = "Test Test Test Test Test Test Test"
    
    #PickleHelper.save_to_pickle()
    PickleHelper.save_to_pickle(path = "../../model/output_data", name = "test.pickle", data = test)
