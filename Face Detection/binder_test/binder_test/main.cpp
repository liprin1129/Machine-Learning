//
//  main.cpp
//  binder_test
//
//  Created by 170 on 2018/06/20.
//  Copyright © 2018 170. All rights reserved.
//

#include <iostream>
#include <boost/python.hpp>
#include <Python.h>

namespace py = boost::python;

int main(int argc, const char * argv[]) {
	//Py_SetPythonHome("/Users/user170/Developments/PyEnvs/3.6PyEnv/");
	Py_Initialize();
	py::object main_module = py::import("__main__");
    //py::object main_namespace = main_module.attr("__dict__");
	
    //boost::python::exec("import random as np", main_namespace);
	
	
	// insert code here...
	std::cout << "Hello, World!\n";
	return 0;
}
