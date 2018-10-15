/*
 * MainDelegate.cpp
 *
 *  Created on: May 30, 2018
 *      Author: user170
 */

#include "MainDelegate.hpp"

int MainDelegate::mainDelegation(int argc, char** argv){
	// Face Detection Coding in progress
	//CameraManager cm;
	//cm.cameraManagerHasLoaded(0);

	//FileSystemManager fs;
	//fs.numOfFiles("/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Face/Face-SJC/170/");

	ViewManager vm;

	/***********************/
	/*  Start View  */
	/***********************/
	//vm.viewHasLoaded(0);
	//vm.mainView();


	boost::thread th1(boost::bind(&ViewManager::viewHasLoaded, &vm, 0));
	boost::thread th2(boost::bind(&ViewManager::saveFaceLoop, &vm));
	boost::thread th3(boost::bind(&ViewManager::insertSoccerBall, &vm));

	//boost::thread th1(&ViewManager::mainView, &vm);
	//boost::thread th1(boost::bind(&ViewManager::mainView, &vm, 0));

	th1.join();
	th2.join();
	th3.join();

	return 0;
}
