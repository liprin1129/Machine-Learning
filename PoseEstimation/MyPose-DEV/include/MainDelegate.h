#ifndef MAINDELEGATE_HPP_
#define MAINDELEGATE_HPP_

#include "CameraManager.h"

class MainDelegate {
	private:
		CameraManager cm;
	public:
		int mainDelegation(int argc, char** argv);
};

#endif /* MAINDELEGATE_HPP_ */