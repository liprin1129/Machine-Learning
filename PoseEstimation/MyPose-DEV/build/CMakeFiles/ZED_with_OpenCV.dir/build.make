# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build

# Include any dependencies generated for this target.
include CMakeFiles/ZED_with_OpenCV.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ZED_with_OpenCV.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ZED_with_OpenCV.dir/flags.make

CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.o: CMakeFiles/ZED_with_OpenCV.dir/flags.make
CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.o: ../src/CameraManager.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.o -c /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/CameraManager.cpp

CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/CameraManager.cpp > CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.i

CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/CameraManager.cpp -o CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.s

CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.o: CMakeFiles/ZED_with_OpenCV.dir/flags.make
CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.o: ../src/JsonFileManager.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.o -c /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/JsonFileManager.cpp

CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/JsonFileManager.cpp > CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.i

CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/JsonFileManager.cpp -o CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.s

CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.o: CMakeFiles/ZED_with_OpenCV.dir/flags.make
CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.o: ../src/MainDelegate.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.o -c /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/MainDelegate.cpp

CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/MainDelegate.cpp > CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.i

CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/MainDelegate.cpp -o CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.s

CMakeFiles/ZED_with_OpenCV.dir/src/main.o: CMakeFiles/ZED_with_OpenCV.dir/flags.make
CMakeFiles/ZED_with_OpenCV.dir/src/main.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/ZED_with_OpenCV.dir/src/main.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ZED_with_OpenCV.dir/src/main.o -c /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/main.cpp

CMakeFiles/ZED_with_OpenCV.dir/src/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ZED_with_OpenCV.dir/src/main.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/main.cpp > CMakeFiles/ZED_with_OpenCV.dir/src/main.i

CMakeFiles/ZED_with_OpenCV.dir/src/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ZED_with_OpenCV.dir/src/main.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/src/main.cpp -o CMakeFiles/ZED_with_OpenCV.dir/src/main.s

# Object files for target ZED_with_OpenCV
ZED_with_OpenCV_OBJECTS = \
"CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.o" \
"CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.o" \
"CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.o" \
"CMakeFiles/ZED_with_OpenCV.dir/src/main.o"

# External object files for target ZED_with_OpenCV
ZED_with_OpenCV_EXTERNAL_OBJECTS =

ZED_with_OpenCV: CMakeFiles/ZED_with_OpenCV.dir/src/CameraManager.o
ZED_with_OpenCV: CMakeFiles/ZED_with_OpenCV.dir/src/JsonFileManager.o
ZED_with_OpenCV: CMakeFiles/ZED_with_OpenCV.dir/src/MainDelegate.o
ZED_with_OpenCV: CMakeFiles/ZED_with_OpenCV.dir/src/main.o
ZED_with_OpenCV: CMakeFiles/ZED_with_OpenCV.dir/build.make
ZED_with_OpenCV: /usr/local/zed/lib/libsl_input.so
ZED_with_OpenCV: /usr/local/zed/lib/libsl_core.so
ZED_with_OpenCV: /usr/local/zed/lib/libsl_zed.so
ZED_with_OpenCV: /usr/lib/x86_64-linux-gnu/libopenblas.so
ZED_with_OpenCV: /usr/local/lib/libopencv_cudabgsegm.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudastereo.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_stitching.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_superres.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_videostab.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_aruco.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_bgsegm.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_bioinspired.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_ccalib.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_dpm.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_face.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_freetype.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_fuzzy.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_hfs.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_img_hash.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_line_descriptor.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_optflow.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_reg.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_rgbd.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_saliency.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_stereo.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_structured_light.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_surface_matching.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_tracking.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_xfeatures2d.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_ximgproc.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_xobjdetect.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_xphoto.so.3.4.5
ZED_with_OpenCV: /usr/lib/x86_64-linux-gnu/libcuda.so
ZED_with_OpenCV: /usr/local/cuda/lib64/libcudart.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppial.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppisu.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppicc.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppicom.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppidei.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppif.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppig.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppim.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppist.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libnppitc.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libcublas.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libcurand.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libcublas.so
ZED_with_OpenCV: /usr/local/cuda-10.0/lib64/libcurand.so
ZED_with_OpenCV: /usr/local/cuda/lib64/libnpps.so
ZED_with_OpenCV: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_shape.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudacodec.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudaoptflow.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudalegacy.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudawarping.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_video.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_datasets.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_plot.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_text.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_dnn.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_ml.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_objdetect.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_calib3d.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_features2d.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_flann.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_highgui.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_videoio.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_imgcodecs.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_photo.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudaimgproc.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudafilters.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudaarithm.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_imgproc.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_core.so.3.4.5
ZED_with_OpenCV: /usr/local/lib/libopencv_cudev.so.3.4.5
ZED_with_OpenCV: CMakeFiles/ZED_with_OpenCV.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable ZED_with_OpenCV"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ZED_with_OpenCV.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ZED_with_OpenCV.dir/build: ZED_with_OpenCV

.PHONY : CMakeFiles/ZED_with_OpenCV.dir/build

CMakeFiles/ZED_with_OpenCV.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ZED_with_OpenCV.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ZED_with_OpenCV.dir/clean

CMakeFiles/ZED_with_OpenCV.dir/depend:
	cd /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build /DEVs/Machine-Learning/PoseEstimation/MyPose-DEV/build/CMakeFiles/ZED_with_OpenCV.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ZED_with_OpenCV.dir/depend

