# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /work/app

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /work/app/build

# Include any dependencies generated for this target.
include CMakeFiles/get_prediction.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/get_prediction.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/get_prediction.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/get_prediction.dir/flags.make

CMakeFiles/get_prediction.dir/get_prediction.cpp.o: CMakeFiles/get_prediction.dir/flags.make
CMakeFiles/get_prediction.dir/get_prediction.cpp.o: ../get_prediction.cpp
CMakeFiles/get_prediction.dir/get_prediction.cpp.o: CMakeFiles/get_prediction.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/get_prediction.dir/get_prediction.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/get_prediction.dir/get_prediction.cpp.o -MF CMakeFiles/get_prediction.dir/get_prediction.cpp.o.d -o CMakeFiles/get_prediction.dir/get_prediction.cpp.o -c /work/app/get_prediction.cpp

CMakeFiles/get_prediction.dir/get_prediction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/get_prediction.dir/get_prediction.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/app/get_prediction.cpp > CMakeFiles/get_prediction.dir/get_prediction.cpp.i

CMakeFiles/get_prediction.dir/get_prediction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/get_prediction.dir/get_prediction.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/app/get_prediction.cpp -o CMakeFiles/get_prediction.dir/get_prediction.cpp.s

# Object files for target get_prediction
get_prediction_OBJECTS = \
"CMakeFiles/get_prediction.dir/get_prediction.cpp.o"

# External object files for target get_prediction
get_prediction_EXTERNAL_OBJECTS =

get_prediction: CMakeFiles/get_prediction.dir/get_prediction.cpp.o
get_prediction: CMakeFiles/get_prediction.dir/build.make
get_prediction: /opt/tensorflow/lib/libtensorflow_cc.so
get_prediction: /opt/tensorflow/lib/libtensorflow_framework.so
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
get_prediction: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
get_prediction: CMakeFiles/get_prediction.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/work/app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable get_prediction"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/get_prediction.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/get_prediction.dir/build: get_prediction
.PHONY : CMakeFiles/get_prediction.dir/build

CMakeFiles/get_prediction.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/get_prediction.dir/cmake_clean.cmake
.PHONY : CMakeFiles/get_prediction.dir/clean

CMakeFiles/get_prediction.dir/depend:
	cd /work/app/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/app /work/app /work/app/build /work/app/build /work/app/build/CMakeFiles/get_prediction.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/get_prediction.dir/depend
