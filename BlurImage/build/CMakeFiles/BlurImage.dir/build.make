# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mpiuser/Paralela-2022/BlurImage

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mpiuser/Paralela-2022/BlurImage/build

# Include any dependencies generated for this target.
include CMakeFiles/BlurImage.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/BlurImage.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BlurImage.dir/flags.make

CMakeFiles/BlurImage.dir/src/main_mpi.cpp.o: CMakeFiles/BlurImage.dir/flags.make
CMakeFiles/BlurImage.dir/src/main_mpi.cpp.o: ../src/main_mpi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mpiuser/Paralela-2022/BlurImage/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/BlurImage.dir/src/main_mpi.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/BlurImage.dir/src/main_mpi.cpp.o -c /home/mpiuser/Paralela-2022/BlurImage/src/main_mpi.cpp

CMakeFiles/BlurImage.dir/src/main_mpi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BlurImage.dir/src/main_mpi.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mpiuser/Paralela-2022/BlurImage/src/main_mpi.cpp > CMakeFiles/BlurImage.dir/src/main_mpi.cpp.i

CMakeFiles/BlurImage.dir/src/main_mpi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BlurImage.dir/src/main_mpi.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mpiuser/Paralela-2022/BlurImage/src/main_mpi.cpp -o CMakeFiles/BlurImage.dir/src/main_mpi.cpp.s

# Object files for target BlurImage
BlurImage_OBJECTS = \
"CMakeFiles/BlurImage.dir/src/main_mpi.cpp.o"

# External object files for target BlurImage
BlurImage_EXTERNAL_OBJECTS =

BlurImage: CMakeFiles/BlurImage.dir/src/main_mpi.cpp.o
BlurImage: CMakeFiles/BlurImage.dir/build.make
BlurImage: /usr/local/lib/libopencv_gapi.so.4.6.0
BlurImage: /usr/local/lib/libopencv_highgui.so.4.6.0
BlurImage: /usr/local/lib/libopencv_ml.so.4.6.0
BlurImage: /usr/local/lib/libopencv_objdetect.so.4.6.0
BlurImage: /usr/local/lib/libopencv_photo.so.4.6.0
BlurImage: /usr/local/lib/libopencv_stitching.so.4.6.0
BlurImage: /usr/local/lib/libopencv_video.so.4.6.0
BlurImage: /usr/local/lib/libopencv_videoio.so.4.6.0
BlurImage: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
BlurImage: /usr/local/lib/libopencv_dnn.so.4.6.0
BlurImage: /usr/local/lib/libopencv_calib3d.so.4.6.0
BlurImage: /usr/local/lib/libopencv_features2d.so.4.6.0
BlurImage: /usr/local/lib/libopencv_flann.so.4.6.0
BlurImage: /usr/local/lib/libopencv_imgproc.so.4.6.0
BlurImage: /usr/local/lib/libopencv_core.so.4.6.0
BlurImage: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
BlurImage: /usr/lib/x86_64-linux-gnu/libpthread.so
BlurImage: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
BlurImage: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
BlurImage: CMakeFiles/BlurImage.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mpiuser/Paralela-2022/BlurImage/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable BlurImage"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BlurImage.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BlurImage.dir/build: BlurImage

.PHONY : CMakeFiles/BlurImage.dir/build

CMakeFiles/BlurImage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BlurImage.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BlurImage.dir/clean

CMakeFiles/BlurImage.dir/depend:
	cd /home/mpiuser/Paralela-2022/BlurImage/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mpiuser/Paralela-2022/BlurImage /home/mpiuser/Paralela-2022/BlurImage /home/mpiuser/Paralela-2022/BlurImage/build /home/mpiuser/Paralela-2022/BlurImage/build /home/mpiuser/Paralela-2022/BlurImage/build/CMakeFiles/BlurImage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BlurImage.dir/depend

