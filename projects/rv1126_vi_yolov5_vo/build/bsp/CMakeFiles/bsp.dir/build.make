# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build

# Include any dependencies generated for this target.
include bsp/CMakeFiles/bsp.dir/depend.make

# Include the progress variables for this target.
include bsp/CMakeFiles/bsp.dir/progress.make

# Include the compile flags for this target's objects.
include bsp/CMakeFiles/bsp.dir/flags.make

bsp/CMakeFiles/bsp.dir/src/bsp.c.obj: bsp/CMakeFiles/bsp.dir/flags.make
bsp/CMakeFiles/bsp.dir/src/bsp.c.obj: /home/zyx/repos/rv1126_bsp/components/bsp/src/bsp.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object bsp/CMakeFiles/bsp.dir/src/bsp.c.obj"
	cd /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/bsp && /home/zyx/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/bsp.dir/src/bsp.c.obj   -c /home/zyx/repos/rv1126_bsp/components/bsp/src/bsp.c

bsp/CMakeFiles/bsp.dir/src/bsp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/bsp.dir/src/bsp.c.i"
	cd /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/bsp && /home/zyx/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zyx/repos/rv1126_bsp/components/bsp/src/bsp.c > CMakeFiles/bsp.dir/src/bsp.c.i

bsp/CMakeFiles/bsp.dir/src/bsp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/bsp.dir/src/bsp.c.s"
	cd /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/bsp && /home/zyx/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zyx/repos/rv1126_bsp/components/bsp/src/bsp.c -o CMakeFiles/bsp.dir/src/bsp.c.s

bsp/CMakeFiles/bsp.dir/src/bsp.c.obj.requires:

.PHONY : bsp/CMakeFiles/bsp.dir/src/bsp.c.obj.requires

bsp/CMakeFiles/bsp.dir/src/bsp.c.obj.provides: bsp/CMakeFiles/bsp.dir/src/bsp.c.obj.requires
	$(MAKE) -f bsp/CMakeFiles/bsp.dir/build.make bsp/CMakeFiles/bsp.dir/src/bsp.c.obj.provides.build
.PHONY : bsp/CMakeFiles/bsp.dir/src/bsp.c.obj.provides

bsp/CMakeFiles/bsp.dir/src/bsp.c.obj.provides.build: bsp/CMakeFiles/bsp.dir/src/bsp.c.obj


# Object files for target bsp
bsp_OBJECTS = \
"CMakeFiles/bsp.dir/src/bsp.c.obj"

# External object files for target bsp
bsp_EXTERNAL_OBJECTS =

bsp/libbsp.a: bsp/CMakeFiles/bsp.dir/src/bsp.c.obj
bsp/libbsp.a: bsp/CMakeFiles/bsp.dir/build.make
bsp/libbsp.a: bsp/CMakeFiles/bsp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library libbsp.a"
	cd /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/bsp && $(CMAKE_COMMAND) -P CMakeFiles/bsp.dir/cmake_clean_target.cmake
	cd /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/bsp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bsp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bsp/CMakeFiles/bsp.dir/build: bsp/libbsp.a

.PHONY : bsp/CMakeFiles/bsp.dir/build

bsp/CMakeFiles/bsp.dir/requires: bsp/CMakeFiles/bsp.dir/src/bsp.c.obj.requires

.PHONY : bsp/CMakeFiles/bsp.dir/requires

bsp/CMakeFiles/bsp.dir/clean:
	cd /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/bsp && $(CMAKE_COMMAND) -P CMakeFiles/bsp.dir/cmake_clean.cmake
.PHONY : bsp/CMakeFiles/bsp.dir/clean

bsp/CMakeFiles/bsp.dir/depend:
	cd /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo /home/zyx/repos/rv1126_bsp/components/bsp /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/bsp /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/bsp/CMakeFiles/bsp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bsp/CMakeFiles/bsp.dir/depend

