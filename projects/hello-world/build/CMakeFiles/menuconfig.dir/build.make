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
CMAKE_SOURCE_DIR = /home/zyx/repos/rv1126_bsp/projects/hello-world

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zyx/repos/rv1126_bsp/projects/hello-world/build

# Utility rule file for menuconfig.

# Include the progress variables for this target.
include CMakeFiles/menuconfig.dir/progress.make

CMakeFiles/menuconfig:
	python3 /home/zyx/repos/rv1126_bsp/tools/kconfig/genconfig.py --kconfig /home/zyx/repos/rv1126_bsp/Kconfig --defaults /home/zyx/repos/rv1126_bsp/projects/hello-world/config_defaults.mk --menuconfig True --env SDK_PATH=/home/zyx/repos/rv1126_bsp --env PROJECT_PATH=/home/zyx/repos/rv1126_bsp/projects/hello-world --output makefile /home/zyx/repos/rv1126_bsp/projects/hello-world/build/config/global_config.mk --output cmake /home/zyx/repos/rv1126_bsp/projects/hello-world/build/config/global_config.cmake --output header /home/zyx/repos/rv1126_bsp/projects/hello-world/build/config/global_config.h

menuconfig: CMakeFiles/menuconfig
menuconfig: CMakeFiles/menuconfig.dir/build.make

.PHONY : menuconfig

# Rule to build all files generated by this target.
CMakeFiles/menuconfig.dir/build: menuconfig

.PHONY : CMakeFiles/menuconfig.dir/build

CMakeFiles/menuconfig.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/menuconfig.dir/cmake_clean.cmake
.PHONY : CMakeFiles/menuconfig.dir/clean

CMakeFiles/menuconfig.dir/depend:
	cd /home/zyx/repos/rv1126_bsp/projects/hello-world/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zyx/repos/rv1126_bsp/projects/hello-world /home/zyx/repos/rv1126_bsp/projects/hello-world /home/zyx/repos/rv1126_bsp/projects/hello-world/build /home/zyx/repos/rv1126_bsp/projects/hello-world/build /home/zyx/repos/rv1126_bsp/projects/hello-world/build/CMakeFiles/menuconfig.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/menuconfig.dir/depend
