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

# Include any dependencies generated for this target.
include main/CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include main/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include main/CMakeFiles/main.dir/flags.make

main/CMakeFiles/main.dir/src/main.c.obj: main/CMakeFiles/main.dir/flags.make
main/CMakeFiles/main.dir/src/main.c.obj: ../main/src/main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zyx/repos/rv1126_bsp/projects/hello-world/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object main/CMakeFiles/main.dir/src/main.c.obj"
	cd /home/zyx/repos/rv1126_bsp/projects/hello-world/build/main && /home/zyx/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/main.dir/src/main.c.obj   -c /home/zyx/repos/rv1126_bsp/projects/hello-world/main/src/main.c

main/CMakeFiles/main.dir/src/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/main.dir/src/main.c.i"
	cd /home/zyx/repos/rv1126_bsp/projects/hello-world/build/main && /home/zyx/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zyx/repos/rv1126_bsp/projects/hello-world/main/src/main.c > CMakeFiles/main.dir/src/main.c.i

main/CMakeFiles/main.dir/src/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/main.dir/src/main.c.s"
	cd /home/zyx/repos/rv1126_bsp/projects/hello-world/build/main && /home/zyx/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zyx/repos/rv1126_bsp/projects/hello-world/main/src/main.c -o CMakeFiles/main.dir/src/main.c.s

main/CMakeFiles/main.dir/src/main.c.obj.requires:

.PHONY : main/CMakeFiles/main.dir/src/main.c.obj.requires

main/CMakeFiles/main.dir/src/main.c.obj.provides: main/CMakeFiles/main.dir/src/main.c.obj.requires
	$(MAKE) -f main/CMakeFiles/main.dir/build.make main/CMakeFiles/main.dir/src/main.c.obj.provides.build
.PHONY : main/CMakeFiles/main.dir/src/main.c.obj.provides

main/CMakeFiles/main.dir/src/main.c.obj.provides.build: main/CMakeFiles/main.dir/src/main.c.obj


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/main.c.obj"

# External object files for target main
main_EXTERNAL_OBJECTS =

main/libmain.a: main/CMakeFiles/main.dir/src/main.c.obj
main/libmain.a: main/CMakeFiles/main.dir/build.make
main/libmain.a: main/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zyx/repos/rv1126_bsp/projects/hello-world/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libmain.a"
	cd /home/zyx/repos/rv1126_bsp/projects/hello-world/build/main && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean_target.cmake
	cd /home/zyx/repos/rv1126_bsp/projects/hello-world/build/main && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
main/CMakeFiles/main.dir/build: main/libmain.a

.PHONY : main/CMakeFiles/main.dir/build

main/CMakeFiles/main.dir/requires: main/CMakeFiles/main.dir/src/main.c.obj.requires

.PHONY : main/CMakeFiles/main.dir/requires

main/CMakeFiles/main.dir/clean:
	cd /home/zyx/repos/rv1126_bsp/projects/hello-world/build/main && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : main/CMakeFiles/main.dir/clean

main/CMakeFiles/main.dir/depend:
	cd /home/zyx/repos/rv1126_bsp/projects/hello-world/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zyx/repos/rv1126_bsp/projects/hello-world /home/zyx/repos/rv1126_bsp/projects/hello-world/main /home/zyx/repos/rv1126_bsp/projects/hello-world/build /home/zyx/repos/rv1126_bsp/projects/hello-world/build/main /home/zyx/repos/rv1126_bsp/projects/hello-world/build/main/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : main/CMakeFiles/main.dir/depend

