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
include CMakeFiles/rv1126_vi_yolov5_vo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rv1126_vi_yolov5_vo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rv1126_vi_yolov5_vo.dir/flags.make

exe_src.c:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating exe_src.c"
	/usr/bin/cmake -E touch /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/exe_src.c

CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj: CMakeFiles/rv1126_vi_yolov5_vo.dir/flags.make
CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj: exe_src.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj"
	/home/zyx/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj   -c /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/exe_src.c

CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.i"
	/home/zyx/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/exe_src.c > CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.i

CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.s"
	/home/zyx/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/exe_src.c -o CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.s

CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj.requires:

.PHONY : CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj.requires

CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj.provides: CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj.requires
	$(MAKE) -f CMakeFiles/rv1126_vi_yolov5_vo.dir/build.make CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj.provides.build
.PHONY : CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj.provides

CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj.provides.build: CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj


# Object files for target rv1126_vi_yolov5_vo
rv1126_vi_yolov5_vo_OBJECTS = \
"CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj"

# External object files for target rv1126_vi_yolov5_vo
rv1126_vi_yolov5_vo_EXTERNAL_OBJECTS =

rv1126_vi_yolov5_vo: CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj
rv1126_vi_yolov5_vo: CMakeFiles/rv1126_vi_yolov5_vo.dir/build.make
rv1126_vi_yolov5_vo: main/libmain.a
rv1126_vi_yolov5_vo: bsp/libbsp.a
rv1126_vi_yolov5_vo: npu/libnpu.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/lib/libopencv_calib3d.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/lib/libopencv_features2d.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/lib/libopencv_imgcodecs.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/share/OpenCV/3rdparty/lib/liblibjpeg-turbo.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/share/OpenCV/3rdparty/lib/liblibwebp.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/share/OpenCV/3rdparty/lib/liblibpng.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/share/OpenCV/3rdparty/lib/liblibtiff.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/share/OpenCV/3rdparty/lib/liblibjasper.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/share/OpenCV/3rdparty/lib/libIlmImf.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/lib/libopencv_video.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/lib/libopencv_imgproc.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/lib/libopencv_core.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/share/OpenCV/3rdparty/lib/libzlib.a
rv1126_vi_yolov5_vo: /home/zyx/repos/rv1126_bsp/components/opencv/share/OpenCV/3rdparty/lib/libtegra_hal.a
rv1126_vi_yolov5_vo: CMakeFiles/rv1126_vi_yolov5_vo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable rv1126_vi_yolov5_vo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rv1126_vi_yolov5_vo.dir/link.txt --verbose=$(VERBOSE)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "-- copy binary files to dist dir ..."
	mkdir -p /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/dist/lib
	cp /home/zyx/repos/rv1126_bsp/components/bsp/lib/libasound.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libavcodec.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libavformat.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libavutil.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libdrm.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libeasymedia.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libmd_share.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libod_share.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/librga.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/librkaiq.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libRKAP_3A.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libRKAP_ANR.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libRKAP_Common.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/librockchip_mpp.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libswresample.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libv4l2.so /home/zyx/repos/rv1126_bsp/components/bsp/lib/libv4lconvert.so /home/zyx/repos/rv1126_bsp/components/npu/lib/librknn_api.so /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/dist/lib
	cp /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/rv1126_vi_yolov5_vo /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/dist
	cp /home/zyx/repos/rv1126_bsp/tools/cmake/start_app_rv1126.sh /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/dist/start_app.sh
	chmod +x /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/dist/start_app.sh && echo "\$$curr_dir/rv1126_vi_yolov5_vo" >> /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/dist/start_app.sh

# Rule to build all files generated by this target.
CMakeFiles/rv1126_vi_yolov5_vo.dir/build: rv1126_vi_yolov5_vo

.PHONY : CMakeFiles/rv1126_vi_yolov5_vo.dir/build

CMakeFiles/rv1126_vi_yolov5_vo.dir/requires: CMakeFiles/rv1126_vi_yolov5_vo.dir/exe_src.c.obj.requires

.PHONY : CMakeFiles/rv1126_vi_yolov5_vo.dir/requires

CMakeFiles/rv1126_vi_yolov5_vo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rv1126_vi_yolov5_vo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rv1126_vi_yolov5_vo.dir/clean

CMakeFiles/rv1126_vi_yolov5_vo.dir/depend: exe_src.c
	cd /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build /home/zyx/repos/rv1126_bsp/projects/rv1126_vi_yolov5_vo/build/CMakeFiles/rv1126_vi_yolov5_vo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rv1126_vi_yolov5_vo.dir/depend

