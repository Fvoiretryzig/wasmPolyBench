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
CMAKE_SOURCE_DIR = /home/tangshimei/wasmPolyBench

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tangshimei/wasmPolyBench

# Include any dependencies generated for this target.
include CMakeFiles/doitgen.wasm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/doitgen.wasm.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/doitgen.wasm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/doitgen.wasm.dir/flags.make

CMakeFiles/doitgen.wasm.dir/src/doitgen.c.obj: CMakeFiles/doitgen.wasm.dir/flags.make
CMakeFiles/doitgen.wasm.dir/src/doitgen.c.obj: src/doitgen.c
CMakeFiles/doitgen.wasm.dir/src/doitgen.c.obj: CMakeFiles/doitgen.wasm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tangshimei/wasmPolyBench/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/doitgen.wasm.dir/src/doitgen.c.obj"
	/home/tangshimei/wasi-sdk-14.0/bin/clang --target=wasm32-wasi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/doitgen.wasm.dir/src/doitgen.c.obj -MF CMakeFiles/doitgen.wasm.dir/src/doitgen.c.obj.d -o CMakeFiles/doitgen.wasm.dir/src/doitgen.c.obj -c /home/tangshimei/wasmPolyBench/src/doitgen.c

CMakeFiles/doitgen.wasm.dir/src/doitgen.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/doitgen.wasm.dir/src/doitgen.c.i"
	/home/tangshimei/wasi-sdk-14.0/bin/clang --target=wasm32-wasi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/tangshimei/wasmPolyBench/src/doitgen.c > CMakeFiles/doitgen.wasm.dir/src/doitgen.c.i

CMakeFiles/doitgen.wasm.dir/src/doitgen.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/doitgen.wasm.dir/src/doitgen.c.s"
	/home/tangshimei/wasi-sdk-14.0/bin/clang --target=wasm32-wasi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/tangshimei/wasmPolyBench/src/doitgen.c -o CMakeFiles/doitgen.wasm.dir/src/doitgen.c.s

# Object files for target doitgen.wasm
doitgen_wasm_OBJECTS = \
"CMakeFiles/doitgen.wasm.dir/src/doitgen.c.obj"

# External object files for target doitgen.wasm
doitgen_wasm_EXTERNAL_OBJECTS =

doitgen.wasm: CMakeFiles/doitgen.wasm.dir/src/doitgen.c.obj
doitgen.wasm: CMakeFiles/doitgen.wasm.dir/build.make
doitgen.wasm: CMakeFiles/doitgen.wasm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tangshimei/wasmPolyBench/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable doitgen.wasm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/doitgen.wasm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/doitgen.wasm.dir/build: doitgen.wasm
.PHONY : CMakeFiles/doitgen.wasm.dir/build

CMakeFiles/doitgen.wasm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/doitgen.wasm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/doitgen.wasm.dir/clean

CMakeFiles/doitgen.wasm.dir/depend:
	cd /home/tangshimei/wasmPolyBench && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tangshimei/wasmPolyBench /home/tangshimei/wasmPolyBench /home/tangshimei/wasmPolyBench /home/tangshimei/wasmPolyBench /home/tangshimei/wasmPolyBench/CMakeFiles/doitgen.wasm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/doitgen.wasm.dir/depend

