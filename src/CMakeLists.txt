#################################################################
#  CMake configure file for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2018/08/26
#################################################################
# Note: when multiple gcc versions are installed, cmake may find
# the highest version, but nvcc may throw an error saying it does
# not support this gcc version. To solve this, one should run
#      CC=gcc-X CXX=g++-X cmake ..
# where "X" is the version (such as 9) that both installed on
# your system and is supported by nvcc
#################################################################

cmake_minimum_required(VERSION 3.3)

project(mcxcl)

cmake_policy(SET CMP0074 NEW)  # to make -DOPENCL_ROOT= work with newer CMake versions as well

#find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)

add_subdirectory(zmat)

option(BUILD_MEX "Build mex" ON)

if(BUILD_PYTHON)
    add_subdirectory(pybind11)
    find_package (Python3 COMPONENTS Interpreter Development)
    include_directories(${PYTHON_INCLUDE_DIRS})
endif()

if(BUILD_MEX)
    find_package(Matlab)
endif()

set(CMAKE_CXX_FLAGS "-DMCX_EMBED_CL -DMCX_OPENCL -DUSE_OS_TIMER -D_hypot=hypot")
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS}")

if( ${CMAKE_COMPILER_IS_GNUCXX} )
    if(WIN32)
        set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -lgomp -Wl,--no-whole-archive -Wl,-Bdynamic")
    else()
        set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -static-libgcc -static-libstdc++")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -pedantic -Wall -fPIC -O3 -std=c99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pedantic -Wall -fPIC -O3")
endif()

# C Options
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../lib)

set(RESOURCE_COMPILER xxd)
set(RESOURCE_INPUT "mcx_core.cl")
set(RESOURCE_OUTPUT "${CMAKE_SOURCE_DIR}/mcx_core.clh")

add_custom_target(
    clheader
    COMMAND bash -c "cd ${CMAKE_SOURCE_DIR}; ${RESOURCE_COMPILER} -i ${RESOURCE_INPUT} | sed 's/\\([0-9a-f]\\)$/\\0, 0x00/' > ${RESOURCE_OUTPUT}"
    comment "Compiling ${RESOURCE_INPUT} to ${RESOURCE_OUTPUT}"
    VERBATIM
)

# Add include directories
include_directories(cjson ubj zmat zmat/easylzma ${OPENCL_INCLUDE_DIR})

# Add link directories
link_directories(zmat)

# Create mcx library
add_library(mcxcl STATIC
    mcx_host.cpp
    mcx_host.h
    mcx_utils.c
    mcx_utils.h
    mcx_shapes.c
    mcx_shapes.h
    mcx_bench.h
    mcx_tictoc.c
    mcx_tictoc.h
    cjson/cJSON.c
    cjson/cJSON.h
    ubj/ubj.h
    ubj/ubjw.c
    )

# Add all project units
add_executable(
    mcxcl-exe
    mcxcl.c
    )

set_target_properties(mcxcl-exe
        PROPERTIES OUTPUT_NAME mcxcl)

# Link options
target_link_libraries(
    mcxcl-exe
    mcxcl
    zmat
    OpenCL::OpenCL
    )

add_dependencies(mcxcl clheader)
add_dependencies(mcxcl-exe clheader)

if (BUILD_PYTHON)
    add_library(_pmcxcl MODULE
            mcx_host.cpp
            mcx_host.h
            mcx_utils.c
            mcx_utils.h
            mcx_shapes.c
            mcx_shapes.h
            mcx_bench.h
            mcx_tictoc.c
            mcx_tictoc.h
            cjson/cJSON.c
            cjson/cJSON.h
            pmcxcl.cpp
            )
    target_compile_definitions(_pmcxcl PUBLIC MCX_CONTAINER PYBIND11_VERSION_MAJOR)

    target_link_libraries(_pmcxcl pybind11::module pybind11::lto pybind11::windows_extras OpenCL::OpenCL)

    pybind11_extension(_pmcxcl)
    pybind11_strip(_pmcxcl)

    add_dependencies(_pmcxcl clheader)
    set_target_properties(_pmcxcl PROPERTIES CXX_VISIBILITY_PRESET "hidden")
endif()

# Build mex file
if(BUILD_MEX AND Matlab_FOUND)
    # Create mcx-matlab library
    add_library(mcxcl-matlab STATIC
            mcx_host.cpp
            mcx_host.h
            mcx_utils.c
            mcx_utils.h
            mcx_shapes.c
            mcx_shapes.h
            mcx_bench.h
            mcx_tictoc.c
            mcx_tictoc.h
            cjson/cJSON.c
            cjson/cJSON.h
            )

    target_compile_definitions(mcxcl-matlab PUBLIC MCX_CONTAINER MATLAB_MEX_FILE)

    if(${CMAKE_VERSION} VERSION_LESS "3.24.0")
            matlab_add_mex(
              NAME mcxlabcl
              SRC mcxlabcl.cpp
              LINK_TO mcxcl-matlab OpenCL::OpenCL
            )
    else()
            matlab_add_mex(
              NAME mcxlabcl
              SRC mcxlabcl.cpp
              NO_IMPLICIT_LINK_TO_MATLAB_LIBRARIES
              LINK_TO ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY} mcxcl-matlab OpenCL::OpenCL
            )
    endif()

    target_compile_definitions(mcxlabcl PUBLIC MCX_CONTAINER MATLAB_MEX_FILE)

    set_target_properties(mcxlabcl
            PROPERTIES OUTPUT_NAME ${CMAKE_SOURCE_DIR}/../mcxlabcl/mcxcl)
    add_dependencies(mcxlabcl clheader)

endif()
