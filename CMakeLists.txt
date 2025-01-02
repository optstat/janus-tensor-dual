cmake_minimum_required(VERSION 3.10)
project(JanusTensorDual VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Enable CMP0077 policy for consistent variable handling
if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

include(ExternalProject)
include(ProcessorCount)

# Define paths
set(EXTERNAL_PROJECT_DIR "${CMAKE_BINARY_DIR}/external")
set(LIBTORCH_INSTALL_DIR "${CMAKE_BINARY_DIR}/libtorch")

# Get the number of available processors
ProcessorCount(NUM_CPUS)
if(NOT NUM_CPUS OR NUM_CPUS EQUAL 0)
    set(NUM_CPUS 1)  # Fallback to 1 if the processor count cannot be determined
endif()

# Add pthread
find_package(Threads REQUIRED)

# Option to enable CUDA for LibTorch
option(USE_CUDA "Enable CUDA support for LibTorch" ON)
set(LIBTORCH_VERSION 2.5.0)
if(USE_CUDA)
    set(LIBTORCH_DOWNLOAD_URL https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu118.zip)
else()
    set(LIBTORCH_DOWNLOAD_URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}.zip)
endif()

# ExternalProject to download and extract LibTorch
ExternalProject_Add(
    libtorch_project
    PREFIX ${CMAKE_BINARY_DIR}/libtorch_download
    URL ${LIBTORCH_DOWNLOAD_URL}
    DOWNLOAD_COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/downloads &&
                     ${CMAKE_COMMAND} -E echo "Downloading LibTorch to ${CMAKE_BINARY_DIR}/downloads/libtorch.zip..." &&
                     curl -L -o ${CMAKE_BINARY_DIR}/downloads/libtorch.zip ${LIBTORCH_DOWNLOAD_URL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBTORCH_INSTALL_DIR} &&
                   ${CMAKE_COMMAND} -E echo_append "Installing LibTorch..." &&
                   unzip -q ${CMAKE_BINARY_DIR}/downloads/libtorch.zip -d ${LIBTORCH_INSTALL_DIR}

)

# Clone pybind11 submodule
ExternalProject_Add(
    pybind11_project
    PREFIX ${CMAKE_BINARY_DIR}/pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG master
    UPDATE_COMMAND ""  # Disable update to avoid network issues during rebuilds
    CONFIGURE_COMMAND ""  # pybind11 doesn't require a configure step
    BUILD_COMMAND ""  # No build step needed for pybind11
    INSTALL_COMMAND ""  # No install step needed for pybind11
)

# Clone bats-core submodule
ExternalProject_Add(
    bats_project
    PREFIX ${CMAKE_BINARY_DIR}/bats
    GIT_REPOSITORY https://github.com/bats-core/bats-core.git
    GIT_TAG master
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

# Add the paths to your TACO project
set(PYBIND11_DIR ${CMAKE_BINARY_DIR}/pybind11/src/pybind11_project)
set(BATS_DIR ${CMAKE_BINARY_DIR}/bats/src/bats_project)

# Pass dependency paths to TACO
add_definitions(-DPYBIND11_DIR=${PYBIND11_DIR} -DBATS_DIR=${BATS_DIR} )

# Add TACO subdirectory
add_subdirectory(third_party/taco)

# Ensure dependencies are built before TACO
add_dependencies(taco pybind11_project bats_project)

# Include directories (if required by TACO)
include_directories(${PYBIND11_DIR} ${BATS_DIR})
# Set options for TACO
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(OPENMP ON CACHE BOOL "Enable OpenMP")


# Add subdirectory for tests
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/cpp)
    add_subdirectory(tests/cpp)

    # Check if the `tests` target exists before adding dependencies
    if(TARGET tests)
        add_dependencies(tests taco_build libtorch_project)
    endif()
else()
    message(WARNING "Tests directory 'tests/cpp' does not exist. Skipping.")
endif()



# Install headers only when `make install` is run
set(HEADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/cpp")
install(DIRECTORY ${HEADER_SOURCE_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/janus
        FILES_MATCHING PATTERN "*.h"
)

# Define the location of the test binary
set(TEST_BINARY "${CMAKE_BINARY_DIR}/bin/dualtest")

# Add a custom target to run the tests
add_custom_target(
    test
    COMMAND ${TEST_BINARY}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests with ./bin/dualtest"
    VERBATIM
)

# Ensure the `test` target depends on building the `dualtest` binary
if(TARGET dualtest)
    add_dependencies(test dualtest)
else()
    message(WARNING "Target 'dualtest' does not exist. Ensure you define it properly.")
endif()