# This will define the following variables:
#   onnxruntime_FOUND        -- True if the system has the onnxruntime library
#   onnxruntime_INCLUDE_DIRS -- The include directories for onnxruntime
#   onnxruntime_LIBRARIES    -- Libraries to link against

include(FindPackageHandleStandardArgs)

find_library(cugraph_LIBRARY
    NAMES cugraph
    HINTS ${cugraph_DIR}
    PATH_SUFFIXES lib lib32 lib64
    DOC "The cuGraph library")

find_path(cugraph_INCLUDE_DIRS
    NAMES cugraph/graph.hpp
    PATH_SUFFIXES include
    HITS ${cugraph_DIR}
    DOC "The cuGraph include directory")

if (NOT cugraph_INCLUDE_DIRS)
    message(FATAL_ERROR "cuGraph includes not found")
endif()

find_package_handle_standard_args(cugraph
    REQUIRED_VARS cugraph_LIBRARY cugraph_INCLUDE_DIRS)


add_library(cugraph SHARED IMPORTED)
set_property(TARGET cugraph PROPERTY IMPORTED_LOCATION "${cugraph_LIBRARY}")
set_property(TARGET cugraph PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${cugraph_INCLUDE_DIRS}")

mark_as_advanced(cugraph_FOUND cugraph_INCLUDE_DIRS cugraph_LIBRARY)