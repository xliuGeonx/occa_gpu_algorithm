# -- Find OCCA --
# Inspired from: 
# - http://www.itk.org/Wiki/CMake:How_To_Find_Libraries
# Find the native OCCA headers and libraries.
#
# Input parameters which could influence the search:
# - OCCA_ROOT - root folder where OCCA is installed
#
# Once done, this will define:
# - OCCA_INCLUDE_DIRS - where to find OCCA headers
# - OCCA_LIBRARIES - List of libraries when using OCCA.
# - OCCA_FOUND - True if OCCA found.

#

if(OCCA_INCLUDE_DIR AND OCCA_LIBRARY)
  set(OCCA_FIND_QUIETLY TRUE)
endif(OCCA_INCLUDE_DIR AND OCCA_LIBRARY)

include(LibFindMacros)

if( NOT OCCA_ROOT)
  set(OCCA_ROOT $ENV{OCCA_DIR})
endif( NOT OCCA_ROOT)
# Check input parameters
if(OCCA_ROOT)
  set( _OCCA_INCLUDE_SEARCH_DIRS
       ${OCCA_ROOT}/include ${OCCA_ROOT} ${_OCCA_INCLUDE_SEARCH_DIRS})
  set( _OCCA_LIBRARY_SEARCH_DIRS
       ${OCCA_ROOT}/lib)
endif(OCCA_ROOT)

# Look for the header file
find_path(OCCA_INCLUDE_DIR NAMES occa_c.h 
          PATHS ${_OCCA_INCLUDE_SEARCH_DIRS} ${OCCA_PKGCONF_INCLUDE_DIRS})

# Look for the occa library
find_library(OCCA_LIBRARY NAMES occa libocca
             PATHS ${_OCCA_LIBRARY_SEARCH_DIRS} ${OCCA_PKGCONF_LIBRARY_DIRS} NO_DEFAULT_PATH)
if(WIN32)
  find_library(OCCA_PTHREADVC_LIBRARY NAMES pthreadVC2 
             PATHS ${_OCCA_LIBRARY_SEARCH_DIRS} ${OCCA_PKGCONF_LIBRARY_DIRS} NO_DEFAULT_PATH)
endif()

# Set the include dir variables and the libraries and let libfind_process do the rest
set(OCCA_PROCESS_INCLUDES OCCA_INCLUDE_DIR)
set(OCCA_PROCESS_LIBS OCCA_LIBRARY)
if(WIN32)
  set(OCCA_PROCESS_LIBS ${OCCA_PROCESS_LIBS} OCCA_PTHREADVC_LIBRARY)
endif()
# handle the QUIETLY and REQUIRED arguments and set OCCA_FOUND
libfind_process(OCCA)
set(OCCA_LIBRARIES)
set(OCCA_INCLUDE_DIRS)
if(OCCA_FOUND)
  find_package(CUDA)
  find_package(OpenCL)  
  if(CUDA_FOUND AND OpenCL_FOUND)
	  set(OCCA_LIBRARIES ${CUDA_LIBRARIES} ${OpenCL_LIBRARIES})
	  set(OCCA_INCLUDE_DIRS  ${CUDA_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
	  set (CUDA_LIBRARY_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
	  find_library(CUDA_LIB NAMES cuda cudart PATHS ${CUDA_LIBRARY_DIRS})
	  set(OCCA_LIBRARIES ${CUDA_LIB} ${OpenCL_LIBRARIES})
	  set(OCCA_LIBRARY_DIRS ${OCCA_LIBRARY_DIRS} ${CUDA_LIBRARY_DIRS} )
  endif()
endif()

set(OCCA_LIBRARIES ${OCCA_LIBRARIES} ${OCCA_LIBRARY} ${OCCA_PTHREADVC_LIBRARY}   )
set(OCCA_INCLUDE_DIRS ${OCCA_INCLUDE_DIR} ${OCCA_INCLUDE_DIRS})
mark_as_advanced(OCCA_INCLUDE_DIRS OCCA_LIBRARIES )
