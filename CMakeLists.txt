#OCCA2 test case by Xin

PROJECT (OCCA2_Test)
Cmake_minimum_required (VERSION 3.0)

FUNCTION(COLLECT_SRC_FROM_DIRS DIRS)
  FOREACH(folder ${${DIRS}})
    FILE(
      GLOB
      source_files_one_dir
      ${folder}/[^.]*.h      
      ${folder}/[^.]*.hxx          
      ${folder}/[^.]*.ipp  
      ${folder}/[^.]*.hpp  
      ${folder}/[^.]*.hh
      ${folder}/[^.]*.cc      
      ${folder}/[^.]*.c
      ${folder}/[^.]*.cxx
      ${folder}/[^.]*.cpp
      ${folder}/[^.]*.occa
    )
    LIST(APPEND SOURCE_FILES_LIST ${source_files_one_dir})
  ENDFOREACH()
  SET(SOURCE_FILES ${SOURCE_FILES_LIST} PARENT_SCOPE)
ENDFUNCTION()# 
# -----------------------------------------------------------------------------
FUNCTION(COLLECT_DDL_FROM_FILES  files_list) 
  FOREACH(f ${${files_list}})
    GET_FILENAME_COMPONENT(f_ext ${f} EXT )
    IF(f_ext)
      IF(${f_ext} STREQUAL ".so" )
        IF(NOT WIN32)
          LIST(APPEND dlls_list ${f})
        ENDIF()
      ELSEIF(${f_ext} STREQUAL ".dll")
        IF(WIN32)
          LIST(APPEND dlls_list ${f})
        ENDIF()
      ELSEIF(${f_ext} STREQUAL ".lib")
        IF(WIN32)        
          GET_FILENAME_COMPONENT(f_dir ${f} DIRECTORY )
          GET_FILENAME_COMPONENT(f_without_ext ${f} NAME_WE )
          SET(f_dd "${f_dir}/${f_without_ext}.dll")
          IF(EXISTS "${f_dd}")          
            LIST(APPEND dlls_list ${f_dd})
          ENDIF()
        ENDIF()
      ENDIF()
    ENDIF()
  ENDFOREACH()
  SET(USED_DLLS ${USED_DLLS} ${dlls_list} PARENT_SCOPE)
ENDFUNCTION()
#---------------------------------------------------------------------------------
LIST(APPEND USED_DLLS ${OCCA_ROOT}/lib/pthreadVC2.dll)
# Additional folders to search for CMake modules
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# find Boost
SET(Boost_NO_SYSTEM_PATHS true)
SET (Boost_USE_STATIC_LIBS ON CACHE BOOL "use static libraries from Boost")
SET(Boost_USE_MULTITHREAD ON) #SET a flag
SET(BOOST_LIB_LIST)
IF(NOT WIN32)
  #  LIST(APPEND BOOST_LIB_LIST program_options log log_setup thread regex mpi serialization iostreams filesystem system)
   LIST(APPEND BOOST_LIB_LIST program_options log log_setup thread regex serialization iostreams filesystem system locale)
  FIND_PACKAGE(ICU REQUIRED COMPONENTS i18n uc)
  INCLUDE_DIRECTORIES(${ICU_INCLUDE_DIRS})
  LINK_DIRECTORIES(${ICU_LIBRARY_DIRS})
  COLLECT_DDL_FROM_FILES(ICU_LIBRARIES)
  FIND_PACKAGE(ZLIB REQUIRED)
  INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIRS})
  LINK_DIRECTORIES(${ZLIB_LIBRARY_DIRS})
  COLLECT_DDL_FROM_FILES(ZLIB_LIBRARIES)
ELSE()
  #  LIST(APPEND BOOST_LIB_LIST regex mpi serialization iostreams system filesystem )
  LIST(APPEND BOOST_LIB_LIST regex serialization iostreams system filesystem )
ENDIF()
FIND_PACKAGE(Boost REQUIRED COMPONENTS ${BOOST_LIB_LIST})
INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
SET(BOOST_DLLS)
IF(Boost_USE_STATIC_LIBS)
FOREACH(l ${BOOST_LIB_LIST})
  STRING(TOUPPER ${l} v)
  LIST(APPEND BOOST_DLLS ${Boost_${v}_LIBRARY_RELEASE})
ENDFOREACH()
COLLECT_DDL_FROM_FILES(BOOST_DLLS)
ENDIF()

#find OCCA
FIND_PACKAGE(OCCA REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${OCCA_INCLUDE_DIRS})
LINK_DIRECTORIES(${OCCA_LIBRARY_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
SET(OCCA_KERNELS_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/OccaKernels)

IF(WIN32)
 COLLECT_DDL_FROM_FILES(OCCA_PTHREADVC_LIBRARY)
ENDIF()

#-----------

LIST (APPEND Tests_dir GeneralTools SimpleKernel Reduce_InnerProduct Scan_add Histogram Compact Compact_inverse Sort_Bitonic Mthreads)
LIST (APPEND Tests_dir Mthreads_Serial Mprocess)
FOREACH (T_dir ${Tests_dir})
	ADD_SUBDIRECTORY( ${T_dir})
ENDFOREACH()