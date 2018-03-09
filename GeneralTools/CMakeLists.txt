#OCCA2 test by XL
#GeneralTools library
# -----------------------------------------------------------------------------
# ----------------------------- Project configuration -------------------------
# -----------------------------------------------------------------------------

# -- library Name name
SET(GeneralTools "${PROJECT_NAME}.GeneralTools")
SET(GeneralTools ${GeneralTools} PARENT_SCOPE)
# -- Include directories --
LIST(APPEND GeneralTools_LIB_INCLUDE_DIRS  ${CMAKE_CURRENT_SOURCE_DIR})
SET(GeneralTools_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
SET(GeneralTools_LIB_INCLUDE_DIR ${GeneralTools_LIB_INCLUDE_DIR} PARENT_SCOPE)
IF(TARGET ${GeneralTools})
  MESSAGE("${GeneralTools} is already defined")
ELSE()
  # -- Source files list ---
  COLLECT_SRC_FROM_DIRS(GeneralTools_LIB_INCLUDE_DIR)
  
  ADD_LIBRARY(${GeneralTools} ${SOURCE_FILES})
  # -- Target --
  TARGET_INCLUDE_DIRECTORIES(${GeneralTools} PUBLIC
    $<BUILD_INTERFACE:${GeneralToolsSrc_DIRS}>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/>  # <prefix>/include
  )
#  LIST(APPEND OCCA2LinkedLibraries )
#  LIST(REMOVE_DUPLICATES OCCA2LinkedLibraries )
    
#  TARGET_LINK_LIBRARIES(${GeneralTools} LINK_PUBLIC ${OCCA2LinkedLibraries} )
  

#  SET_TARGET_PROPERTIES(${GeneralTools} PROPERTIES COMPILE_DEFINITIONS "${DefinitionFlags}")
#  SET_PROPERTY(TARGET ${GeneralTools} PROPERTY CXX_STANDARD 98)
#  ADD_SUBDIRECTORY(TestCases)
#  ADD_SUBDIRECTORY(Validations)
ENDIF()