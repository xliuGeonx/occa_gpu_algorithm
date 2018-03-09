#OCCA2 test case SimpleKernel by XL

# -----------------------------------------------------------------------------
# ----------------------------- Project configuration -------------------------
# -----------------------------------------------------------------------------

# Environment
SET(OCCA2_Reduce_InnerProduct_exe OCCA2_Reduce_InnerProduct)
SET(OCCA2_Reduce_InnerProduct_exe ${OCCA2_Reduce_InnerProduct_exe} PARENT_SCOPE)

IF(TARGET ${OCCA2_Reduce_InnerProduct_exe})
  MESSAGE("${OCCA2_Reduce_InnerProduct_exe} is already defined.")
ELSE()
	#Add execute_process	
	LIST(APPEND OCCA2_Reduce_InnerProduct_exe_include_dir  ${CMAKE_CURRENT_SOURCE_DIR}/src)
	ADD_EXECUTABLE(${OCCA2_Reduce_InnerProduct_exe} ${OCCA2_Reduce_InnerProduct_exe_include_dir}/main.cpp)

	#Add include
	LIST(APPEND OCCA2_inlcude_dir ${OCCA_ROOT}/include)
	INCLUDE_DIRECTORIES( ${OCCA2_inlcude_dir} ${GeneralTools_LIB_INCLUDE_DIR})

	FOREACH( f ${USED_DLLS})
    ADD_CUSTOM_COMMAND(TARGET ${OCCA2_Reduce_InnerProduct_exe} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${f} $<TARGET_FILE_DIR:${OCCA2_Reduce_InnerProduct_exe}> )  
	ENDFOREACH()
#SET_TARGET_PROPERTIES(${OCCA2_Reduce_InnerProduct_exe} PROPERTIES LINKER_LANGUAGE CXX)

	#Add libs 
	LIST(APPEND OCCA2LinkedLibraries ${OCCA_LIBRARIES} ${GeneralTools})
	LIST(REMOVE_DUPLICATES OCCA2LinkedLibraries)
	TARGET_LINK_LIBRARIES(${OCCA2_Reduce_InnerProduct_exe} LINK_PUBLIC ${OCCA2LinkedLibraries})

ENDIF()