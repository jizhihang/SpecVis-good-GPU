# Tries to find the RTS include directory

  FIND_PATH( RTS_INCLUDE_DIR NAMES rts_glShaderProgram.h
    PATHS
	${CMAKE_CURRENT_SOURCE_DIR}/rts
	${RTS_ROOT_PATH}
)
 
IF (RTS_FOUND)    
  #The following deprecated settings are for backwards compatibility with CMake1.4
  SET (RTS_INCLUDE_PATH ${RTS_INCLUDE_DIR})
ENDIF(RTS_FOUND)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(RTS REQUIRED_VARS TRUE RTS_INCLUDE_DIR)
