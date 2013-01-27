#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 2.8)
#Name your project here
project(SpecVis)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#find the Qt
find_package(Qt4 REQUIRED)
include_directories(${QT_INCLUDE_DIRECTORY})
include(${QT_USE_FILE})

#set up CUDA
find_package(CUDA)

#find OpenGL
find_package(OpenGL REQUIRED)

#find GLUT
set(GLUT_ROOT_PATH $ENV{GLUT_ROOT_PATH})
find_package(GLUT REQUIRED)

#find GLEW
find_package(GLEW REQUIRED)

#add Qt OpenGL stuff
set(QT_USE_QTOPENGL TRUE)

#ask the user for the RTS location
set(RTS_ROOT_PATH $ENV{RTS_ROOT_PATH})
find_package(RTS REQUIRED)

#set the include directories
include_directories(
	${CMAKE_CURRENT_BINARY_DIR}
	${QT_INCLUDES}
	${QT_QTOPENGL_INCLUDE_DIR}
	${OPENGL_INCLUDE_DIR}
	${GLEW_INCLUDE_PATH}
	${GLUT_INCLUDE_DIR}
	${RTS_INCLUDE_DIR}
)

#Assign source files to the appropriate variables
file(GLOB SRC_CPP "*.cpp")
file(GLOB SRC_H "*.h")
file(GLOB MOC_H "qt*.h")
file(GLOB SRC_UI "*.ui")
file(GLOB SRC_QRC "*.qrc")
file(GLOB SRC_CU "*.cu")

#determine which source files have to be moc'd
Qt4_wrap_cpp(UI_MOC ${MOC_H})
Qt4_wrap_ui(UI_H ${SRC_UI})
Qt4_add_resources(ALL_RCC ${ALL_QRC})

#moc the necessary files
Qt4_automoc(${ALL_CPP})

source_group(QtMoc FILES ${UI_MOC})
source_group(QtUI FILES ${SRC_UI})

#create an executable
cuda_add_executable(SpecVis ${SRC_CPP} ${SRC_H} ${UI_H} ${UI_MOC} ${ALL_RCC} ${SRC_CU})

#set the link libraries
target_link_libraries(SpecVis ${QT_LIBRARIES} ${QT_QTOPENGL_LIBRARY} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${GLEW_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cufft_LIBRARY} ${GLUT_glut_LIBRARY})



