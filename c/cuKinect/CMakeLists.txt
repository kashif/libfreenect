include_directories (${CMAKE_SOURCE_DIR}/include)
link_directories (${CMAKE_BINARY_DIR}/lib)

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(USB REQUIRED)
find_package(CUDA REQUIRED)

find_path(CUDA_CUT_INCLUDE_DIR
  cutil.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
)
find_path(CUDA_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")
mark_as_advanced(CUDA_CUT_INCLUDE_DIR)

if(CMAKE_HOST_WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(cuda_cutil_name cutil64)
  else(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(cuda_cutil_name cutil32)
  endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
elseif(CMAKE_HOST_UNIX)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(cuda_cutil_name cutil_x86_64)
  else()
    set(cuda_cutil_name cutil_i386)
  endif()
endif()

find_library(CUDA_CUT_LIBRARY
  NAMES cutil ${cuda_cutil_name}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/lib/darwin" "common/lib/linux" "common/lib" "lib"
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
)
find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
mark_as_advanced(CUDA_CUT_LIBRARY)

include_directories(
  ${USB_INCLUDE_DIRS}
  ${CUDA_CUT_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR} 
  ${GLUT_INCLUDE_DIR}
)

CUDA_ADD_EXECUTABLE (cuKinect
  cuKinect.cu
)

TARGET_LINK_LIBRARIES (cuKinect
  freenect
  ${CUDA_CUT_LIBRARY}
  ${GLUT_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)
