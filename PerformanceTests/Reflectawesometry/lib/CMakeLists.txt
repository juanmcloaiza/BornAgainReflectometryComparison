add_library(Reflectawesometry Reflectawesometry.h)
target_include_directories (Reflectawesometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_SWIG_FLAGS "")
set_property(SOURCE Reflectawesometry.i PROPERTY CPLUSPLUS ON)
#set_property(SOURCE Reflectawesometry.i PROPERTY SWIG_FLAGS "-includeall")
swig_add_library(Reflectawesometry LANGUAGE python SOURCES Reflectawesometry.i Reflectawesometry.h)
swig_link_libraries(Reflectawesometry ${PYTHON_LIBRARIES})

#target_link_libraries(Reflectawesometry
#    Qt5::Core
#    ${PYTHON_LIBRARIES}
#    )

#SWIG_LINK_LIBRARIES(ReflectawesometryM ${PYTHON_LIBRARIES})
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

#SET(CMAKE_SWIG_FLAGS "")

#SET_SOURCE_FILES_PROPERTIES(Reflectawesometry.i PROPERTIES CPLUSPLUS ON)
#SET_SOURCE_FILES_PROPERTIES(Reflectawesometry.i PROPERTIES SWIG_FLAGS "-includeall")
#SWIG_ADD_MODULE(Reflectawesometry python Reflectawesometry.i Reflectawesometry.h)
#SWIG_LINK_LIBRARIES(Reflectawesometry ${PYTHON_LIBRARIES})
