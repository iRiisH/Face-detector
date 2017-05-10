CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(FaceDetector)

ADD_EXECUTABLE(viola viola.cpp feature.cpp feature.h classifier.cpp classifier.h)
TARGET_LINK_LIBRARIES(viola ${OpenCV_LIBS})

