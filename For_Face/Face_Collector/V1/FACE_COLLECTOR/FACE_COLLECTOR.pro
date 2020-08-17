#-------------------------------------------------
#
# Project created by QtCreator 2020-06-29T07:19:53
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FACE_COLLECTOR
TEMPLATE = app

CONFIG += c++17

INCLUDEPATH += /usr/local/include/opencv /usr/local/include \
/usr/local/zed/include/ \
/usr/local/cuda-10.2/targets/x86_64-linux/include/

LIBS += -L/usr/local/lib -lopencv_cudabgsegm -lopencv_cudaobjdetect \
-lopencv_cudastereo -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d \
-lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow \
-lopencv_cudalegacy -lopencv_cudawarping -lopencv_aruco -lopencv_bgsegm \
-lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dnn_objdetect \
-lopencv_dpm -lopencv_highgui -lopencv_videoio -lopencv_face -lopencv_freetype \
-lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow \
-lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light \
-lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets \
-lopencv_text -lopencv_dnn -lopencv_video -lopencv_plot -lopencv_ml -lopencv_ximgproc \
-lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d \
-lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_cudaimgproc -lopencv_cudafilters \
-lopencv_imgproc -lopencv_cudaarithm -lopencv_core -lopencv_cudev \
-L/usr/local/zed/lib/ -lsl_ai -lsl_zed \
-L/usr/local/cuda-10.2/targets/x86_64-linux/lib/ -lcudart

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        dialog.cpp \
    cameramng.cpp \
    capturegamedialog.cpp \
    facelandmarker.cpp \
    headinggame.cpp \
    imagesave.cpp

HEADERS += \
        dialog.h \
    cameramng.h \
    capturegamedialog.h \
    facelandmarker.h \
    headinggame.h \
    imagesave.h

FORMS += \
        dialog.ui \
    capturegamedialog.ui
