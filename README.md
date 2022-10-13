# Paralela-2022
Ejercicios de clase. Computación Paralela y Distribuida 2022-2. Universidad Nacional de Colombia

# BlurVideo
Instalar OpenCV4 - https://www.geeksforgeeks.org/how-to-install-opencv-in-c-on-linux/

Arreglar problema: Inappropriate ioctl for device - https://www.edureka.co/community/52428/opencv-unable-stop-the-stream-inappropriate-ioctl-for-device

Montar proyecto en Visual Studio Code - https://youtu.be/m9HBM1m_EMU

Cargar y buildear el proyecto preferiblemente usando la extensión CMake de VSCode.

Ejecución del proyecto: ./build/VideoBlur ./resources/videoIn.mp4 ./resources/videoOut.avi THREADS_NUM

# Build OpenCV
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_FFMPEG=ON -D WITH_TBB=ON -D WITH_GTK=ON -D WITH_V4L=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -DWITH_QT=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D WITH_OPENMP=ON ../opencv
