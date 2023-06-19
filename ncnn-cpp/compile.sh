
g++ -Wall -Wno-unknown-pragmas -isystem -fPIE -fopenmp -pthread -DNDEBUG -I/usr/local/include/opencv4 -I/usr/local/include/ncnn model.cpp main.cpp -fopenmp `pkg-config --libs --cflags opencv4` -ldl -lpthread -pthread -lgomp -DNDEBUG -rdynamic /usr/local/lib/ncnn/libncnn.a
