CC = g++
CXX = $(CC)

CFLAGS = -Wc++11-extensions -std=c++11 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF $(shell pkg-config --cflags opencv4)
CXXFLAGS = $(CFLAGS)
LDLIBS = $(shell pkg-config --libs opencv4)

BINDIR = ../bin

clean:
	rm -f *.o *~