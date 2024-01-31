CC = g++
CXX = $(CC)

CFLAGS = -Wc++11-extensions -std=c++11 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF $(shell pkg-config --cflags opencv4)
CXXFLAGS = $(CFLAGS)
LDLIBS = $(shell pkg-config --libs opencv4)

BINDIR = ../bin

baseline_match: baseline_match.o
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $^ $(LDLIBS)

clean:
	rm -f *.o *~