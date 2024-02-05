CC = g++
CXX = $(CC)

CFLAGS = -Wc++11-extensions -std=c++11 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF $(shell pkg-config --cflags opencv4)
CXXFLAGS = $(CFLAGS)
LDLIBS = $(shell pkg-config --libs opencv4)

BINDIR = ../bin

baseline_match: baseline_match.o feature_utils.o
	$(CXX) $^ -o $(BINDIR)/$@.exe $(LDLIBS)

baseline_match_1: baseline_match_1.o feature_utils.o
	$(CXX) $^ -o $(BINDIR)/$@.exe $(LDLIBS)

feature_extract: feature_extract.o feature_utils.o csv_util.o
	$(CXX) $^ -o $(BINDIR)/$@.exe $(LDLIBS)

clean:
	rm -f *.o *~