CXX = clang++

CXXFLAGS += -c -Wall -std=c++14 $(shell pkg-config --cflags opencv)
LDFLAGS += -lboost_filesystem -lboost_system $(shell pkg-config --libs --static opencv)

all: cbox_finder

cbox_finder: cbox_finder.o; $(CXX) $< -o $@ $(LDFLAGS)

%.o: %.cpp; $(CXX) $< -o $@ $(CXXFLAGS)

clean: ; rm -f cbox_finder.o cbox_finder
