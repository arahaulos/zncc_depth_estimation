CXX = g++

CXXFLAGS = -Wall -std=c++20 -march=native -O3 -s
LDFLAGS = -static-libstdc++ -static-libgcc


LIBS = OpenCL
SRCS = $(wildcard *.cpp) $(wildcard lodepng/*.cpp)
OBJS = $(patsubst %.cpp, obj/%.o, $(SRCS))

TARGET = bin/zncc

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)

	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS) -l $(LIBS)


obj/%.o: %.cpp
	@mkdir -p $(dir $@)

	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf obj
