OBJS = sample.cpp
OBJ_NAME = sample

all : $(OBJS)
		g++ $(OBJS) -std=c++14 -Wall -o $(OBJ_NAME)
