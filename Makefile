CC = g++

# Available flags = [CV, DIRECTED, DEBUG, VERIFY, VISUALIZE]


CFLAGS = -Wall -Wno-reorder -Wno-unused-variable  -O2 -std=c++20 -I utils -I queue -fopenmp


all: ogpart ogpartcv

ogpart: ogpart.o timer.o
	${CC} ${CFLAGS} -o ogpart ogpart.o refine.o timer.o

ogpartcv: ogpartcv.o timer.o
	${CC} ${CFLAGS} -DCV -o ogpartcv ogpart.o refine.o timer.o

ogpart.o:
	${CC} ${CFLAGS} -c partitioners/ogpart.cpp partitioners/refine.cpp

ogpartcv.o:
	${CC} ${CFLAGS} -DCV -c partitioners/ogpart.cpp partitioners/refine.cpp


timer.o:
	${CC} ${CFLAGS} -c ${wildcard utils/*.cpp}

clean:
	rm -rf *.o
