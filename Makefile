CC = g++

# Available flags = [CV, DIRECTED, DEBUG, VERIFY, VISUALIZE]


CFLAGS = -Wall -Wno-reorder -Wno-unused-variable  -O2 -std=c++20 -I utils -I queue -fopenmp


all: ogpart ogpartcv ogpart_random ogpart_nodegree
ogpart_single_thread_test: ogpartsinglethread

ogpart: ogpart.o timer.o
	${CC} ${CFLAGS} -o ogpart ogpart.o refine.o timer.o

ogpartcv: ogpartcv.o timer.o
	${CC} ${CFLAGS} -DCV -o ogpartcv ogpart.o refine.o timer.o

ogpart_nodegree: ogpart_nodegree.o timer.o
	${CC} ${CFLAGS} -o ogpart_nodegree ogpart_nodegree.o refine.o timer.o

ogpart_random: ogpart_random.o timer.o
	${CC} ${CFLAGS} -o ogpart_random ogpart_random.o refine.o timer.o


ogpartsinglethread: ogpart_single_thread.o timer.o
	${CC} ${CFLAGS} -DCV -o ogpart ogpart_single_thread.o refine_single_thread.o timer.o

ogpart.o:
	${CC} ${CFLAGS} -c partitioners/ogpart.cpp partitioners/refine.cpp

ogpart_nodegree.o:
	${CC} ${CFLAGS} -c partitioners/ogpart_nodegree.cpp partitioners/refine.cpp

ogpart_random.o:
	${CC} ${CFLAGS} -c partitioners/ogpart_random.cpp partitioners/refine.cpp

ogpartcv.o:
	${CC} ${CFLAGS} -DCV -c partitioners/ogpart.cpp partitioners/refine.cpp


ogpart_single_thread.o:
	${CC} ${CFLAGS} -c partitioners/ogpart_single_thread.cpp partitioners/refine_single_thread.cpp

timer.o:
	${CC} ${CFLAGS} -c ${wildcard utils/*.cpp}

clean:
	rm -rf *.o
