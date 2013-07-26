all: onePAE
CXX = g++
CFLAGS = -Wall -Wconversion -O3 -fPIC

fileInt.o: fileInt.cpp 
	$(CXX) $(CFLAGS)  -c -o fileInt.o fileInt.cpp

getRP.o: getRP.cpp
	$(CXX) $(CFLAGS)  -c -o getRP.o getRP.cpp 

onePdRS.o: onePdRS.cpp
	$(CXX) $(CFLAGS)  -c -o onePdRS.o onePdRS.cpp 

svmSolver.o: svmSolver.cpp
	$(CXX) $(CFLAGS)  -c -o svmSolver.o svmSolver.cpp 

onePAE: onePAE.cpp fileInt.o getRP.o onePdRS.o svmSolver.o 
	$(CXX) $(CFLAGS)  -o onePAE onePAE.cpp fileInt.o getRP.o onePdRS.o svmSolver.o 

clean:
	rm -f *~ fileInt.o getRP.o onePdRS.o svmSolver.o onePAE
