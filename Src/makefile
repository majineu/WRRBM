CC=g++
#CFLAGS=-c -Wall -g -std=c++11	 
CFLAGS=-c -Wall -O4 -std=c++11 
LDFLAGS= -lpthread
SOURCES=main.cpp Config.cpp ReTrain.cpp DeepEmbedding.cpp CDLearner.cpp WRDBNTrainer.cpp RBM.cpp WRDBN.cpp WRRBM.cpp Pool.cpp util.cpp WRRBMTrainer.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=WRDBN

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o WRDBN
