# Conditional settings based on passed in variables
ifdef INTEL
    CC=icc
else
    CC=g++
endif

ifdef DEBUG
    CFLAGS=-c -g -fPIC -fpermissive -std=c++11
else
    CFLAGS=-c -fPIC -fpermissive -O3 -std=c++11
endif

LIBPATH = /home/vagrant/ss/sdk/RCM/StrategyStudio/libs/x64
INCLUDEPATH = /home/vagrant/ss/sdk/RCM/StrategyStudio/includes

INCLUDES=-I/usr/include -I$(INCLUDEPATH)
LDFLAGS=$(LIBPATH)/libstrategystudio_analytics.a \
		$(LIBPATH)/libstrategystudio.a \
		$(LIBPATH)/libstrategystudio_transport.a \
		$(LIBPATH)/libstrategystudio_marketmodels.a \
		$(LIBPATH)/libstrategystudio_utilities.a \
		$(LIBPATH)/libstrategystudio_flashprotocol.a

LIBRARY=strategy.so
SOURCES=strategy.cpp
HEADERS=strategy.h
 
OBJECTS=$(SOURCES:.cpp=.o)

all: $(HEADERS) $(LIBRARY)

$(LIBRARY) : $(OBJECTS)
	$(CC) -shared -Wl,-soname,$(LIBRARY).1 -o $(LIBRARY) $(OBJECTS) $(LDFLAGS)
	
.cpp.o: $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -rf *.o $(LIBRARY)
