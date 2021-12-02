
CXX := /usr/local/bin/g++-10
CXXFLAGS := -MMD -MP
CXXFLAGS += -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup -fPIC $(shell python3 -m pybind11 --includes)

#Code execution options
CXXFLAGS += -DDOUBLE_PRECISION
CXXFLAGS += -DTIMING_INFO

SRCDIR := src
BUILDDIR := build
LIBNAME := Cvortrace
TARGET = $(LIBNAME)$(shell python3-config --extension-suffix)
 
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
DEP = $(OBJECTS:%.o=%.d)

LIB := -L /usr/local/lib -lhdf5 
INC := -I include
INC += -I /usr/local/include

CXXFLAGS += -fopenmp
LIB += -fopenmp

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CXX) $^ $(LIB) -o $(TARGET) "; $(CXX) $(CXXFLAGS) $^  $(LIB) -o $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<

-include $(DEP)

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

.PHONY: clean
