CXX := g++
CC := gcc
LINK := g++ #-fPIC
NVCC := nvcc
#Make sure you have given the Library path appropriately and modify it according to your library path 
INCLUDES = -I. -I./src/ -I./ext_src/ -I./inc/  
INCLUDES += -I/home/usr/Library/CGAL-4.9/include -I/home/usr/Library/gmp-6.1.2/include -I/home/usr/Library/mpfr-3.1.5/include -I/home/usr/Library/netcdf-cxx/4.2/include -I/home/usr/Library/eigen-git-mirror -I/home/usr/Library/netcdf-c/4.7.2/include  
LIB_CUDA = -lcuda -lcudart
LIB_CGAL += -L/home/usr/Library/gmp-6.1.2/lib -L/home/usr/Library/mpfr-3.1.5/lib 
LIB_CGAL += -L/home/usr/Library/CGAL-4.9/build/lib -lCGAL -lCGAL_Core -lgmp -lmpfr
LIB_NETCDF = -lnetcdf -lnetcdf_c++ -L/home/usr/Library/netcdf-cxx/4.2/lib -L/home/usr/Library/netcdf-c/4.7.2/lib 

#common flags
COMMONFLAGS += $(INCLUDES) -DCGAL_DISABLE_ROUNDING_MATH_CHECK -O3 -D_GLIBCXX_USE_CXX11_ABI=0
NVCCFLAGS += -D_FORCE_INLINES $(COMMONFLAGS) -Wno-deprecated-gpu-targets -arch=sm_70 #-Xptxas -fmad=false#-O0#-dlcm=ca#-G
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -w -frounding-math
CFLAGS += $(COMMONFLAGS) -frounding-math

CUOBJ_DIR=obj/cuobj
MODULES = databases models updaters utility
INCLUDES += -I./inc/databases -I./inc/models -I./inc/updaters -I./inc/utility -I./inc/analysis
OBJ_DIR=obj
SRC_DIR=src
BIN_DIR=.

.SECONDARY:

PROGS := $(wildcard *.cpp)
PROG_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.main.o,$(PROGS))
PROG_MAINS := $(patsubst %.cpp,$(BIN_DIR)/%.out,$(PROGS))


CPP_FILES := $(wildcard src/*/*.cpp)
CPP_FILES += $(wildcard src/*.cpp)
CU_FILES := $(wildcard src/*/*.cu)
CU_FILES += $(wildcard src/*.cu)

CLASS_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_FILES))


#cuda objects
$(OBJ_DIR)/%.cu.o : $(SRC_DIR)/%.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA)  -o $@ -c $<

#cpp class objects
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<

#program objects
$(OBJ_DIR)/%.main.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<

#Programs
%.out: $(OBJ_DIR)/%.main.o $(CLASS_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+

#target rules

all:build

float: CXXFLAGS += -DSCALARFLOAT
float: NVCCFLAGS += -DSCALARFLOAT
float: build

debug: CXXFLAGS += -g -DCUDATHREADSYNC -DDEBUGFLAGUP
debug: NVCCFLAGS += -g -lineinfo -Xptxas --generate-line-info -DDEBUGFLAGUP # -G note that in debug mode noise will always be reproducible
debug: build
build: $(CLASS_OBJS) $(CU_OBJS) $(PROG_MAINS)  $(PROGS)

clean: 
	rm -f $(PROG_OBJS) $(CLASS_OBJS) $(CU_OBJS) $(PROG_OBJS) $(PROG_MAINS)

print-%  : ; @echo $* = $($*)

