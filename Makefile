### Makefile

SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
### Define the list of executable names to be built from corresponding source files
EXECS :=  vecadd00 matmult00 vecadd01 matmult01
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
### Compiles the vecaddKernel00.cu to an object file, setting the number of values each thread computes
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

### Compiles and links the vecadd CUDA program with vecaddKernel00 object file and timer object file, producing the executable vecadd00
vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
### Compiles the vecaddKernel01.cu with optimized coalesced memory reads into an object file
vecaddKernel01.o : vecaddKernel01.cu vecaddKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS)

### Compiles and links the vecadd CUDA program with vecaddKernel01 object file and timer object file, creating the vecadd01 executable optimized for coalesced memory reads
vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
## Provided Kernel
### Compiles the provided matrix multiplication kernel (matmultKernel00.cu) into an object file
matmultKernel00.o : matmultKernel00.cu matmultKernel.h 
	${NVCC} $< -c -o $@ $(OPTIONS)

### Compiles and links the matmult CUDA program with the matmultKernel00 object file and timer object file, producing the matmult00 executable
matmult00 : matmult.cu  matmultKernel.h matmultKernel00.o timer.o
	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
## Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
### Compiles the expanded matrix multiplication kernel (matmultKernel01.cu) with a redefined FOOTPRINT_SIZE of 32, aiming for more optimized tiling
matmultKernel01.o : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

### Compiles and links the matmult CUDA program with the matmultKernel01 object file and timer object file, creating the matmult01 executable with an expanded footprint size for potentially improved performance
matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32
