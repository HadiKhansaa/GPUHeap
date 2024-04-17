NVCC        = nvcc
NVCC_FLAGS  = -O3
INCL_DIR    = header
OBJ         = heapInsertKernelTest.o
EXE         = heap

default: $(EXE)

%.o: %.cu
	$(NVCC) -I $(INCL_DIR) $(NVCC_FLAGS) -c -o $@ $<

device_link.o: $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $@

clean:
	del /Q $(OBJ) $(EXE).exe $(EXE).exp $(EXE).lib
