GCC = g++
GCC_FLAGS = -Ofast -faggressive-loop-optimizations
OBJ_DIR = bin
INCL_DIR = include 
SRC_DIR = src
OBJ = $(OBJ_DIR)/main.o
EXE = gpuHeap.exe

default: $(EXE)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(GCC) -I $(INCL_DIR) $(GCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(GCC) -I $(INCL_DIR) $(GCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	del /f /q $(OBJ_DIR)\*.o $(EXE)
