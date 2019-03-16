SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

# Compilers
CC=gcc
NVCC=nvcc
MPICC=mpicc

# Compilation Flags
CFLAGS=-O3 -I$(HEADER_DIR)
OMPFLAGS=-fopenmp

# Linking Flags
CUDA_LINK_FLAGS=-L/usr/local/cuda/lib64/ -lcuda -lcudart
OMP_LINK_FLAGS=-lgomp
LDFLAGS=-lm $(OMP_LINK_FLAGS) $(CUDA_LINK_FLAGS)

# Not needed, we link with mpicc and compile main with mpicc as well
MPIFLAGS=$(shell mpicc --showme:compile)
MPI_LINK_FLAGS=$(shell mpicc --showme:link)

SRC= dgif_lib.c \
	egif_lib.c \
	gif_err.c \
	gif_font.c \
	gif_hash.c \
	gifalloc.c \
	openbsd-reallocarray.c \
	quantize.c \
	gif_io.c \
	main.c \
	cuda_filters.cu



OBJ= $(OBJ_DIR)/dgif_lib.o \
	$(OBJ_DIR)/egif_lib.o \
	$(OBJ_DIR)/gif_err.o \
	$(OBJ_DIR)/gif_font.o \
	$(OBJ_DIR)/gif_hash.o \
	$(OBJ_DIR)/gifalloc.o \
	$(OBJ_DIR)/openbsd-reallocarray.o \
	$(OBJ_DIR)/quantize.o \
	$(OBJ_DIR)/gif_io.o \
	$(OBJ_DIR)/cuda_filters.o \
	$(OBJ_DIR)/main.o

all: $(OBJ_DIR) sobelf

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o:$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

$(OBJ_DIR)/cuda_filters.o:$(SRC_DIR)/cuda_filters.cu
	$(NVCC) $(CFLAGS) -c -o $@ $^

$(OBJ_DIR)/main.o:$(SRC_DIR)/main.c
	$(MPICC) $(CFLAGS) -c -o $@ $^ $(OMPFLAGS)

sobelf:$(OBJ)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f sobelf $(OBJ)
	rm -f images/processed/*
