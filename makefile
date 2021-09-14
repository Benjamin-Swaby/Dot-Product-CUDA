

library_FLAGS = -pg -O2


src_DIR  = ./src
build_DIR = ./build

SRCS := $(wildcard $(src_DIR)/*.cu)

objects := $(SRCS:$(src_DIR)/%.cu=$(build_DIR)/%.o)


link: $(objects)
	nvcc $(library_FLAGS) $(objects) -o main -arch=compute_61

run: link
	./main	

clean:
	rm $(build_DIR)/*
	rm gmon.out
	rm main

$(build_DIR)/%.o: $(src_DIR)/%.cu
	nvcc -c $< -o $@ -O2 -arch=compute_61

