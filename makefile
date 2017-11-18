all: parallel 

parallel:
	nvcc -arch=sm_30 -o parallel.x main.cu ppmFile.c;

clean:
	rm parallel.x;
