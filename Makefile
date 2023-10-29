base:
	gcc -o canny canny.c util.c -lm -Wall

opt_no_vec:
	gcc -o canny util.c canny.c -lm -O2 -Wall

opt_vec_show:
	gcc -o canny util.c canny.c -lm -O2 -ftree-vectorize -mavx2 -fopt-info-vec

opt_vec_mrel:
	gcc -o canny util.c canny.c -lm -O2 -ftree-vectorize -mavx2 -fopt-info-vec -ffast-math

fastest:
	gcc -o canny util.c canny.c -lm -O3 -ftree-vectorize -mavx2 -fopt-info-vec -ffast-math

opt_vec_mrel_openmp:
	gcc -o canny util.c canny.c -lm -O2 -ftree-vectorize -mavx2 -fopt-info-vec -ffast-math -fopenmp

support_opencl:
	gcc -o canny util.c opencl util.c canny.c -lm -O2 -ftree-vectorize -mavx2 -fopt-info-vec -ffast-math -fopenmp -lOpenCL

clean:
	@rm -f *.o
	@rm -f canny
