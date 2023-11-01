CC = gcc
OPENCL_INCLUDE_PATH = /opt/AMDAPP/include
OPENCL_LIB_PATH = /opt/AMDAPP/lib/x86_64

vectoradd: vectoradd.c
	$(CC) -o vectoradd vectoradd.c -I${OPENCL_INCLUDE_PATH} -L${OPENCL_LIB_PATH} -lOpenCL
