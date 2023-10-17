base:
	gcc -o canny canny.c util.c -Wall -lm



clean:
	rm -f *.o
	rm canny
