/*
* Program that sum integers
*/

#include <stdio.h>
#include <stdlib.h>
#include "cputils.h"
#include <omp.h>

int main(int argc, char *argv[]){

	double start_time = cp_Wtime();

	int num = 0;
	int i;
    #pragma omp parallel for shared(argv), private(i), reduction(+:num)
	for(i=1; i<argc; i++){
		num += atoi(argv[i]);
	}

	double end_time = cp_Wtime();

	printf("Result: %d\n",num);
	printf("Time: %f\n",end_time-start_time);

	return EXIT_SUCCESS;
}