/*
 * Simulacion simplificada de bombardeo de particulas de alta energia
 *
 * Computacion Paralela (Grado en Informatica)
 * 2017/2018
 *
 * (c) 2018 Arturo Gonzalez Escribano
 * Version: 2.0 (Atenuacion no lineal)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cputils.h>
#include <mpi.h>

#define PI 3.14159f
#define UMBRAL 0.001f
#define ROOT_RANK 0

/* Estructura para almacenar los datos de una tormenta de particulas */
typedef struct
{
	int size;
	int *posval;
} Storm;

/* FUNCIONES AUXILIARES: No se utilizan dentro de la medida de tiempo, dejar como estan */
/* Funcion de DEBUG: Imprimir el estado de la capa */
void debug_print(int layer_size, float *layer, int *posiciones, float *maximos, int num_storms)
{
	int i, k;
	if (layer_size <= 35)
	{
		/* Recorrer capa */
		for (k = 0; k < layer_size; k++)
		{
			/* Escribir valor del punto */
			printf("%10.4f |", layer[k]);

			/* Calcular el numero de caracteres normalizado con el maximo a 60 */
			int ticks = (int)(60 * layer[k] / maximos[num_storms - 1]);

			/* Escribir todos los caracteres menos el ultimo */
			for (i = 0; i < ticks - 1; i++)
				printf("o");

			/* Para maximos locales escribir ultimo caracter especial */
			if (k > 0 && k < layer_size - 1 && layer[k] > layer[k - 1] && layer[k] > layer[k + 1])
				printf("x");
			else
				printf("o");

			/* Si el punto es uno de los maximos especiales, annadir marca */
			for (i = 0; i < num_storms; i++)
				if (posiciones[i] == k)
					printf(" M%d", i);

			/* Fin de linea */
			printf("\n");
		}
	}
}

/*
 * Funcion: Lectura de fichero con datos de tormenta de particulas
 */
Storm read_storm_file(char *fname)
{
	FILE *fstorm = cp_abrir_fichero(fname);
	if (fstorm == NULL)
	{
		fprintf(stderr, "Error: Opening storm file %s\n", fname);
		exit(EXIT_FAILURE);
	}

	Storm storm;
	int ok = fscanf(fstorm, "%d", &(storm.size));
	if (ok != 1)
	{
		fprintf(stderr, "Error: Reading size of storm file %s\n", fname);
		exit(EXIT_FAILURE);
	}

	storm.posval = (int *)malloc(sizeof(int) * storm.size * 2);
	if (storm.posval == NULL)
	{
		fprintf(stderr, "Error: Allocating memory for storm file %s, with size %d\n", fname, storm.size);
		exit(EXIT_FAILURE);
	}

	int elem;
	for (elem = 0; elem < storm.size; elem++)
	{
		ok = fscanf(fstorm, "%d %d\n",
					&(storm.posval[elem * 2]),
					&(storm.posval[elem * 2 + 1]));
		if (ok != 2)
		{
			fprintf(stderr, "Error: Reading element %d in storm file %s\n", elem, fname);
			exit(EXIT_FAILURE);
		}
	}
	fclose(fstorm);

	return storm;
}

/*
 * PROGRAMA PRINCIPAL
 */
int main(int argc, char *argv[])
{
	int i, j, k;
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	/* 1.1. Leer argumentos */
	if (argc < 3)
	{
		if (rank == ROOT_RANK)
			fprintf(stderr, "Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0]);
		exit(EXIT_FAILURE);
	}

	int layer_size = atoi(argv[1]);
	int num_storms = argc - 2;
	Storm storms[num_storms];

	/* 1.2. Leer datos de storms */
	for (i = 2; i < argc; i++)
		storms[i - 2] = read_storm_file(argv[i]);

	/* 1.3. Inicializar maximos a cero */
	float maximos[num_storms];
	int posiciones[num_storms];
	for (i = 0; i < num_storms; i++)
	{
		maximos[i] = 0.0f;
		posiciones[i] = 0;
	}

	/* 2. Inicia medida de tiempo */
	MPI_Barrier(MPI_COMM_WORLD);
	double ttotal = cp_Wtime();

	/* COMIENZO: No optimizar/paralelizar el main por encima de este punto */
	//float *layer = (float *)malloc(sizeof(float) * layer_size);

	/*---------------------------------------------------------------------*/

	int local_layer_size = layer_size / size;
	if (rank < layer_size % size)
		local_layer_size += 1;

	int desplazamiento = (rank > layer_size % size) ? rank * (layer_size / size) + layer_size % size : rank * (layer_size / size) + rank;
	// if(rank > layer_size%size)
	// 	desplazamiento += layer_size%size;
	// else
	// 	desplazamiento+=rank;

	float *miniLayer = (float *)malloc(sizeof(float) * local_layer_size);
	// if (layer == NULL)
	// {
	// 	fprintf(stderr, "Error: Allocating the layer memory\n");
	// 	exit(EXIT_FAILURE);
	// }
	/* 3. Reservar memoria para las capas e inicializar a cero */
	float *layer_copy = (float *)malloc(sizeof(float) * local_layer_size);

	if (miniLayer == NULL || layer_copy == NULL)
	{
		fprintf(stderr, "Error: Allocating the layer memory\n");
		exit(EXIT_FAILURE);
	}

	for (k = 0; k < local_layer_size; k++)
		miniLayer[k] = 0.0f;
	for (k = 0; k < local_layer_size; k++)
		layer_copy[k] = 0.0f;

	float *raiz = (float *)malloc(sizeof(float) * layer_size);

	int inicio;
	int final;
	float ini;
	float fin;
	float energia_k;
	int posicion;
	float energia;
	struct
	{
		float valor;
		int posicion;
	} local[num_storms], global[num_storms];

	for (int i = 0; i < layer_size; i++)
		raiz[i] = 0.0;//sqrtf(i + 1);
	float aux;
	/* 4. Fase de bombardeos */
	for (i = 0; i < num_storms; i++)
	{

		/* 4.1. Suma energia de impactos */
		/* Para cada particula */
		for (j = 0; j < storms[i].size; j++)
		{
			/* Energia de impacto (en milesimas) */
			energia = (float)storms[i].posval[j * 2 + 1] / 1000;
			/* Posicion de impacto */
			posicion = storms[i].posval[j * 2];

			/* Para cada posicion de la capa */
			for (k = 0; k < local_layer_size; k++)
			{
				/* Actualizar posicion */

				/* 1. Calcular valor absoluto de la distancia entre el
		punto de impacto y el punto k de la capa */
				//int distancia = posicion - k;
				//if (distancia < 0)
				//	distancia = -distancia;

				/* 2. El punto de impacto tiene distancia 1 */
				//distancia = distancia + 1;

				/* 3. Raiz cuadrada de la distancia */
				//float atenuacion = sqrtf((float)distancia);

				/* 4. Calcular energia atenuada */
				//printf("Posicion: %d", posicion - (k + rank * (local_layer_size)));
				//fflush(stdout);

				if(raiz[abs(posicion - (k + desplazamiento))]==0){
					aux = raiz[abs(posicion - (k + desplazamiento))]=sqrtf(abs(posicion - (k + desplazamiento)));
				}else{
					aux = raiz[abs(posicion - (k + desplazamiento))];
				}
				energia_k = energia / aux;
				energia_k = energia / aux;

				// /* 5. No sumar si el valor absoluto es menor que umbral */
				if (energia_k >= UMBRAL)
					miniLayer[k] = miniLayer[k] + energia_k;
			}
		}
		// MPI_Barrier(MPI_COMM_WORLD);
		// for (j = 0; j < local_layer_size; j++)
		// {
		// 	printf("Rank: %d, Valor %f, Posicion: %d\n", rank, miniLayer[j], j+desplazamiento);
		// 	fflush(stdout);
		// }
		// printf("\n");

		/* 4.2. Relajacion entre tormentas de particulas */
		/* 4.2.1. Copiar valores a capa auxiliar */
		for (k = 0; k < local_layer_size; k++)
			layer_copy[k] = miniLayer[k];

		ini = 0;
		fin = 0;
		/* 4.2.2. Actualizar capa, menos los extremos, usando valores del array auxiliar */
		if (rank != 0)
			MPI_Send(&layer_copy[0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);

		if (rank != size - 1)
		{
			MPI_Recv(&fin, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
			MPI_Send(&layer_copy[local_layer_size - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
		}
		if (rank != 0)
		{
			MPI_Recv(&ini, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
		}
		// if (rank != 0)
		// MPI_Sendrecv(&layer_copy[0], 1,MPI_FLOAT,rank-1, 0, &ini, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status );
		// if (rank != size - 1)
		// MPI_Sendrecv(&layer_copy[local_layer_size-1], 1,MPI_FLOAT,rank+1, 0, &fin, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status );

		if (local_layer_size == 1)
		{
			if (rank != 0 && rank != size - 1)
				miniLayer[0] = (ini + layer_copy[0] + fin) / 3;
		}
		else
		{
			if (rank != size - 1)
				miniLayer[local_layer_size - 1] = (layer_copy[local_layer_size - 1 - 1] + layer_copy[local_layer_size - 1] + fin) / 3;
			if (rank != 0)
				miniLayer[0] = (ini + layer_copy[0] + layer_copy[1]) / 3;
		}

		for (k = 1; k < local_layer_size - 1; k++)
			miniLayer[k] = (layer_copy[k - 1] + layer_copy[k] + layer_copy[k + 1]) / 3;

		//MPI_Gather(miniLayer, local_layer_size, MPI_FLOAT, layer, local_layer_size, MPI_FLOAT, ROOT_RANK, MPI_COMM_WORLD);
		/* MPI_Barrier(MPI_COMM_WORLD);
		for (j = 0; j < local_layer_size; j++)
		{
			printf("Rank: %d, Valor %f, Posicion: %d\n", rank, miniLayer[j], j + desplazamiento);
			fflush(stdout);
		}
		printf("\n"); */

		local[i].valor = 0;
		local[i].posicion = 0;
		// for (int x = 0; x < local_layer_size; x++)
		// {
		// 	printf("[%d] %d, %lf\n", rank, x + desplazamiento, miniLayer[x]);
		// 	fflush(stdout);
		// }

		if (rank == 0)
		{
			inicio = 1;
			final = local_layer_size;
		}
		else if (rank == size - 1)
		{
			inicio = 0;
			final = local_layer_size - 1;
		}
		else
		{
			inicio = 0;
			final = local_layer_size;
		}
		for (k = inicio; k < final; k++)
		{
			//printf(" [%d] %lf %d\n",rank, miniLayer[k], k+desplazamiento);
			/* Comprobar solo maximos locales */
			if (miniLayer[k] > local[i].valor)
			{
				local[i].valor = miniLayer[k];
				local[i].posicion = k + desplazamiento;
			}
		}

		// MPI_Reduce(&local, &global, 1, MPI_FLOAT_INT, MPI_MAXLOC, ROOT_RANK, MPI_COMM_WORLD);

		// if (rank == 0)
		// {
		// 	maximos[i] = global.valor;
		// 	posiciones[i] = global.posicion;
		// }
	}

	MPI_Reduce(&local, &global, num_storms, MPI_FLOAT_INT, MPI_MAXLOC, ROOT_RANK, MPI_COMM_WORLD);
	for (i = 0; i < num_storms; i++)
	{
		maximos[i] = global[i].valor;
		posiciones[i] = global[i].posicion;
	}
	/* FINAL: No optimizar/paralelizar por debajo de este punto */
	/* 5. Final de medida de tiempo */
	// MPI_Barrier(MPI_COMM_WORLD);
	// if (rank == ROOT_RANK)
	// {
	// 	printf("%d\n", layer_size);
	// 	fflush(stdout);
	// }

	// printf("[%d] %d ", rank, desplazamiento);
	// fflush(stdout);
	free(raiz);
	free(miniLayer);
	free(layer_copy);
	MPI_Barrier(MPI_COMM_WORLD);
	ttotal = cp_Wtime() - ttotal;

/* 6. DEBUG: Dibujar resultado (Solo para capas con hasta 35 puntos) */
#ifdef DEBUG
	debug_print(layer_size, layer, posiciones, maximos, num_storms, MPI_Maximo, );
#endif

	if (rank == ROOT_RANK)
	{
		/* 7. Salida de resultados para tablon */
		printf("\n");
		/* 7.1. Tiempo total de la computacion */
		printf("Time: %lf\n", ttotal);
		/* 7.2. Escribir los maximos */
		printf("Result:");
		for (i = 0; i < num_storms; i++)
			printf(" %d %f", posiciones[i], maximos[i]);
		printf("\n");
	}
	/* 8. Liberar recursos */
	for (i = 0; i < argc - 2; i++)
		free(storms[i].posval);

	/* 9. Final correcto */
	MPI_Finalize();
	return 0;
}
