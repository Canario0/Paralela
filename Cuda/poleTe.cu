/*
 * Simulacion simplificada de bombardeo de particulas de alta energia
 *
 * Computacion Paralela (Grado en Informatica)
 * 2017/2018
 *
 * (c) 2018 Arturo Gonzalez Escribano
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cputils.h>

#define PI 3.14159f
#define UMBRAL 0.001f
#define BLOCK_SIZE 128

/* Estructura para almacenar los datos de una tormenta de particulas */
typedef struct
{
    int size;
    int *posval;
} Storm;

typedef struct
{
    float val;
    int pos;
} Max;

/* ESTA FUNCION PUEDE SER MODIFICADA */
/* Funcion para actualizar una posicion de la capa */
__global__ void actualiza(float *layer, int pos, float energia, int tam)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    /* 1. Calcular valor absoluto de la distancia entre el
         punto de impacto y el punto k de la capa */
    if (index < tam)
    {
        int distancia = pos - index;
        if (distancia < 0)
            distancia = -distancia;

        /* 2. El punto de impacto tiene distancia 1 */
        distancia = distancia + 1;

        /* 3. Raiz cuadrada de la distancia */
        float atenuacion = sqrtf((float)distancia);

        /* 4. Calcular energia atenuada */
        float energia_k = energia / atenuacion;

        /* 5. No sumar si el valor absoluto es menor que umbral */
        if (energia_k >= UMBRAL)
            layer[index] = layer[index] + energia_k;
    }
}

__global__ void cMax(float *layer, float *layer_copy, Max *maximosTemp, int layer_size)
{
    extern __shared__ Max sdata[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x; // Global idº
    int lid = threadIdx.x;                           // Local id
    if (gid == 0)
        layer[0] = layer_copy[0];
    if (gid == layer_size - 1)
        layer[gid] = layer_copy[gid];
    if (0 < gid && gid < layer_size - 1)
        layer[gid] = (layer_copy[gid - 1] + layer_copy[gid] + layer_copy[gid + 1]) / 3;
    __syncthreads();

    if (0 < gid && gid < layer_size - 1)
    {
        if (layer[gid] > layer[gid - 1] && layer[gid] > layer[gid + 1])
        {
            sdata[lid].val = layer[gid];
            sdata[lid].pos = gid;
        }
        else
        {
            sdata[lid].val = 0.0f;
            sdata[lid].pos = 0;
        }
    }
    else
    {
        sdata[lid].val = 0.0f;
        sdata[lid].pos = 0;
    }
    __syncthreads();

    // do reduction in shared mem
    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (lid < s)
        {
            if (sdata[lid].val < sdata[(lid + s)].val)
            {
                sdata[lid] = sdata[(lid + s)]; // Copy Max
            }
            // }
        }
        __syncthreads();
    }
    if (lid < 32)
    {
        if (sdata[lid].val < sdata[(lid + 32)].val)
        {
            sdata[lid] = sdata[(lid + 32)]; // Copy Max
        }

        if (sdata[lid].val < sdata[(lid + 16)].val)
        {
            sdata[lid] = sdata[(lid + 16)]; // Copy Max
        }

        if (sdata[lid].val < sdata[(lid + 8)].val)
        {
            sdata[lid] = sdata[(lid + 8)]; // Copy Max
        }

        if (sdata[lid].val < sdata[(lid + 4)].val)
        {
            sdata[lid] = sdata[(lid + 4)]; // Copy Max
        }

        if (sdata[lid].val < sdata[(lid + 2)].val)
        {
            sdata[lid] = sdata[(lid + 2)]; // Copy Max
        }

        if (sdata[lid].val < sdata[(lid + 1)].val)
        {
            sdata[lid] = sdata[(lid + 1)]; // Copy Max
        }
    }
    if (lid == 0)
    {
        maximosTemp[blockIdx.x] = sdata[0];
    }

    __syncthreads();
}

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

    /* 1.1. Leer argumentos */
    if (argc < 3)
    {
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
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    double ttotal = cp_Wtime();

    /* COMIENZO: No optimizar/paralelizar el main por encima de este punto */

    /* 3. Reservar memoria para las capas e inicializar a cero */

    unsigned int grid_size = layer_size / BLOCK_SIZE + (layer_size % BLOCK_SIZE ? 1 : 0);
    //dim3 gridShapeGpuFunc1(layer_size / BLOCK_SIZE + (layer_size % BLOCK_SIZE ? 1 : 0), 1);
    unsigned int block_size = BLOCK_SIZE;
    //dim3 bloqShapeGpuFunc1(BLOCK_SIZE, 1);

    float *dlayer;
    float *layer_copy;
    Max *dMaximosTemp;
    Max *maximosTemp;
    cudaMalloc((void **)&dlayer, sizeof(float) * layer_size);
    cudaMallocHost(&maximosTemp, sizeof(Max) * grid_size);
    cudaMalloc((void **)&dMaximosTemp, sizeof(Max) * grid_size);
    cudaMalloc((void **)&layer_copy, sizeof(float) * layer_size);
    if (layer_copy == NULL)
    {
        fprintf(stderr, "Error: Allocating the layer memory\n");
        exit(EXIT_FAILURE);
    }

    /* 4. Fase de bombardeos */
    for (i = 0; i < num_storms; i++)
    {
        if (i % 2 == 0)
        {
            /* 4.1. Suma energia de impactos */
            /* Para cada particula */
            for (j = 0; j < storms[i].size; j++)
            {
                /* Energia de impacto (en milesimas) */
                float energia = (float)storms[i].posval[j * 2 + 1] / 1000;
                /* Posicion de impacto */
                int posicion = storms[i].posval[j * 2];

                /* Para cada posicion de la capa */
                /* Actualizar posicion */
                actualiza<<<grid_size, block_size>>>(dlayer, posicion, energia, layer_size);
            }
            cMax<<<grid_size, block_size, block_size * sizeof(Max)>>>(layer_copy, dlayer, dMaximosTemp, layer_size);
        }
        else
        {
            for (j = 0; j < storms[i].size; j++)
            {
                /* Energia de impacto (en milesimas) */
                float energia = (float)storms[i].posval[j * 2 + 1] / 1000;
                /* Posicion de impacto */
                int posicion = storms[i].posval[j * 2];

                /* Para cada posicion de la capa */
                /* Actualizar posicion */
                actualiza<<<grid_size, block_size>>>(layer_copy, posicion, energia, layer_size);
            }
            cMax<<<grid_size, block_size, block_size * sizeof(Max)>>>(dlayer, layer_copy, dMaximosTemp, layer_size);
        }

        cudaMemcpy(maximosTemp, dMaximosTemp, sizeof(Max) * grid_size, cudaMemcpyDeviceToHost);
        maximos[i] = maximosTemp[0].val;
        posiciones[i] = maximosTemp[0].pos;

        for (int x = 1; x < grid_size - 1; x++)
        {
            if (maximosTemp[x].val > maximosTemp[x - 1].val && maximosTemp[x].val > maximosTemp[x + 1].val)
            {
                if (maximos[i] < maximosTemp[x].val)
                {
                    maximos[i] = maximosTemp[x].val;
                    posiciones[i] = maximosTemp[x].pos;
                }
            }
        }

        if (maximos[i] < maximosTemp[grid_size - 1].val)
        {
            maximos[i] = maximosTemp[grid_size - 1].val;
            posiciones[i] = maximosTemp[grid_size - 1].pos;
        }
    }
    cudaDeviceSynchronize();
    ttotal = cp_Wtime() - ttotal;

/* 7. DEBUG: Dibujar resultado (Solo para capas con hasta 35 puntos) */
#ifdef DEBUG
    debug_print(layer_size, layer, posiciones, maximos, num_storms);
#endif

    /* 8. Salida de resultados para tablon */
    printf("\n");
    /* 8.1. Tiempo total de la computacion */
    printf("Time: %lf\n", ttotal);
    /* 8.2. Escribir los maximos */
    printf("Result:");
    for (i = 0; i < num_storms; i++)
        printf(" %d %f", posiciones[i], maximos[i]);
    printf("\n");

    /* 9. Liberar recursos */
    for (i = 0; i < argc - 2; i++)
        free(storms[i].posval);

    /* 10. Final correcto */
    return 0;
}
