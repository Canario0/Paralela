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
#define BLOCK_SIZE 256

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
    //  long x = blockDim.y * blockIdx.y + threadIdx.y;
    //  long y = blockDim.x * blockIdx.x + threadIdx.x;
    //  long index = x * gridDim.x + y;
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
        //float atenuacion = (float)distancia*distancia;
        //float atenuacion = (float)distancia / PI;
        float atenuacion = sqrtf((float)distancia);

        /* 4. Calcular energia atenuada */
        float energia_k = energia / atenuacion;

        /* 5. No sumar si el valor absoluto es menor que umbral */
        if (energia_k >= UMBRAL)
            layer[index] = layer[index] + energia_k;
    }
}

__global__ void relaja(float *layer, float *layer_copy, int tam)
{
    // long x = blockDim.y * blockIdx.y + threadIdx.y;
    // long y = blockDim.x * blockIdx.x + threadIdx.x;
    // long k = x * gridDim.x + y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k == 0)
        layer[0] = layer_copy[0];
    if (k == tam - 1)
        layer[k] = layer_copy[k];
    if (0 < k && k < tam - 1)
        layer[k] = (layer_copy[k - 1] + layer_copy[k] + layer_copy[k + 1]) / 3;
}

__global__ void copia(float *layer, float *layer_copy, int tam)
{
    // long x = blockDim.y * blockIdx.y + threadIdx.y;
    // long y = blockDim.x * blockIdx.x + threadIdx.x;
    // long k = x * gridDim.x + y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < tam)
        layer_copy[k] = layer[k];
}
__global__ void cMax(float *layer, float *layer_copy, Max *maximosTemp, int layer_size, int i)
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
    // if(gid ==0){
    // for (int g = 0; g < layer_size;g++)
    // printf("%f |",layer[g]);

    // printf("\n");
    // }
    if (0 < gid && gid < layer_size - 1)
    {
        //if ( gid < layer_size && gid > 0){
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
    for (int s = blockDim.x/2 ; s > 0; s >>= 1)
    {
        if (lid < s)
        {
            // if (s ==2 && new_lid <=layer_size)
            // printf("lid: %d new_lid[%d]: %d new_lid+s[%d]: %d |", lid, new_lid, sdata[new_lid].pos, new_lid +s, sdata[new_lid+s].pos);
            // if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
            if (sdata[lid].val < sdata[(lid + s)].val)
            {
                sdata[lid] = sdata[(lid + s)]; // Copy Max
            }
            // }
        }
        __syncthreads();
    }

    if (lid == 0)
    {
        maximosTemp[blockIdx.x] = sdata[0];
        //printf("Maximo: %f y Posición : %d candidato: %f y Poisción : %d\n",maximosTemp[blockIdx.x].val, maximosTemp[blockIdx.x].pos, sdata[8].val, sdata[8].pos);
    }

    __syncthreads();

    // if(gid == 0){
    //     maximos[i] = maximosTemp[0].val;
    //     posiciones[i] = maximosTemp[0].pos;
    //     // printf("%d::%g %d, ", gridDim.x, maximosTemp[0].val, maximosTemp[0].pos);

    //     for(int j = 1; j < gridDim.x; j++) {
    //         // printf("%g %d, ", maximosTemp[j].val, maximosTemp[j].pos);
    //         if (maximos[i] < maximosTemp[j].val){
    //             maximos[i] = maximosTemp[j].val;
    //             posiciones[i] = maximosTemp[j].pos;
    //         }
    //     }
    //     // printf("\n");
    //     // printf("Maximo: %f y Posición : %d\n",maximos[i], posiciones[i]);
    // }
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

    float *dlayer;
    float *aux;
    Max *dMaximosTemp;
    cudaMalloc((void **)&dlayer, sizeof(float) * layer_size);
    dim3 gridShapeGpuFunc1(layer_size / BLOCK_SIZE + (layer_size % BLOCK_SIZE ? 1 : 0), 1);
    dim3 bloqShapeGpuFunc1(BLOCK_SIZE, 1);
    Max *maximosTemp = (Max *)malloc(sizeof(Max) * gridShapeGpuFunc1.x);
    cudaMalloc((void **)&dMaximosTemp, sizeof(Max) * gridShapeGpuFunc1.x);

    float *layer_copy;
    cudaMalloc((void **)&layer_copy, sizeof(float) * layer_size);
    if (layer_copy == NULL)
    {
        fprintf(stderr, "Error: Allocating the layer memory\n");
        exit(EXIT_FAILURE);
    }
    //cudaMemcpy(dlayer,layer, sizeof(float) * layer_size,cudaMemcpyHostToDevice);

    /* 4. Fase de bombardeos */
    for (i = 0; i < num_storms; i++)
    {
        if (i%2==0){
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
            actualiza<<<gridShapeGpuFunc1, bloqShapeGpuFunc1>>>(dlayer, posicion, energia, layer_size);
        }
        //cudaMemcpy(layer, dlayer, sizeof(float) * layer_size,cudaMemcpyDeviceToHost);

        /* 4.2. Relajacion entre tormentas de particulas */
        /* 4.2.1. Copiar valores a capa auxiliar */
        //  for( k=0; k<layer_size; k++ )
        //      layer_copy[k] = layer[k];

        /* 4.2.2. Actualizar capa, menos los extremos, usando valores del array auxiliar */
        //  for( k=1; k<layer_size-1; k++ )
        //    layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;
        // aux = layer_copy;
        // layer_copy = dlayer;
        // dlayer = aux;
        //copia<<<gridShapeGpuFunc1,bloqShapeGpuFunc1>>>(dlayer, layer_copy, layer_size);
        //relaja<<<gridShapeGpuFunc1, bloqShapeGpuFunc1>>>(dlayer, layer_copy, layer_size);
        
        cMax<<<gridShapeGpuFunc1, bloqShapeGpuFunc1, bloqShapeGpuFunc1.x * sizeof(Max)>>>(layer_copy, dlayer, dMaximosTemp, layer_size, i);
        }else{
            for (j = 0; j < storms[i].size; j++)
        {
            /* Energia de impacto (en milesimas) */
            float energia = (float)storms[i].posval[j * 2 + 1] / 1000;
            /* Posicion de impacto */
            int posicion = storms[i].posval[j * 2];

            /* Para cada posicion de la capa */
            /* Actualizar posicion */
            actualiza<<<gridShapeGpuFunc1, bloqShapeGpuFunc1>>>(layer_copy, posicion, energia, layer_size);
        }
            cMax<<<gridShapeGpuFunc1, bloqShapeGpuFunc1, bloqShapeGpuFunc1.x * sizeof(Max)>>>(dlayer, layer_copy, dMaximosTemp, layer_size, i);
        }
        
        cudaMemcpy(maximosTemp, dMaximosTemp, sizeof(Max) * gridShapeGpuFunc1.x, cudaMemcpyDeviceToHost);
        maximos[i] = maximosTemp[0].val;
        posiciones[i] = maximosTemp[0].pos;
        // printf("%d::%g %d, ", gridDim.x, maximosTemp[0].val, maximosTemp[0].pos);

        for (int j = 1; j < gridShapeGpuFunc1.x; j++)
        {
            // printf("%g %d, ", maximosTemp[j].val, maximosTemp[j].pos);
            if (maximos[i] < maximosTemp[j].val)
            {
                maximos[i] = maximosTemp[j].val;
                posiciones[i] = maximosTemp[j].pos;
            }
        }
        /* 4.3. Localizar maximo */
        //  for( k=1; k<layer_size-1; k++ ) {
        // 	/* Comprobar solo maximos locales */
        // 	if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
        // 		if ( layer[k] > maximos[i] ) {
        // 			maximos[i] = layer[k];
        // 			posiciones[i] = k;
        // 		}
        // 	}
        // }
    }
    //  cudaDeviceSynchronize();
    // for (i=0; i<num_storms; i++)
    //          printf(" %d %f", posiciones[i], maximos[i] );
    //  cudaMemcpy(maximos, dMaximos, sizeof(float)*num_storms, cudaMemcpyDeviceToHost);
    //     cudaMemcpy(posiciones, dPosiciones, sizeof(int)*num_storms, cudaMemcpyDeviceToHost);
    /* FINAL: No optimizar/paralelizar por debajo de este punto */

    /* 6. Final de medida de tiempo */
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
