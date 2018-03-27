#!/bin/bash
mpirun -n $1 ./energy 20 polla1; mpirun -n 1 ./secuencial 20 polla1
mpirun -n $1 ./energy 20 polla1  polla2; mpirun -n 1 ./secuencial 20 polla1 polla2
mpirun -n $1 ./energy 20 polla1  polla2 polla3; mpirun -n 1 ./secuencial 20 polla1 polla2 polla3