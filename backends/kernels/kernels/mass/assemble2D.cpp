// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../kernels.hpp"

// *****************************************************************************
extern "C" kernel
void rMassAssemble2D0(const int numElements,
                      const int NUM_QUAD_2D,
                      const double COEFF,
                      const double* __restrict quadWeights,
                      const double* __restrict J,
                      double* __restrict oper)
{
#ifdef __NVCC__
   const int e = blockDim.x * blockIdx.x + threadIdx.x;
   if (e < numElements)
#else
   forall(e,numElements,
#endif
   {
      for (int q = 0; q < NUM_QUAD_2D; ++q)
      {
         const double J11 = J[ijklNM(0,0,q,e,2,NUM_QUAD_2D)];
         const double J12 = J[ijklNM(1,0,q,e,2,NUM_QUAD_2D)];
         const double J21 = J[ijklNM(0,1,q,e,2,NUM_QUAD_2D)];
         const double J22 = J[ijklNM(1,1,q,e,2,NUM_QUAD_2D)];
         const double detJ = ((J11 * J22)-(J21 * J12));
         oper[ijN(q,e,NUM_QUAD_2D)] = quadWeights[q] * COEFF * detJ;
      }
   }
#ifndef __NVCC__
   );
#endif
}


// *****************************************************************************
void rMassAssemble2D(const int numElements,
                     const int NUM_QUAD_2D,
                     const double COEFF,
                     const double* quadWeights,
                     const double* J,
                     double* oper)
{
   nvtx_push(Lime);
   cuKer(rMassAssemble2D,numElements,NUM_QUAD_2D,COEFF,quadWeights,J,oper);
   nvtx_pop();
}
