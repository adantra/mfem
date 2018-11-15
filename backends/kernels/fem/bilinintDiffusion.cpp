// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

// *****************************************************************************
KernelsDiffusionIntegrator::KernelsDiffusionIntegrator(const KernelsCoefficient
                                                       &coeff_)
   :
   KernelsIntegrator(coeff_.KernelsEngine()),
   coeff(coeff_),
   assembledOperator(*(new Layout(coeff_.KernelsEngine(), 0)))
{
   nvtx_push();
   coeff.SetName("COEFF");
   nvtx_pop();
}

// *****************************************************************************
KernelsDiffusionIntegrator::~KernelsDiffusionIntegrator() {}

// *****************************************************************************
std::string KernelsDiffusionIntegrator::GetName()
{
   return "DiffusionIntegrator";
}

// *****************************************************************************
void KernelsDiffusionIntegrator::SetupIntegrationRule()
{
   nvtx_push();
   const FiniteElement &trialFE = *(trialFESpace->GetFE(0));
   const FiniteElement &testFE  = *(testFESpace->GetFE(0));
   ir = &mfem::DiffusionIntegrator::GetRule(trialFE, testFE);
   nvtx_pop();
}

// *****************************************************************************
void KernelsDiffusionIntegrator::Setup()
{
   nvtx_push();
   nvtx_pop();
}

// *****************************************************************************
void KernelsDiffusionIntegrator::Assemble()
{
   nvtx_push();
   const mfem::FiniteElement &fe = *(trialFESpace->GetFE(0));
   const int dim = mesh->Dimension();
   const int dims = fe.GetDim();
   assert(dim==dims);

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int elements = trialFESpace->GetNE();
   assert(elements==mesh->GetNE());

   const int quadraturePoints = ir->GetNPoints();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();

   kGeometry *geo = GetGeometry(kGeometry::Jacobian);
   assert(geo);

   assembledOperator.Resize<double>(symmDims * quadraturePoints * elements,NULL);
   rDiffusionAssemble(dim,
                      quad1D,
                      mesh->GetNE(),
                      maps->quadWeights,
                      geo->J,
                      1.0,//COEFF
                      (double*)assembledOperator.KernelsMem().ptr());
   nvtx_pop();
}

// *****************************************************************************
void KernelsDiffusionIntegrator::MultAdd(Vector &x, Vector &y)
{
   nvtx_push();
   const int dim = mesh->Dimension();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   const int dofs1D = trialFESpace->GetFE(0)->GetOrder() + 1;
   rDiffusionMultAdd(dim,
                     dofs1D,
                     quad1D,
                     mesh->GetNE(),
                     maps->dofToQuad,
                     maps->dofToQuadD,
                     maps->quadToDof,
                     maps->quadToDofD,
                     (double*)assembledOperator.KernelsMem().ptr(),
                     (const double*)x.KernelsMem().ptr(),
                     (double*)y.KernelsMem().ptr());
   nvtx_pop();
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
