// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "restriction.hpp"
#include "prestriction.hpp"
#include "pgridfunc.hpp"
#include "pfespace.hpp"
#include "fespace.hpp"
#include "../general/forall.hpp"

#define MFEM_DEBUG_COLOR 197
#include "../general/debug.hpp"

namespace mfem
{

ParNCH1FaceRestriction::ParNCH1FaceRestriction(const ParFiniteElementSpace &fes,
                                               ElementDofOrdering ordering,
                                               FaceType type)
   : H1FaceRestriction(fes,type),
     type(type),
     interpolations(fes,ordering,type)
{
   if (nf==0) { return; }
   // If fespace == H1
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "H1FaceRestriction.");

   // Assuming all finite elements are using Gauss-Lobatto.
   height = vdim*nf*face_dofs;
   width = fes.GetVSize();
   const bool dof_reorder = (ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe = fes.GetFaceElement(f);
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
      const FiniteElement *fe = fes.GetFaceElement(0);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &fe_dof_map = el->GetDofMap();
      MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
   }
   // End of verifications
   x_interp.UseDevice(true);

   ComputeScatterIndicesAndOffsets(ordering, type);

   ComputeGatherIndices(ordering, type);
}

void ParNCH1FaceRestriction::Mult(const Vector &x, Vector &y) const
{
   //assert(false);
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;

   if ( type==FaceType::Boundary )
   {
      auto d_indices = scatter_indices.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nface_dofs, vd, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nface_dofs;
         const int face = i / nface_dofs;
         const int idx = d_indices[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
         }
      });
   }
   else // type==FaceType::Interior
   {
      auto d_indices = scatter_indices.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nface_dofs, vd, nf);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
      static constexpr int max_nd = 1024;
      MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nface_dofs, 1, 1,
      {
         MFEM_SHARED double dof_values[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int master_side = conf.GetNonConformingMasterSide();
         const int interp_index = conf.GetInterpolatorIndex();
         const int side = 0;
         if ( interp_index==InterpConfig::conforming || side!=master_side )
         {
            MFEM_FOREACH_THREAD(dof,x,nface_dofs)
            {
               const int i = face*nface_dofs + dof;
               const int idx = d_indices[i];
               for (int c = 0; c < vd; ++c)
               {
                  d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
               }
            }
         }
         else // Interpolation from coarse to fine
         {
            for (int c = 0; c < vd; ++c)
            {
               // Load the face dofs in shared memory
               MFEM_FOREACH_THREAD(dof,x,nface_dofs)
               {
                  const int i = face*nface_dofs + dof;
                  const int idx = d_indices[i];
                  dof_values[dof] = d_x(t?c:idx, t?idx:c);
               }
               MFEM_SYNC_THREAD;
               // Apply the interpolation to the face dofs
               MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
               {
                  double res = 0.0;
                  for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
                  {
                     res += d_interp(dof_out, dof_in, interp_index)*dof_values[dof_in];
                  }
                  d_y(dof_out, c, face) = res;
               }
               MFEM_SYNC_THREAD;
            }
         }
      });
   }
}

void ParNCH1FaceRestriction::AddMultTranspose(const Vector &x, Vector &y) const
{
   assert(false);
   if (x_interp.Size()==0)
   {
      x_interp.SetSize(x.Size());
   }
   x_interp = x;
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   if ( type==FaceType::Interior )
   {
      // Interpolation from slave to master face dofs
      auto d_x = Reshape(x_interp.ReadWrite(), nface_dofs, vd, nf);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
      static constexpr int max_nd = 1024;
      MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nface_dofs, 1, 1,
      {
         MFEM_SHARED double dof_values[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int master_side = conf.GetNonConformingMasterSide();
         const int interp_index = conf.GetInterpolatorIndex();
         if ( interp_index!=InterpConfig::conforming && master_side==0 )
         {
            // Interpolation from fine to coarse
            for (int c = 0; c < vd; ++c)
            {
               MFEM_FOREACH_THREAD(dof,x,nface_dofs)
               {
                  dof_values[dof] = d_x(dof, c, face);
               }
               MFEM_SYNC_THREAD;
               MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
               {
                  double res = 0.0;
                  for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
                  {
                     res += d_interp(dof_in, dof_out, interp_index)*dof_values[dof_in];
                  }
                  d_x(dof_out, c, face) = res;
               }
               MFEM_SYNC_THREAD;
            }
         }
      });
   }

   // Gathering of face dofs into element dofs
   auto d_offsets = gather_offsets.Read();
   auto d_indices = gather_indices.Read();
   auto d_x = Reshape(x_interp.Read(), nface_dofs, vd, nf);
   auto d_y = Reshape(y.ReadWrite(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dof_value = 0;
         for (int j = offset; j < next_offset; ++j)
         {
            int idx_j = d_indices[j];
            dof_value +=  d_x(idx_j % nface_dofs, c, idx_j / nface_dofs);
         }
         d_y(t?c:i,t?i:c) += dof_value;
      }
   });
}

void ParNCH1FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      gather_offsets[i] = 0;
   }

   // Computation of scatter indices and offsets
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsLocal() && face.IsNonConformingMaster() )
      {
         // We skip local non-conforming master faces as they are treated by the
         // local non-conforming slave faces.
         continue;
      }
      else if (type==FaceType::Interior && face.IsInterior())
      {
         if ( face.IsConforming() )
         {
            interpolations.RegisterFaceConformingInterpolation(face,f_ind);
            SetFaceDofsScatterIndices(face, f_ind, ordering);
            f_ind++;
         }
         else // Non-conforming face
         {
            SetFaceDofsScatterIndices(face, f_ind, ordering);
            if (face.IsSharedNonConformingSlave())
            {
               // In this case the local face is the master (coarse) face, thus
               // we need to interpolate the values on the slave (fine) face.
               interpolations.RegisterFaceCoarseToFineInterpolation(face,f_ind);
            }
            else
            {
               // Treated as a conforming face since we only extract values from
               // the local slave (fine) face.
               interpolations.RegisterFaceConformingInterpolation(face,f_ind);
            }
            f_ind++;
         }
      }
      else if (type==FaceType::Boundary && face.IsBoundary())
      {
         SetFaceDofsScatterIndices(face, f_ind, ordering);
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      gather_offsets[i] += gather_offsets[i - 1];
   }

   // Transform the interpolation matrix map into a contiguous memory structure.
   interpolations.LinearizeInterpolatorMapIntoVector();
}

void ParNCH1FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsLocal() && face.IsNonConformingMaster() )
      {
         // We skip local non-conforming master faces as they are treated by the
         // local non-conforming slave faces.
         continue;
      }
      else if (face.IsOfFaceType(type))
      {
         SetFaceDofsGatherIndices(face, f_ind, ordering);
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Reset offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      gather_offsets[i] = gather_offsets[i - 1];
   }
   gather_offsets[0] = 0;
}

ParL2FaceRestriction::ParL2FaceRestriction(const ParFiniteElementSpace &fes,
                                           ElementDofOrdering ordering,
                                           FaceType type,
                                           L2FaceValues m)
   : L2FaceRestriction(fes, type, m)
{
   //assert(false);
   if (nf==0) { return; }
   // If fespace == L2
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   const FiniteElement *fe = pfes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "ParL2FaceRestriction.");
   MFEM_VERIFY(pfes.GetMesh()->Conforming(),
               "Non-conforming meshes not yet supported with partial assembly.");
   // Assuming all finite elements are using Gauss-Lobatto dofs
   height = (m==L2FaceValues::DoubleValued? 2 : 1)*vdim*nf*face_dofs;
   width = pfes.GetVSize();
   const bool dof_reorder = (ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder)
   {
      MFEM_ABORT("Non-Tensor L2FaceRestriction not yet implemented.");
   }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < pfes.GetNF(); ++f)
      {
         const FiniteElement *fe =
            pfes.GetTraceElement(f, pfes.GetMesh()->GetFaceBaseGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
   }
   // End of verifications

   ComputeScatterIndicesAndOffsets(ordering, type);

   ComputeGatherIndices(ordering, type);
}

void ParL2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   //assert(false);
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   ParGridFunction x_gf;
   x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(&pfes),
                const_cast<Vector&>(x), 0);

   x_gf.ExchangeFaceNbrData();

   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   const int threshold = ndofs;
   const int nsdofs = pfes.GetFaceNbrVSize();

   if (m==L2FaceValues::DoubleValued)
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_x_shared = Reshape(x_gf.FaceNbrData().Read(),
                                t?vd:nsdofs, t?nsdofs:vd);
      auto d_y = Reshape(y.Write(), nface_dofs, vd, 2, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nface_dofs;
         const int face = i / nface_dofs;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 0, face) = d_x(t?c:idx1, t?idx1:c);
         }
         const int idx2 = d_indices2[i];
         for (int c = 0; c < vd; ++c)
         {
            if (idx2>-1 && idx2<threshold) // interior face
            {
               d_y(dof, c, 1, face) = d_x(t?c:idx2, t?idx2:c);
            }
            else if (idx2>=threshold) // shared boundary
            {
               d_y(dof, c, 1, face) = d_x_shared(t?c:(idx2-threshold),
                                                 t?(idx2-threshold):c);
            }
            else // true boundary
            {
               d_y(dof, c, 1, face) = 0.0;
            }
         }
      });
   }
   else
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nface_dofs, vd, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nface_dofs;
         const int face = i / nface_dofs;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, face) = d_x(t?c:idx1, t?idx1:c);
         }
      });
   }
}

static MFEM_HOST_DEVICE int AddNnz(const int iE, int *I, const int dofs)
{
   int val = AtomicAdd(I[iE],dofs);
   return val;
}

void ParL2FaceRestriction::FillI(SparseMatrix &mat,
                                 const bool keep_nbr_block) const
{
   if (keep_nbr_block)
   {
      return L2FaceRestriction::FillI(mat, keep_nbr_block);
   }
   const int nface_dofs = face_dofs;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   MFEM_FORALL(fdof, nf*nface_dofs,
   {
      const int f  = fdof/nface_dofs;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      if (iE1 < Ndofs)
      {
         AddNnz(iE1,I,nface_dofs);
      }
      const int iE2 = d_indices2[f*nface_dofs+iF];
      if (iE2 < Ndofs)
      {
         AddNnz(iE2,I,nface_dofs);
      }
   });
}

void ParL2FaceRestriction::FillI(SparseMatrix &mat,
                                 SparseMatrix &face_mat) const
{
   const int nface_dofs = face_dofs;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   auto I_face = face_mat.ReadWriteI();
   MFEM_FORALL(i, ne*elem_dofs*vdim+1,
   {
      I_face[i] = 0;
   });
   MFEM_FORALL(fdof, nf*nface_dofs,
   {
      const int f  = fdof/nface_dofs;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      if (iE1 < Ndofs)
      {
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE2 = d_indices2[f*nface_dofs+jF];
            if (jE2 < Ndofs)
            {
               AddNnz(iE1,I,1);
            }
            else
            {
               AddNnz(iE1,I_face,1);
            }
         }
      }
      const int iE2 = d_indices2[f*nface_dofs+iF];
      if (iE2 < Ndofs)
      {
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE1 = d_indices1[f*nface_dofs+jF];
            if (jE1 < Ndofs)
            {
               AddNnz(iE2,I,1);
            }
            else
            {
               AddNnz(iE2,I_face,1);
            }
         }
      }
   });
}

void ParL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                        SparseMatrix &mat,
                                        const bool keep_nbr_block) const
{
   if (keep_nbr_block)
   {
      return L2FaceRestriction::FillJAndData(ea_data, mat, keep_nbr_block);
   }
   const int nface_dofs = face_dofs;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto mat_fea = Reshape(ea_data.Read(), nface_dofs, nface_dofs, 2, nf);
   auto I = mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   MFEM_FORALL(fdof, nf*nface_dofs,
   {
      const int f  = fdof/nface_dofs;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      if (iE1 < Ndofs)
      {
         const int offset = AddNnz(iE1,I,nface_dofs);
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE2 = d_indices2[f*nface_dofs+jF];
            J[offset+jF] = jE2;
            Data[offset+jF] = mat_fea(jF,iF,1,f);
         }
      }
      const int iE2 = d_indices2[f*nface_dofs+iF];
      if (iE2 < Ndofs)
      {
         const int offset = AddNnz(iE2,I,nface_dofs);
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE1 = d_indices1[f*nface_dofs+jF];
            J[offset+jF] = jE1;
            Data[offset+jF] = mat_fea(jF,iF,0,f);
         }
      }
   });
}

void ParL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                        SparseMatrix &mat,
                                        SparseMatrix &face_mat) const
{
   const int nface_dofs = face_dofs;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto mat_fea = Reshape(ea_data.Read(), nface_dofs, nface_dofs, 2, nf);
   auto I = mat.ReadWriteI();
   auto I_face = face_mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto J_face = face_mat.WriteJ();
   auto Data = mat.WriteData();
   auto Data_face = face_mat.WriteData();
   MFEM_FORALL(fdof, nf*nface_dofs,
   {
      const int f  = fdof/nface_dofs;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      if (iE1 < Ndofs)
      {
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE2 = d_indices2[f*nface_dofs+jF];
            if (jE2 < Ndofs)
            {
               const int offset = AddNnz(iE1,I,1);
               J[offset] = jE2;
               Data[offset] = mat_fea(jF,iF,1,f);
            }
            else
            {
               const int offset = AddNnz(iE1,I_face,1);
               J_face[offset] = jE2-Ndofs;
               Data_face[offset] = mat_fea(jF,iF,1,f);
            }
         }
      }
      const int iE2 = d_indices2[f*nface_dofs+iF];
      if (iE2 < Ndofs)
      {
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE1 = d_indices1[f*nface_dofs+jF];
            if (jE1 < Ndofs)
            {
               const int offset = AddNnz(iE2,I,1);
               J[offset] = jE1;
               Data[offset] = mat_fea(jF,iF,0,f);
            }
            else
            {
               const int offset = AddNnz(iE2,I_face,1);
               J_face[offset] = jE1-Ndofs;
               Data_face[offset] = mat_fea(jF,iF,0,f);
            }
         }
      }
   });
}

void ParL2FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      gather_offsets[i] = 0;
   }

   // Computation of scatter indices and offsets
   int f_ind=0;
   for (int f = 0; f < pfes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if (type==FaceType::Interior && face.IsInterior())
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if (m==L2FaceValues::DoubleValued)
         {
            if (face.IsShared())
            {
               PermuteAndSetSharedFaceDofsScatterIndices2(face,f_ind);
            }
            else
            {
               PermuteAndSetFaceDofsScatterIndices2(face,f_ind);
            }
         }
         f_ind++;
      }
      else if (type==FaceType::Boundary && face.IsBoundary())
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if (m==L2FaceValues::DoubleValued)
         {
            SetBoundaryDofsScatterIndices2(face,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      gather_offsets[i] += gather_offsets[i - 1];
   }
}


void ParL2FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if (face.IsOfFaceType(type))
      {
         SetFaceDofsGatherIndices1(face,f_ind);
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             face.IsLocal())
         {
            PermuteAndSetFaceDofsGatherIndices2(face,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Reset offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      gather_offsets[i] = gather_offsets[i - 1];
   }
   gather_offsets[0] = 0;
}

ParNCL2FaceRestriction::ParNCL2FaceRestriction(const ParFiniteElementSpace &fes,
                                               ElementDofOrdering ordering,
                                               FaceType type,
                                               L2FaceValues m)
   : L2FaceRestriction(fes, type, m), interpolations(fes, ordering, type)
{
   //assert(false);
   if (nf==0) { assert(false); return; }
   // If fespace==L2
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   const FiniteElement *fe = pfes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "ParNCL2FaceRestriction.");
   // Assuming all finite elements are using Gauss-Lobatto dofs
   height = (m==L2FaceValues::DoubleValued? 2 : 1)*vdim*nf*face_dofs;
   width = pfes.GetVSize();
   const bool dof_reorder = (ordering==ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder)
   {
      MFEM_ABORT("Non-Tensor L2FaceRestriction not yet implemented.");
   }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < pfes.GetNF(); ++f)
      {
         const FiniteElement *fe =
            pfes.GetTraceElement(f, pfes.GetMesh()->GetFaceBaseGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
   }
   // End of verifications
   x_interp.UseDevice(true);

   ComputeScatterIndicesAndOffsets(ordering, type);

   ComputeGatherIndices(ordering, type);
}

void ParNCL2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   //assert(false);
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   ParGridFunction x_gf;
   x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(&pfes),
                const_cast<Vector&>(x), 0);
   x_gf.SetTrueVector();
   x_gf.SetFromTrueVector();
   x_gf.ExchangeFaceNbrData();

   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   const int threshold = ndofs;
   const int nsdofs = pfes.GetFaceNbrVSize();

   if ( type==FaceType::Interior && m==L2FaceValues::DoubleValued )
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_x_shared = Reshape(x_gf.FaceNbrData().Read(),
                                t?vd:nsdofs, t?nsdofs:vd);
      auto d_y = Reshape(y.Write(), nface_dofs, vd, 2, nf);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
      static constexpr int max_nd = 1024;
      MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nface_dofs, 1, 1,
      {
         MFEM_SHARED double dof_values[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int master_side = conf.GetNonConformingMasterSide();
         const int interp_index = conf.GetInterpolatorIndex();
         for (int side = 0; side < 2; side++)
         {
            if ( interp_index==InterpConfig::conforming || side!=master_side )
            {
               // No interpolation
               MFEM_FOREACH_THREAD(dof,x,nface_dofs)
               {
                  const int i = face*nface_dofs + dof;
                  const int idx = side==0 ? d_indices1[i] : d_indices2[i];
                  if (idx>-1 && idx<threshold) // local interior face
                  {
                     for (int c = 0; c < vd; ++c)
                     {
                        d_y(dof, c, side, face) = d_x(t?c:idx, t?idx:c);
                     }
                  }
                  else if (idx>=threshold) // shared interior face
                  {
                     const int sidx = idx-threshold;
                     for (int c = 0; c < vd; ++c)
                     {
                        d_y(dof, c, side, face) = d_x_shared(t?c:sidx, t?sidx:c);
                     }
                  }
                  else // true boundary
                  {
                     for (int c = 0; c < vd; ++c)
                     {
                        d_y(dof, c, side, face) = 0.0;
                     }
                  }
               }
            }
            else // Interpolation from coarse to fine
            {
               for (int c = 0; c < vd; ++c)
               {
                  MFEM_FOREACH_THREAD(dof,x,nface_dofs)
                  {
                     const int i = face*nface_dofs + dof;
                     const int idx = side==0 ? d_indices1[i] : d_indices2[i];
                     if (idx>-1 && idx<threshold) // local interior face
                     {
                        dof_values[dof] = d_x(t?c:idx, t?idx:c);
                     }
                     else if (idx>=threshold) // shared interior face
                     {
                        const int sidx = idx-threshold;
                        dof_values[dof] = d_x_shared(t?c:sidx, t?sidx:c);
                     }
                     else // true boundary
                     {
                        dof_values[dof] = 0.0;
                     }
                  }
                  MFEM_SYNC_THREAD;
                  MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
                  {
                     double res = 0.0;
                     for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
                     {
                        res += d_interp(dof_out, dof_in, interp_index)*dof_values[dof_in];
                     }
                     d_y(dof_out, c, side, face) = res;
                  }
                  MFEM_SYNC_THREAD;
               }
            }
         }
      });
   }
   else if ( type==FaceType::Boundary && m==L2FaceValues::DoubleValued )
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nface_dofs, vd, 2, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nface_dofs;
         const int face = i / nface_dofs;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 0, face) = d_x(t?c:idx1, t?idx1:c);
         }
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 1, face) = 0.0;
         }
      });
   }
   else if ( type==FaceType::Interior && m==L2FaceValues::SingleValued )
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nface_dofs, vd, nf);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
      static constexpr int max_nd = 16*16;
      MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nface_dofs, 1, 1,
      {
         MFEM_SHARED double dof_values[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int master_side = conf.GetNonConformingMasterSide();
         const int interp_index = conf.GetInterpolatorIndex();
         const int side = 0;
         if ( interp_index==InterpConfig::conforming || side!=master_side )
         {
            MFEM_FOREACH_THREAD(dof,x,nface_dofs)
            {
               const int i = face*nface_dofs + dof;
               const int idx = d_indices1[i];
               if (idx>-1 && idx<threshold) // interior face
               {
                  for (int c = 0; c < vd; ++c)
                  {
                     d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
                  }
               }
               // else if (idx>=threshold) // shared interior face
               // {
               //    const int sidx = idx-threshold;
               //    for (int c = 0; c < vd; ++c)
               //    {
               //       d_y(dof, c, face) = d_x_shared(t?c:sidx, t?sidx:c);
               //    }
               // }
               else // true boundary
               {
                  for (int c = 0; c < vd; ++c)
                  {
                     d_y(dof, c, face) = 0.0;
                  }
               }
            }
         }
         else // Interpolation from coarse to fine
         {
            for (int c = 0; c < vd; ++c)
            {
               MFEM_FOREACH_THREAD(dof,x,nface_dofs)
               {
                  const int i = face*nface_dofs + dof;
                  const int idx = d_indices1[i];
                  if (idx>-1 && idx<threshold) // interior face
                  {
                     dof_values[dof] = d_x(t?c:idx, t?idx:c);
                  }
                  // else if (idx>=threshold) // shared interior face
                  // {
                  //    const int sidx = idx-threshold;
                  //    dof_values[dof] = d_x_shared(t?c:sidx, t?sidx:c);
                  // }
                  else // true boundary
                  {
                     dof_values[dof] = 0.0;
                  }
               }
               MFEM_SYNC_THREAD;
               MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
               {
                  double res = 0.0;
                  for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
                  {
                     res += d_interp(dof_out, dof_in, interp_index)*dof_values[dof_in];
                  }
                  d_y(dof_out, c, face) = res;
               }
               MFEM_SYNC_THREAD;
            }
         }
      });
   }
   else if ( type==FaceType::Boundary && m==L2FaceValues::SingleValued )
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nface_dofs, vd, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nface_dofs;
         const int face = i / nface_dofs;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, face) = d_x(t?c:idx1, t?idx1:c);
         }
      });
   }
   else
   {
      MFEM_ABORT("Unknown type and multiplicity combination.");
   }
}

void ParNCL2FaceRestriction::AddMultTranspose(const Vector &x, Vector &y) const
{
   if (x_interp.Size()==0)
   {
      x_interp.SetSize(x.Size());
   }
   x_interp = x;
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   // Interpolation from slave to master face dofs
   if ( type==FaceType::Interior && m==L2FaceValues::DoubleValued )
   {
      auto d_x = Reshape(x_interp.ReadWrite(), nface_dofs, vd, 2, nf);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
      static constexpr int max_nd = 1024;
      MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nface_dofs, 1, 1,
      {
         MFEM_SHARED double dof_values[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int master_side = conf.GetNonConformingMasterSide();
         const int interp_index = conf.GetInterpolatorIndex();
         if ( interp_index!=InterpConfig::conforming )
         {
            // Interpolation from fine to coarse
            for (int c = 0; c < vd; ++c)
            {
               MFEM_FOREACH_THREAD(dof,x,nface_dofs)
               {
                  dof_values[dof] = d_x(dof, c, master_side, face);
               }
               MFEM_SYNC_THREAD;
               MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
               {
                  double res = 0.0;
                  for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
                  {
                     res += d_interp(dof_in, dof_out, interp_index)*dof_values[dof_in];
                  }
                  d_x(dof_out, c, master_side, face) = res;
               }
               MFEM_SYNC_THREAD;
            }
         }
      });
   }
   else if ( type==FaceType::Interior && m==L2FaceValues::SingleValued )
   {
      assert(false);
      /*auto d_x = Reshape(x_interp.ReadWrite(), nface_dofs, vd, nf);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
      static constexpr int max_nd = 1024;
      MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nface_dofs, 1, 1,
      {
         MFEM_SHARED double dof_values[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int master_side = conf.GetNonConformingMasterSide();
         const int interp_index = conf.GetInterpolatorIndex();
         if ( interp_index!=InterpConfig::conforming && master_side==0 )
         {
            // Interpolation from fine to coarse
            for (int c = 0; c < vd; ++c)
            {
               MFEM_FOREACH_THREAD(dof,x,nface_dofs)
               {
                  dof_values[dof] = d_x(dof, c, face);
               }
               MFEM_SYNC_THREAD;
               MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
               {
                  double res = 0.0;
                  for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
                  {
                     res += d_interp(dof_in, dof_out, interp_index)*dof_values[dof_in];
                  }
                  d_x(dof_out, c, face) = res;
               }
               MFEM_SYNC_THREAD;
            }
         }
      });*/
   }

   // Gathering of face dofs into element dofs
   const int dofs = nfdofs;
   const auto d_offsets = gather_offsets.Read();
   const auto d_indices = gather_indices.Read();
   if ( m==L2FaceValues::DoubleValued )
   {
      const auto d_x = Reshape(x_interp.Read(), nface_dofs, vd, 2, nf);
      auto d_y = Reshape(y.ReadWrite(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, ndofs,
      {
         const int offset = d_offsets[i];
         const int next_offset = d_offsets[i + 1];
         for (int c = 0; c < vd; ++c)
         {
            double dof_value = 0;
            for (int j = offset; j < next_offset; ++j)
            {
               int idx_j = d_indices[j];
               const bool isE1 = idx_j < dofs;
               idx_j = isE1 ? idx_j : idx_j - dofs;

               const double value =  isE1 ?
               d_x(idx_j % nface_dofs, c, 0, idx_j / nface_dofs)
               :
               d_x(idx_j % nface_dofs, c, 1, idx_j / nface_dofs);
               assert(idx_j / nface_dofs < nf);
               dof_value += value;
            }
            d_y(t?c:i,t?i:c) += dof_value;
         }
      });
   }
   else // Single valued
   {
      auto d_x = Reshape(x_interp.Read(), nface_dofs, vd, nf);
      auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, ndofs,
      {
         const int offset = d_offsets[i];
         const int next_offset = d_offsets[i + 1];
         for (int c = 0; c < vd; ++c)
         {
            double dof_value = 0;
            for (int j = offset; j < next_offset; ++j)
            {
               int idx_j = d_indices[j];
               dof_value +=  d_x(idx_j % nface_dofs, c, idx_j / nface_dofs);
            }
            d_y(t?c:i,t?i:c) += dof_value;
         }
      });
   }
}

void ParNCL2FaceRestriction::FillI(SparseMatrix &mat,
                                   const bool keep_nbr_block) const
{
   MFEM_ABORT("Not yet implemented.");
}

void ParNCL2FaceRestriction::FillI(SparseMatrix &mat,
                                   SparseMatrix &face_mat) const
{
   MFEM_ABORT("Not yet implemented.");
}

void ParNCL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                          SparseMatrix &mat,
                                          const bool keep_nbr_block) const
{
   MFEM_ABORT("Not yet implemented.");
}

void ParNCL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                          SparseMatrix &mat,
                                          SparseMatrix &face_mat) const
{
   MFEM_ABORT("Not yet implemented.");
}

void ParNCL2FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      gather_offsets[i] = 0;
   }

   // Computation of scatter and offsets indices
   int f_ind=0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsLocal() && face.IsNonConformingMaster() )
      {
         // We skip local non-conforming master faces as they are treated by the
         // local non-conforming slave faces.
         assert(face.IsNonConformingMaster());
         continue;
      }
      else if ( type==FaceType::Interior && face.IsInterior() )
      {
         if ( face.IsConforming() )
         {
            interpolations.RegisterFaceConformingInterpolation(face,f_ind);
            SetFaceDofsScatterIndices1(face,f_ind);
            if ( m==L2FaceValues::DoubleValued )
            {
               if ( face.IsShared() )
               {
                  PermuteAndSetSharedFaceDofsScatterIndices2(face,f_ind);
               }
               else
               {
                  PermuteAndSetFaceDofsScatterIndices2(face,f_ind);
               }
            }
         }
         else // Non-conforming face
         {
            interpolations.RegisterFaceCoarseToFineInterpolation(face,f_ind);
            SetFaceDofsScatterIndices1(face,f_ind);
            if ( m==L2FaceValues::DoubleValued )
            {
               if ( face.IsShared() )
               {
                  PermuteAndSetSharedFaceDofsScatterIndices2(face,f_ind);
               }
               else // local non-conforming slave
               {
                  PermuteAndSetFaceDofsScatterIndices2(face,f_ind);
               }
            }
         }
         f_ind++;
      }
      else if (type==FaceType::Boundary && face.IsBoundary())
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if ( m==L2FaceValues::DoubleValued )
         {
            SetBoundaryDofsScatterIndices2(face,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );

   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      gather_offsets[i] += gather_offsets[i - 1];
   }

   // Transform the interpolation matrix map into a contiguous memory structure.
   interpolations.LinearizeInterpolatorMapIntoVector();
}

void ParNCL2FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsLocal() && face.IsNonConformingMaster() )
      {
         // We skip local non-conforming master faces as they are treated by the
         // local non-conforming slave faces.
         assert(face.IsNonConformingMaster());
         continue;
      }
      else if ( face.IsOfFaceType(type) )
      {
         SetFaceDofsGatherIndices1(face,f_ind);
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             face.IsLocal()
            )
         {
            PermuteAndSetFaceDofsGatherIndices2(face,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );

   // Switch back offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      gather_offsets[i] = gather_offsets[i - 1];
   }
   gather_offsets[0] = 0;
}

} // namespace mfem

#endif
