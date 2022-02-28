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

#ifndef MFEM_SUBMESH
#define MFEM_SUBMESH

#include "mesh.hpp"

namespace mfem
{
// SubMesh represents a topological subset of a Mesh.
class SubMesh : public Mesh
{
public:
   // Indicator from which part of the parent Mesh the SubMesh is created.
   enum From
   {
      Domain,
      Boundary
   };

   SubMesh(SubMesh &&submesh) = default;

   SubMesh& operator=(SubMesh &&submesh) = delete;

   SubMesh& operator=(const SubMesh &submesh) = delete;

   SubMesh() = delete;

   // Create a domain SubMesh from it's parent.
   //
   // The attributes have to mark exactly one connected subset of the parent
   // Mesh.
   static SubMesh CreateFromDomain(const Mesh &parent,
                                   Array<int> domain_attributes);

   // Create a boundary SubMesh from it's parent.
   //
   // The attributes have to mark exactly one connected subset of the parent
   // Mesh.
   static SubMesh CreateFromBoundary(const Mesh &parent,
                                     Array<int> boundary_attributes);

   const Mesh* GetParent()
   {
      return &parent_;
   }

   const From GetFrom()
   {
      return from_;
   }

   const Array<int>& GetParentElementIDMap() const
   {
      return parent_element_ids_;
   }

   const Array<int>& GetParentVertexIDMap() const
   {
      return parent_vertex_ids_;
   }

   // Transfer the dofs of a GridFunction on a SubMesh to a GridFunction on the
   // Mesh.
   static void Transfer(const GridFunction &src, GridFunction &dst);

   ~SubMesh();

private:
   // Private constructor
   SubMesh(const Mesh &parent, From from, Array<int> attributes);

   // The parent Mesh
   const Mesh &parent_;

   From from_;

   Array<int> attributes_;
   Array<int> element_ids_;

   // Mapping from submesh element ids (index of the array), to
   // the parent element ids.
   Array<int> parent_element_ids_;

   // Mapping from submesh vertex ids (index of the array), to
   // the parent vertex ids.
   Array<int> parent_vertex_ids_;
};
};

#endif
