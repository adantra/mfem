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

#ifndef MFEM_PSUBMESH
#define MFEM_PSUBMESH

#include "pmesh.hpp"
#include "submesh.hpp"

namespace mfem
{
class ParSubMesh : public ParMesh
{
public:
   ParSubMesh(ParSubMesh &&ParSubMesh) = default;

   ParSubMesh &operator=(ParSubMesh &&ParSubMesh) = delete;

   ParSubMesh &operator=(const ParSubMesh &ParSubMesh) = delete;

   ParSubMesh() = delete;

   // Create a domain ParSubMesh from it's parent.
   //
   // The attributes have to mark exactly one connected subset of the parent
   // Mesh.
   static ParSubMesh CreateFromDomain(ParMesh &parent,
                                      Array<int> &domain_attributes);

private:
   // Private constructor
   ParSubMesh(ParMesh &parent, SubMesh::From from, Array<int> &attributes);

   void FindSharedVerticesRanks(Array<int> &rhvtx);

   void FindSharedEdgesRanks(Array<int> &rhe);

   void AppendSharedVerticesGroups(ListOfIntegerSets &groups, Array<int> &rhvtx);

   void AppendSharedEdgesGroups(ListOfIntegerSets &groups, Array<int> &rhe);

   void BuildVertexGroup(int ngroups, const Array<int>& rhvtx, int& svert_ct);

   void BuildEdgeGroup(int ngroups, const Array<int>& rhe, int& sedge_ct);

   void BuildSharedVerticesMapping(const int svert_ct, const Array<int>& rhvtx);

   void BuildSharedEdgesMapping(const int sedges_ct, const Array<int>& rhe);

   // The parent Mesh
   ParMesh &parent_;

   SubMesh::From from_;

   Array<int> attributes_;
   Array<int> element_ids_;

   // Mapping from submesh element ids (index of the array), to
   // the parent element ids.
   Array<int> parent_element_ids_;

   // Mapping from submesh vertex ids (index of the array), to
   // the parent vertex ids.
   Array<int> parent_vertex_ids_;

   Array<int> parent_edge_ids_;

   Array<int> parent_to_submesh_vertex_ids_;

   Array<int> parent_to_submesh_edge_ids_;
};
};

#endif