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

#include <iostream>
#include <unordered_set>
#include <algorithm>
#include "psubmesh.hpp"
#include "submesh_utils.hpp"
#include "segment.hpp"

using namespace mfem;

ParSubMesh ParSubMesh::CreateFromDomain(ParMesh &parent,
                                        Array<int> &domain_attributes)
{
   return ParSubMesh(parent, SubMesh::From::Domain, domain_attributes);
}

ParSubMesh::ParSubMesh(ParMesh &parent, SubMesh::From from,
                       Array<int> &attributes) : parent_(parent), from_(from), attributes_(attributes)
{
   MyComm = parent.GetComm();
   NRanks = parent.GetNRanks();
   MyRank = parent.GetMyRank();

   if (from == SubMesh::From::Domain)
   {
      InitMesh(parent.Dimension(), parent.SpaceDimension(), 0, 0, 0);

      Array<int> vtxids, elids;
      std::tie(vtxids, elids) = SubMeshUtils::AddElementsToMesh(parent_, *this,
                                                                attributes_);
      parent_vertex_ids_ = vtxids;
      parent_element_ids_ = elids;

      // Don't let boundary elements get generated automatically. This would
      // generate boundary elements on each rank locally, which is topologically
      // wrong for the distributed SubMesh.
      FinalizeTopology(false);
   }

   parent_to_submesh_vertex_ids_.SetSize(parent_.GetNV());
   parent_to_submesh_vertex_ids_ = -1;
   for (int i = 0; i < parent_vertex_ids_.Size(); i++)
   {
      parent_to_submesh_vertex_ids_[parent_vertex_ids_[i]] = i;
   }

   parent_to_submesh_edge_ids_.SetSize(parent.GetNEdges());
   parent_to_submesh_edge_ids_ = -1;
   for (int i = 0; i < parent_edge_ids_.Size(); i++)
   {
      parent_to_submesh_edge_ids_[parent_edge_ids_[i]] = i;
   }

   ListOfIntegerSets groups;
   IntegerSet group;
   // the first group is the local one
   group.Recreate(1, &MyRank);
   groups.Insert(group);

   // Every rank containing elements of the SubMesh attributes now has a local
   // SubMesh. We have to connect the local meshes and assign global boundaries
   // correctly.

   // Array of integer bitfields to indicate if rank X (bit location) has shared
   // vtx Y (array index).
   //
   // Example with 4 ranks and X shared vertices.
   // * R0-R3 indicate ranks 0 to 3
   // * v0-v3 indicate vertices 0 to 3
   // The array is used as follows (only relevant bits shown):
   //
   // rhvtx[0] = [0...0 1 0 1] Rank 0 and 2 have shared vertex 0
   // rhvtx[1] = [0...0 1 1 1] Rank 0, 1 and 2 have shared vertex 1
   // rhvtx[2] = [0...0 0 1 1] Rank 0 and 1 have shared vertex 2
   // rhvtx[3] = [0...1 0 1 0] Rank 1 and 3 have shared vertex 3. Corner case
   // which shows that a rank can contribute the shared vertex, but the adjacent
   // element or edge might not be included in the relevant SubMesh.
   //
   //  +--------------+--------------+...
   //  |              |v0            |
   //  |      R0      |      R2      |     R3
   //  |              |              |
   //  +--------------+--------------+...
   //  |              |v1            |
   //  |      R0      |      R1      |     R3
   //  |              |v2            |v3
   //  +--------------+--------------+...
   Array<int> rhvtx, rhe;
   FindSharedVerticesRanks(rhvtx);
   AppendSharedVerticesGroups(groups, rhvtx);

   FindSharedEdgesRanks(rhe);
   AppendSharedEdgesGroups(groups, rhe);

   // build the group communication topology
   gtopo.SetComm(MyComm);
   gtopo.Create(groups, 822);
   int ngroups = groups.Size()-1;

   int svert_ct, sedge_ct;
   BuildVertexGroup(ngroups, rhvtx, svert_ct);
   BuildEdgeGroup(ngroups, rhe, sedge_ct);

   // TODO: BuildFaceGroup
   {
      group_stria.MakeI(ngroups);
      group_squad.MakeI(ngroups);
      group_stria.MakeJ();
      group_squad.MakeJ();
      group_stria.ShiftUpI();
      group_squad.ShiftUpI();
   }

   BuildSharedVerticesMapping(svert_ct, rhvtx);
   BuildSharedEdgesMapping(sedge_ct, rhe);
   // TODO: BuildSharedFacesMapping

   Finalize();
}

void ParSubMesh::FindSharedVerticesRanks(Array<int> &rhvtx)
{
   // create a GroupCommunicator on the shared vertices
   GroupCommunicator svert_comm(parent_.gtopo);
   parent_.GetSharedVertexCommunicator(svert_comm);
   // Number of shared vertices
   int nsvtx = svert_comm.GroupLDofTable().Size_of_connections();

   rhvtx.SetSize(nsvtx);
   rhvtx = 0;

   // On each rank of the group, locally determine if the shared vertex is in
   // the SubMesh.
   for (int g = 1, sv = 0; g < parent_.GetNGroups(); g++)
   {
      const int group_sz = parent_.gtopo.GetGroupSize(g);
      MFEM_VERIFY(group_sz <= 8*sizeof(int), // 32
                  "Group size too large. Groups with more than 32 ranks are not supported, yet.");
      const int* group_lproc = parent_.gtopo.GetGroup(g);

      const int* my_group_id_ptr = std::find(group_lproc, group_lproc+group_sz, 0);
      MFEM_ASSERT(my_group_id_ptr != group_lproc+group_sz, "internal error");

      const int my_group_id = my_group_id_ptr-group_lproc;

      for (int gv = 0; gv < parent_.GroupNVertices(g); gv++, sv++)
      {
         int plvtx = parent_.GroupVertex(g, gv);
         int submesh_vertex_id = parent_to_submesh_vertex_ids_[plvtx];
         if (submesh_vertex_id != -1)
         {
            rhvtx[sv] |= 1 << my_group_id;
         }
      }
   }

   // Compute the sum on the root rank and broadcast the result to all ranks.
   svert_comm.Reduce(rhvtx, GroupCommunicator::Sum);
   svert_comm.Bcast<int>(rhvtx, 0);
}

void ParSubMesh::FindSharedEdgesRanks(Array<int> &rhe)
{
   DSTable v2v(parent_.GetNV());
   parent_.GetVertexToVertexTable(v2v);
   for (int i = 0; i < NumOfEdges; i++)
   {
      Array<int> lv;
      GetEdgeVertices(i, lv);

      // Find vertices/edge in parent mesh
      int parent_edge_id = v2v(parent_vertex_ids_[lv[0]], parent_vertex_ids_[lv[1]]);
      parent_edge_ids_.Append(parent_edge_id);
   }

   // create a GroupCommunicator on the shared edges
   GroupCommunicator sedge_comm(parent_.gtopo);
   parent_.GetSharedEdgeCommunicator(sedge_comm);

   int nsedge = sedge_comm.GroupLDofTable().Size_of_connections();

   // see rhvtx description
   rhe.SetSize(nsedge);
   rhe = 0;

   // On each rank of the group, locally determine if the shared edge is in
   // the SubMesh.
   for (int g = 1, se = 0; g < parent_.GetNGroups(); g++)
   {
      const int group_sz = parent_.gtopo.GetGroupSize(g);
      MFEM_VERIFY(group_sz <= 8*sizeof(int), // 32
                  "Group size too large. Groups with more than 32 ranks are not supported, yet.");
      const int* group_lproc = parent_.gtopo.GetGroup(g);

      const int* my_group_id_ptr = std::find(group_lproc, group_lproc+group_sz, 0);
      MFEM_ASSERT(my_group_id_ptr != group_lproc+group_sz, "internal error");

      const int my_group_id = my_group_id_ptr-group_lproc;

      for (int ge = 0; ge < parent_.GroupNEdges(g); ge++, se++)
      {
         int ple, o;
         parent_.GroupEdge(g, ge, ple, o);
         int submesh_edge_id = parent_to_submesh_edge_ids_[ple];
         if (submesh_edge_id != -1)
         {
            rhe[se] |= 1 << my_group_id;
         }
      }
   }

   // Compute the sum on the root rank and broadcast the result to all ranks.
   sedge_comm.Reduce(rhe, GroupCommunicator::Sum);
   sedge_comm.Bcast<int>(rhe, 0);
}

void ParSubMesh::AppendSharedVerticesGroups(ListOfIntegerSets &groups,
                                            Array<int> &rhvtx)
{
   IntegerSet group;

   for (int g = 1, sv = 0; g < parent_.GetNGroups(); g++)
   {
      const int group_sz = parent_.gtopo.GetGroupSize(g);
      MFEM_VERIFY(group_sz <= 8*sizeof(int), // 32
                  "Group size too large. Groups with more than 32 ranks are not supported, yet.");
      const int* group_lproc = parent_.gtopo.GetGroup(g);

      const int* my_group_id_ptr = std::find(group_lproc, group_lproc+group_sz, 0);
      MFEM_ASSERT(my_group_id_ptr != group_lproc+group_sz, "internal error");

      const int my_group_id = my_group_id_ptr-group_lproc;

      for (int gv = 0; gv < parent_.GroupNVertices(g); gv++, sv++)
      {
         // Returns the parents local vertex id
         int plvtx = parent_.GroupVertex(g, gv);
         int submesh_vtx = parent_to_submesh_vertex_ids_[plvtx];

         // Reusing the `rhvtx` array as shared vertex to group array.
         if (submesh_vtx == -1)
         {
            // parent shared vertex is not in SubMesh
            rhvtx[sv] = -1;
         }
         else if (rhvtx[sv] & ~(1 << my_group_id))
         {
            // shared vertex is present on this rank and others

            // determine which other ranks have the shared vertex
            Array<int> &ranks = group;
            ranks.SetSize(0);
            for (int i = 0; i < group_sz; i++)
            {
               if ((rhvtx[sv] >> i) & 1)
               {
                  ranks.Append(parent_.gtopo.GetNeighborRank(group_lproc[i]));
               }
            }
            MFEM_ASSERT(ranks.Size() >= 2, "internal error");

            rhvtx[sv] = groups.Insert(group) - 1;
         }
         else
         {
            // previously shared vertex is only present on this rank
            rhvtx[sv] = -1;
         }
      }
   }
}

void ParSubMesh::AppendSharedEdgesGroups(ListOfIntegerSets &groups,
                                         Array<int> &rhe)
{
   IntegerSet group;

   for (int g = 1, se = 0; g < parent_.GetNGroups(); g++)
   {
      const int group_sz = parent_.gtopo.GetGroupSize(g);
      MFEM_VERIFY(group_sz <= 8*sizeof(int), // 32
                  "Group size too large. Groups with more than 32 ranks are not supported, yet.");
      const int* group_lproc = parent_.gtopo.GetGroup(g);

      const int* my_group_id_ptr = std::find(group_lproc, group_lproc+group_sz, 0);
      MFEM_ASSERT(my_group_id_ptr != group_lproc+group_sz, "internal error");

      const int my_group_id = my_group_id_ptr-group_lproc;

      for (int ge = 0; ge < parent_.GroupNEdges(g); ge++, se++)
      {
         int ple, o;
         parent_.GroupEdge(g, ge, ple, o);
         int submesh_edge = parent_to_submesh_edge_ids_[ple];

         // Reusing the `rhe` array as shared edge to group array.
         if (submesh_edge == -1)
         {
            // parent shared edge is not in SubMesh
            rhe[se] = -1;
         }
         else if (rhe[se] & ~(1 << my_group_id))
         {
            // shared edge is present on this rank and others

            // determine which other ranks have the shared edge
            Array<int> &ranks = group;
            ranks.SetSize(0);
            for (int i = 0; i < group_sz; i++)
            {
               if ((rhe[se] >> i) & 1)
               {
                  ranks.Append(parent_.gtopo.GetNeighborRank(group_lproc[i]));
               }
            }
            MFEM_ASSERT(ranks.Size() >= 2, "internal error");

            rhe[se] = groups.Insert(group) - 1;
         }
         else
         {
            // previously shared edge is only present on this rank
            rhe[se] = -1;
         }
      }
   }
}

void ParSubMesh::BuildVertexGroup(int ngroups, const Array<int>& rhvtx,
                                  int& svert_ct)
{
   group_svert.MakeI(ngroups);
   for (int i = 0; i < rhvtx.Size(); i++)
   {
      if (rhvtx[i] >= 0)
      {
         group_svert.AddAColumnInRow(rhvtx[i]);
      }
   }

   group_svert.MakeJ();
   svert_ct = 0;
   for (int i = 0; i < rhvtx.Size(); i++)
   {
      if (rhvtx[i] >= 0)
      {
         group_svert.AddConnection(rhvtx[i], svert_ct++);
      }
   }
   group_svert.ShiftUpI();
}

void ParSubMesh::BuildEdgeGroup(int ngroups, const Array<int>& rhe,
                                int& sedge_ct)
{
   group_sedge.MakeI(ngroups);
   for (int i = 0; i < rhe.Size(); i++)
   {
      if (rhe[i] >= 0)
      {
         group_sedge.AddAColumnInRow(rhe[i]);
      }
   }

   group_sedge.MakeJ();
   sedge_ct = 0;
   for (int i = 0; i < rhe.Size(); i++)
   {
      if (rhe[i] >= 0)
      {
         group_sedge.AddConnection(rhe[i], sedge_ct++);
      }
   }
   group_sedge.ShiftUpI();
}

void ParSubMesh::BuildSharedVerticesMapping(const int svert_ct,
                                            const Array<int>& rhvtx)
{
   svert_lvert.Reserve(svert_ct);

   for (int g = 1, sv = 0; g < parent_.GetNGroups(); g++)
   {
      for (int gv = 0; gv < parent_.GroupNVertices(g); gv++, sv++)
      {
         // Returns the parents local vertex id
         int plvtx = parent_.GroupVertex(g, gv);
         int submesh_vtx_id = parent_to_submesh_vertex_ids_[plvtx];
         if ((submesh_vtx_id == -1) || (rhvtx[sv] == -1))
         {
            // parent shared vertex is not in SubMesh or is not shared
         }
         else
         {
            svert_lvert.Append(submesh_vtx_id);
         }
      }
   }
}

void ParSubMesh::BuildSharedEdgesMapping(const int sedges_ct,
                                         const Array<int>& rhe)
{
   shared_edges.Reserve(sedges_ct);
   sedge_ledge.Reserve(sedges_ct);

   for (int g = 1, se = 0; g < parent_.GetNGroups(); g++)
   {
      for (int ge = 0; ge < parent_.GroupNEdges(g); ge++, se++)
      {
         int ple, o;
         parent_.GroupEdge(g, ge, ple, o);
         int submesh_edge_id = parent_to_submesh_edge_ids_[ple];
         if ((submesh_edge_id = -1) || rhe[se] == -1)
         {
            // parent shared edge is not in SubMesh or is not shared
         }
         else
         {
            Array<int> vert;
            GetEdgeVertices(submesh_edge_id, vert);

            shared_edges.Append(new Segment(vert[0], vert[1], 1));
            sedge_ledge.Append(submesh_edge_id);
         }
      }
   }
}

void ParSubMesh::Transfer(const ParGridFunction &src, ParGridFunction &dst)
{
   Array<int> src_vdofs;
   Array<int> dst_vdofs;
   Vector vec;

   if (dynamic_cast<const ParSubMesh *>(src.ParFESpace()->GetParMesh()) != nullptr)
   {
      MFEM_ABORT("not implemented yet");
   }
   else if (dynamic_cast<const ParSubMesh *>(dst.ParFESpace()->GetParMesh()) !=
            nullptr)
   {
      // ParMesh to ParSubMesh transfer
      ParMesh *src_mesh = src.ParFESpace()->GetParMesh();
      ParSubMesh *dst_mesh = static_cast<ParSubMesh *>
                             (dst.ParFESpace()->GetParMesh());
      MFEM_ASSERT(dst_mesh->GetParent() == src_mesh,
                  "The Meshes of the specified GridFunction are not related in a Mesh -> SubMesh relationship.");

      auto &parent_element_ids = dst_mesh->GetParentElementIDMap();

      IntegrationPointTransformation Tr;
      DenseMatrix vals, vals_transpose;
      for (int i = 0; i < dst_mesh->GetNE(); i++)
      {
         dst.ParFESpace()->GetElementVDofs(i, dst_vdofs);
         if (dst_mesh->GetFrom() == SubMesh::From::Domain)
         {
            src.ParFESpace()->GetElementVDofs(parent_element_ids[i], src_vdofs);
         }
         else if (dst_mesh->GetFrom() == SubMesh::From::Boundary)
         {
            MFEM_ABORT("TODO");
         }
         src.GetSubVector(src_vdofs, vec);
         dst.SetSubVector(dst_vdofs, vec);
      }
   }
   else
   {
      MFEM_ABORT("Trying to do a transfer between ParGridFunctions but none of them is defined on a ParSubMesh");
   }
}