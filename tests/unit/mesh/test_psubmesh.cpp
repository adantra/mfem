#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

#ifdef MFEM_USE_MPI

TEST_CASE("ParSubMesh", "[Parallel],[ParSubMesh]")
{
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, true, 1.0,
                                     1.0,
                                     false);
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      Element *el = mesh.GetElement(i);
      el->SetAttribute(1);

      Array<int> vertices;
      el->GetVertices(vertices);

      for (int j = 0; j < vertices.Size(); j++)
      {
         double *coords = mesh.GetVertex(vertices[j]);

         if (coords[0] >= 0.125 &&
             coords[0] <= 0.5 &&
             coords[1] >= 0.125 &&
             coords[1] <= 0.5)
         {
            el->SetAttribute(2);
         }
      }
   }
   mesh.SetAttributes();

   // Deform original mesh
   // mesh.EnsureNodes();
   // mesh.SetCurvature(2);

   auto node_movement_coeff = VectorFunctionCoefficient(mesh.Dimension(),
                                                        [](const Vector &coords, Vector &u)
   {
      double x = coords(0);
      double y = coords(1);

      u(0) = x;
      u(1) = y + 0.05 * sin(x * 2.0 * M_PI);
   });

   mesh.Transform(node_movement_coeff);

   // Create parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);


   auto fec = L2_FECollection(2, 2, BasisType::GaussLobatto);
   FiniteElementSpace parent_fes(&pmesh, &fec);
   GridFunction parent_gf(&parent_fes);
   parent_gf = pmesh.GetMyRank();

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   auto submesh = ParSubMesh::CreateFromDomain(pmesh, subdomain_attributes);

   // FiniteElementSpace submesh_fes(&submesh, &fec);
   // GridFunction gf(&submesh_fes);
   // gf = 1.0;

   // SubMesh::Transfer(gf, parent_gf);

   int num_procs, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   char vishost[] = "128.15.198.77";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << rank << "\n";
   sol_sock.precision(8);
   sol_sock << "mesh\n" << submesh << std::flush;
   // sol_sock << "mesh\n" << pmesh << std::flush;
   // sol_sock << "solution\n" << pmesh << parent_gf << std::flush;
   sol_sock << "keys ennrRlm\n" << std::flush;
}

#endif // MFEM_USE_MPI
