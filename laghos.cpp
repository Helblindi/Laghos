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
//
//                     __                __
//                    / /   ____  ____  / /_  ____  _____
//                   / /   / __ `/ __ `/ __ \/ __ \/ ___/
//                  / /___/ /_/ / /_/ / / / / /_/ (__  )
//                 /_____/\__,_/\__, /_/ /_/\____/____/
//                             /____/
//
//             High-order Lagrangian Hydrodynamics Miniapp
//
// Laghos(LAGrangian High-Order Solver) is a miniapp that solves the
// time-dependent Euler equation of compressible gas dynamics in a moving
// Lagrangian frame using unstructured high-order finite element spatial
// discretization and explicit high-order time-stepping. Laghos is based on the
// numerical algorithm described in the following article:
//
//    V. Dobrev, Tz. Kolev and R. Rieben, "High-order curvilinear finite element
//    methods for Lagrangian hydrodynamics", SIAM Journal on Scientific
//    Computing, (34) 2012, pp. B606–B641, https://doi.org/10.1137/120864672.
//
// Test problems:
//    p = 0  --> Taylor-Green vortex (smooth problem).
//    p = 1  --> Sedov blast.
//    p = 2  --> 1D Sod shock tube.
//    p = 3  --> Triple point.
//    p = 4  --> Gresho vortex (smooth problem).
//    p = 5  --> 2D Riemann problem, config. 12 of doi.org/10.1002/num.10025
//    p = 6  --> 2D Riemann problem, config.  6 of doi.org/10.1002/num.10025
//    p = 7  --> 2D Rayleigh-Taylor instability problem.
//
// Sample runs: see README.md, section 'Verification of Results'.
//
// Combinations resulting in 3D uniform Cartesian MPI partitionings of the mesh:
// -m data/cube01_hex.mesh   -pt 211 for  2 / 16 / 128 / 1024 ... tasks.
// -m data/cube_922_hex.mesh -pt 921 for    / 18 / 144 / 1152 ... tasks.
// -m data/cube_522_hex.mesh -pt 522 for    / 20 / 160 / 1280 ... tasks.
// -m data/cube_12_hex.mesh  -pt 311 for  3 / 24 / 192 / 1536 ... tasks.
// -m data/cube01_hex.mesh   -pt 221 for  4 / 32 / 256 / 2048 ... tasks.
// -m data/cube_922_hex.mesh -pt 922 for    / 36 / 288 / 2304 ... tasks.
// -m data/cube_522_hex.mesh -pt 511 for  5 / 40 / 320 / 2560 ... tasks.
// -m data/cube_12_hex.mesh  -pt 321 for  6 / 48 / 384 / 3072 ... tasks.
// -m data/cube01_hex.mesh   -pt 111 for  8 / 64 / 512 / 4096 ... tasks.
// -m data/cube_922_hex.mesh -pt 911 for  9 / 72 / 576 / 4608 ... tasks.
// -m data/cube_522_hex.mesh -pt 521 for 10 / 80 / 640 / 5120 ... tasks.
// -m data/cube_12_hex.mesh  -pt 322 for 12 / 96 / 768 / 6144 ... tasks.

#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include "laghos_solvers.hpp"
#include "laghos_solver.hpp"
#include "laglos_solver.hpp"
#include "test_problems_include.h"
#include "limiter.h"
// #include "ode_idp.hpp"

using std::cout;
using std::endl;
using namespace mfem;

// Choice for the problem setup.
static int problem, dim;

// Forward declarations.
double e0(const Vector &);
double rho0(const Vector &);
double gamma_func(const Vector &);
void v0(const Vector &, Vector &);
void MassesAndVolumesAtPosition(const ParGridFunction &u, const GridFunction &x, Vector &el_mass, Vector &el_vol);

static long GetMaxRssMB();
static void display_banner(std::ostream&);
static void Checks(const int ti, const double norm, int &checks);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Print the banner.
   if (Mpi::Root()) { display_banner(cout); }

   // Parse command-line options.
   problem = 1;
   dim = 3;
   const char *mesh_file = "default";
   int rs_levels = 2;
   int rp_levels = 0;
   Array<int> cxyz;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   int order_e_lo = 0; // low-order approximation space
   int order_v_lo = 1;
   bool idp_limit = false;
   int ode_solver_type = 4;
   double t_init = 0.0;
   double t_final = 0.6;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   double ftz_tol = 0.0;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool p_assembly = true;
   bool impose_visc = false;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool pview = false;
   bool gfprint = false;
   const char *basename = "results/";
   int partition_type = 0;
   const char *device = "cpu";
   bool check = false;
   bool mem_usage = false;
   bool fom = false;
   bool gpu_aware_mpi = false;
   int dev = 0;
   double blast_energy = 0.25;
   double blast_position[] = {0.0, 0.0, 0.0};

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&cxyz, "-c", "--cartesian-partitioning",
                  "Use Cartesian partitioning.");
   args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_v_lo, "-oklo", "--order-kinematic-lo",
                  "Order (degree) of the kinematic finite element space of Low order approx.");
   args.AddOption(&order_q, "-oq", "--order-intrule",
                  "Order  of the integration rule.");
   args.AddOption(&idp_limit, "-idp", "--invariant-domain-preserving", "-no-idp", "--no-invariant-domain-preserving",
                  "Use limiter and low order Laglos solver to ensure invariant domain is preserved.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6, 7 - RK2Avg\n\t"
                  "            12 - RK2 IDP, 14 - RK4 IDP.");
   args.AddOption(&t_init, "-ti", "--t-init", "Initial time.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&ftz_tol, "-ftz", "--ftz-tol",
                  "Absolute flush-to-zero tolerance.");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&p_assembly, "-pa", "--partial-assembly", "-fa",
                  "--full-assembly",
                  "Activate 1D tensor-based assembly (partial assembly).");
   args.AddOption(&impose_visc, "-iv", "--impose-viscosity", "-niv",
                  "--no-impose-viscosity",
                  "Use active viscosity terms even for smooth problems.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&pview, "-pview", "--paraview", "-no-pview", "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&partition_type, "-pt", "--partition",
                  "Customized x/y/z Cartesian MPI partitioning of the serial mesh.\n\t"
                  "Here x,y,z are relative task ratios in each direction.\n\t"
                  "Example: with 48 mpi tasks and -pt 321, one would get a Cartesian\n\t"
                  "partition of the serial mesh by (6,4,2) MPI tasks in (x,y,z).\n\t"
                  "NOTE: the serially refined mesh must have the appropriate number\n\t"
                  "of zones in each direction, e.g., the number of zones in direction x\n\t"
                  "must be divisible by the number of MPI tasks in direction x.\n\t"
                  "Available options: 11, 21, 111, 211, 221, 311, 321, 322, 432.");
   args.AddOption(&device, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&check, "-chk", "--checks", "-no-chk", "--no-checks",
                  "Enable 2D checks.");
   args.AddOption(&mem_usage, "-mb", "--mem", "-no-mem", "--no-mem",
                  "Enable memory usage.");
   args.AddOption(&fom, "-f", "--fom", "-no-fom", "--no-fom",
                  "Enable figure of merit output.");
   args.AddOption(&gpu_aware_mpi, "-gam", "--gpu-aware-mpi", "-no-gam",
                  "--no-gpu-aware-mpi", "Enable GPU aware MPI communications.");
   args.AddOption(&dev, "-dev", "--dev", "GPU device to use.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }
   
   std::string basename_LO_str = basename;
   basename_LO_str += "LO/";
   const char *basename_LO = basename_LO_str.c_str();

   // Configure the device from the command line options
   Device backend;
   backend.Configure(device, dev);
   if (Mpi::Root()) { backend.Print(); }
   backend.SetGPUAwareMPI(gpu_aware_mpi);

   /* IDP checks */
   if (idp_limit)
   {
      cout << "checking idp_limit" << endl;
      MFEM_VERIFY(ode_solver_type > 10, "If IDP is desired, must use IDP ODE solver");
      MFEM_VERIFY(order_q >= 4, "IDP requires at least 4th order integration rule");
      if (p_assembly)
      {
         MFEM_ABORT("IDP does not support partial assembly");
      }
   }

   // On all processors, use the default builtin 1D/2D/3D mesh or read the
   // serial one given on the command line.
   Mesh *mesh;
   if (strncmp(mesh_file, "default", 7) != 0)
   {
      mesh = new Mesh(mesh_file, true, true);
   }
   else
   {
      if (dim == 1)
      {
         mesh = new Mesh(Mesh::MakeCartesian1D(2));
         mesh->GetBdrElement(0)->SetAttribute(1);
         mesh->GetBdrElement(1)->SetAttribute(1);
      }
      if (dim == 2)
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/2) ? 2 : 1;
            bel->SetAttribute(attr);
         }
      }
      if (dim == 3)
      {
         mesh = new Mesh(Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/3) ? 3 : (b < 2*NBE/3) ? 1 : 2;
            bel->SetAttribute(attr);
         }
      }
   }
   dim = mesh->Dimension();

   // 1D vs partial assembly sanity check.
   if (p_assembly && dim == 1)
   {
      p_assembly = false;
      if (Mpi::Root())
      {
         cout << "Laghos does not support PA in 1D. Switching to FA." << endl;
      }
   }

   // Refine the mesh in serial to increase the resolution.
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int mesh_NE = mesh->GetNE();
   if (Mpi::Root())
   {
      cout << "Number of zones in the serial mesh: " << mesh_NE << endl;
   }

   // Parallel partitioning of the mesh.
   ParMesh *pmesh = nullptr;
   const int num_tasks = Mpi::WorldSize(); int unit = 1;
   int *nxyz = new int[dim];
   switch (partition_type)
   {
      case 0:
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 11:
      case 111:
         unit = static_cast<int>(floor(pow(num_tasks, 1.0 / dim) + 1e-2));
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 21: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 2) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit;
         break;
      case 31: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit;
         break;
      case 32: // 2D
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit;
         break;
      case 49: // 2D
         unit = static_cast<int>(floor(pow(9 * num_tasks / 4, 1.0 / 2) + 1e-2));
         nxyz[0] = 4 * unit / 9; nxyz[1] = unit;
         break;
      case 51: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 2) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit;
         break;
      case 211: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 221: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 4, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 311: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 321: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 6, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 322: // 3D.
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 432: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 3 * unit / 2; nxyz[2] = unit;
         break;
      case 511: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 521: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 10, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 522: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 20, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
         break;
      case 911: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 9, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 921: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 18, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 922: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 36, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
         break;
      default:
         if (myid == 0)
         {
            cout << "Unknown partition type: " << partition_type << '\n';
         }
         delete mesh;
         MPI_Finalize();
         return 3;
   }
   int product = 1;
   for (int d = 0; d < dim; d++) { product *= nxyz[d]; }
   const bool cartesian_partitioning = (cxyz.Size()>0)?true:false;
   if (product == num_tasks || cartesian_partitioning)
   {
      if (cartesian_partitioning)
      {
         int cproduct = 1;
         for (int d = 0; d < dim; d++) { cproduct *= cxyz[d]; }
         MFEM_VERIFY(!cartesian_partitioning || cxyz.Size() == dim,
                     "Expected " << mesh->SpaceDimension() << " integers with the "
                     "option --cartesian-partitioning.");
         MFEM_VERIFY(!cartesian_partitioning || num_tasks == cproduct,
                     "Expected cartesian partitioning product to match number of ranks.");
      }
      int *partitioning = cartesian_partitioning ?
                          mesh->CartesianPartitioning(cxyz):
                          mesh->CartesianPartitioning(nxyz);
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
      delete [] partitioning;
   }
   else
   {
      if (myid == 0)
      {
         cout << "Non-Cartesian partitioning through METIS will be used.\n";
#ifndef MFEM_USE_METIS
         cout << "MFEM was built without METIS. "
              << "Adjust the number of tasks to use a Cartesian split." << endl;
#endif
      }
#ifndef MFEM_USE_METIS
      return 1;
#endif
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   }
   delete [] nxyz;
   delete mesh;

   // Refine the mesh further in parallel to increase the resolution.
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   int NE = pmesh->GetNE(), ne_min, ne_max;
   MPI_Reduce(&NE, &ne_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&NE, &ne_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
   double hmin, hmax, kmin, kmax;
   pmesh->GetCharacteristics(hmin, hmax, kmin, kmax);
   if (myid == 0)
   { cout << "Zones min/max: " << ne_min << " " << ne_max << endl; }

   // Define the low order mesh
   ParMesh *pmesh_lo = NULL;
   if (order_e > 0)
   {
      pmesh_lo = new ParMesh(ParMesh::MakeRefined(*pmesh, order_e + 1, BasisType::ClosedUniform));
   }
   else
   {
      pmesh_lo = new ParMesh(*pmesh);
   }

   // Set up problem
   MFEM_WARNING("Will not get proper results from 3d tests.\n");
   hydroLO::ProblemBase * problem_class = NULL;
   switch (problem)
   {
      case 0: // Taylor-Green
         problem_class = new hydroLO::TaylorGreenProblem(dim);
         break;
      case 1: // Sedov
         problem_class = new hydroLO::SedovLLNLProblem(dim);
         break;
      case 2: // Sod
         problem_class = new hydroLO::SodProblem(dim);
         break;
      case 3: // Triple Point
         problem_class = new hydroLO::TriplePoint(dim);
         break;
      case 4: // gresho vortex
      case 5: // 2D Riemann problem
      case 6: // 2D Riemann problem
      case 7: // 2D Rayleigh-Taylor instability
         MFEM_ABORT("Not implemented.\n");
      case 8: // Radial Sod
         problem_class = new hydroLO::SodRadial(dim);
         break;
      case 9: // Isentropic Vortex, stationary center
         problem_class = new hydroLO::IsentropicVortex(dim);
         break;
      case 10: // Noh
         problem_class = new hydroLO::NohProblem(dim);
         break;
      case 11: // Saltzmann
         problem_class = new hydroLO::SaltzmannProblem(dim);
         break;
      /* VDW */
      case 12:
         problem_class = new hydroLO::VdwTest1(dim);
         break;
      case 13:
         problem_class = new hydroLO::VdwTest2(dim);
         break;
      case 14:
         problem_class = new hydroLO::VdwTest3(dim);
         break;
      case 15:
         problem_class = new hydroLO::VdwTest4(dim);
         break;
      case 16: // Kidder shell
         problem_class = new hydroLO::KidderProblem(dim);
         break;
      case 17: // Kidder ball
         problem_class = new hydroLO::KidderBallProblem(dim);
         break;
      case 18: // ICF
         problem_class = new hydroLO::ICFProblem(dim);
         break;
      case 21: // Sedov
      {
         MFEM_ABORT("Not implemented\n");
         // assert(hmin == hmax);
         // Vector params(2);
         // params[0] = hmax, params[1] = pmesh->GetElementVolume(0);

         // problem_class = new hydroLO::SedovProblem(dim);
         // problem_class->update(params, t_init);
         // // TODO: Will need to modify initialization of internal energy
         // //       if distorted meshes are used.
         // break;
      }
      case 40: // Smooth
         problem_class = new hydroLO::SmoothWave(dim);
         break;
      case 41: // Lax
         problem_class = new hydroLO::LaxProblem(dim);
         break;
      case 42: // Leblanc
         problem_class = new hydroLO::LeblancProblem(dim);
         break;
      case 43: // Riemann Problem
         problem_class = new hydroLO::RiemannProblem(dim);
         break;
      case 100:
         problem_class = new hydroLO::TestBCs(dim);
         break;
      default:
         MFEM_ABORT("Failed to initiate a problem.\n");
   }

   // Change class variables into static std::functions since virtual static member functions are not an option
   // and Coefficient class requires std::function arguments
   using namespace std::placeholders;
   std::function<double(const Vector &,const double)> sv0_static =
      std::bind(&hydroLO::ProblemBase::sv0, problem_class, std::placeholders::_1, std::placeholders::_2);
   std::function<void(const Vector &, const double, Vector &)> v0_static =
      std::bind(&hydroLO::ProblemBase::v0, problem_class, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
   std::function<double(const Vector &,const double)> ste0_static =
      std::bind(&hydroLO::ProblemBase::ste0, problem_class, std::placeholders::_1, std::placeholders::_2);
   std::function<double(const Vector &,const double)> sie0_static =
      std::bind(&hydroLO::ProblemBase::sie0, problem_class, std::placeholders::_1, std::placeholders::_2);
   std::function<double(const Vector &,const double)> rho0_static =
      std::bind(&hydroLO::ProblemBase::rho0, problem_class, std::placeholders::_1, std::placeholders::_2);
   std::function<double(const Vector &,const double)> p0_static =
      std::bind(&hydroLO::ProblemBase::p0, problem_class, std::placeholders::_1, std::placeholders::_2);
   std::function<double(const Vector &,const double)> gamma_func_static =
      std::bind(&hydroLO::ProblemBase::gamma_func, problem_class, std::placeholders::_1, std::placeholders::_2);

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());

   /* IDP Limiter needs internal data of L2FESpace to be built to access adjacency information */
   if (idp_limit)
   {
      L2FESpace.BuildDofToArrays();
   }

   // Define the parallel finite element spaces for 
   // the low order approximation. We use:
   // - H1 (Q2, continuous) for mesh movement.
   // - L2 (Q0, discontinuous) for state variables
   // - CR/RT for mesh reconstruction at nodes
   H1_FECollection LO_H1FEC(order_v_lo, dim);
   H1_FECollection LO_H1FEC_L(1, dim);
   L2_FECollection LO_L2FEC(order_e_lo, dim, BasisType::Positive);
   FiniteElementCollection * LO_CRFEC;
   if (dim == 1)
   {
      LO_CRFEC = new CrouzeixRaviartFECollection();
   }
   else
   {
      LO_CRFEC = new RT_FECollection(0, dim);
   }

   ParFiniteElementSpace LO_H1FESpace(pmesh_lo, &LO_H1FEC, dim);
   ParFiniteElementSpace LO_H1FESpace_L(pmesh_lo, &LO_H1FEC_L, dim);
   /* Finite element space solely constructed for continuous representation of density field */
   ParFiniteElementSpace LO_L2FESpace(pmesh_lo, &LO_L2FEC);
   ParFiniteElementSpace LO_L2VFESpace(pmesh_lo, &LO_L2FEC, dim);
   ParFiniteElementSpace LO_CRFESpace(pmesh_lo, LO_CRFEC, dim);

   /* Objects used to project HO velocity onto LO */
   ParGridFunction dx(&H1FESpace);
   ParGridFunction dx_LO(&LO_H1FESpace);

   cout << "LO L2 dofs: " << LO_L2FESpace.GetNDofs() << endl;
   cout << "LO H1 dofs: " << LO_H1FESpace.GetNDofs() << endl;
   cout << "pmesh ho # cells: " << pmesh->GetNE() << endl;
   cout << "pmesh_lo # cells: " << pmesh_lo->GetNE() << endl;

   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs, ess_vdofs;
   if (problem_class->has_mv_boundary_conditions())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), dofs_marker, dofs_list;
      for (int d = 0; d < pmesh->Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries,
         // i.e., we must enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, d);
         ess_tdofs.Append(dofs_list);
         H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, d);
         FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
         ess_vdofs.Append(dofs_list);
      }
   }

   // Define the explicit ODE solver used for time integration.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1:
         ode_solver = new ForwardEulerSolver;
         break;
      case 2:
         ode_solver = new RK2Solver(0.5);
         break;
      case 3:
         ode_solver = new RK3SSPSolver;
         break;
      case 4:
         ode_solver = new RK4Solver;
         break;
      case 6:
         ode_solver = new RK6Solver;
         break;
      case 7:
         ode_solver = new RK2AvgSolver;
         break;
      case 11:
         ode_solver = new ForwardEulerIDPSolver();
         break;
      case 12:
         ode_solver = new RK2IDPSolver();
         break;
      case 13:
         ode_solver = new RK3IDPSolver();
         break;
      case 14:
         ode_solver = new RK4IDPSolver();
         break;
      case 16:
         ode_solver = new RK6IDPSolver();
         break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete pmesh;
         delete pmesh_lo;
         MPI_Finalize();
         return 3;
   }

   const HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
   const HYPRE_Int glob_size_h1 = H1FESpace.GlobalTrueVSize();
   const HYPRE_Int glob_size_l2_LO = LO_L2FESpace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of kinematic (position, velocity) dofs: "
           << glob_size_h1 << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_l2 << endl;
      cout << "Number of low order DG0 dofs: "
           << glob_size_l2_LO << endl;
   }

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy
   const int Vsize_l2 = L2FESpace.GetVSize();
   const int Vsize_h1 = H1FESpace.GetVSize();
   Array<int> offset(4);
   offset[0] = 0;
   offset[1] = offset[0] + Vsize_h1;
   offset[2] = offset[1] + Vsize_h1;
   offset[3] = offset[2] + Vsize_l2;
   BlockVector S(offset, Device::GetMemoryType());

   /* The monolithic BlockVector stores unknown fields as:
   *   - 0 -> position
   *   - 1 -> specific volume
   *   - 2 -> velocity (L2V)
   *   - 3 -> speific total energy
   */
   const int Vsize_l2_LO = LO_L2FESpace.GetVSize();
   const int Vsize_l2v_LO = LO_L2VFESpace.GetVSize();
   const int Vsize_h1_LO = LO_H1FESpace.GetVSize();
   Array<int> offset_LO(5);
   offset_LO[0] = 0;
   offset_LO[1] = offset_LO[0] + Vsize_h1_LO;
   offset_LO[2] = offset_LO[1] + Vsize_l2_LO;
   offset_LO[3] = offset_LO[2] + Vsize_l2v_LO;
   offset_LO[4] = offset_LO[3] + Vsize_l2_LO;
   BlockVector S_LO(offset_LO, Device::GetMemoryType());

   // Define GridFunction objects for the position, velocity and specific
   // internal energy. There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, offset[0]);
   v_gf.MakeRef(&H1FESpace, S, offset[1]);
   e_gf.MakeRef(&L2FESpace, S, offset[2]);

   /* Define the low order grid functions*/
   ParGridFunction x_gf_LO, sv_gf_LO, v_gf_LO, ste_gf_LO;
   ParGridFunction rho_gf_LO(&LO_L2FESpace), mc_gf_LO(&LO_L2FESpace);
   ParGridFunction rho_gf(&L2FESpace), rho_gf_limited(&L2FESpace);
   mc_gf_LO = 0.; // if a cells value is 0, mass is conserved
   x_gf_LO.MakeRef(&LO_H1FESpace, S_LO, offset_LO[0]);
   sv_gf_LO.MakeRef(&LO_L2FESpace, S_LO, offset_LO[1]);
   v_gf_LO.MakeRef(&LO_L2VFESpace, S_LO, offset_LO[2]);
   ste_gf_LO.MakeRef(&LO_L2FESpace, S_LO, offset_LO[3]);

   // Initialize x_gf using the starting mesh coordinates.
   pmesh->SetNodalGridFunction(&x_gf);
   pmesh_lo->SetNodalGridFunction(&x_gf_LO);
   // Sync the data location of x_gf with its base, S
   x_gf.SyncAliasMemory(S);
   x_gf_LO.SyncAliasMemory(S_LO);

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0_static);
   v_coeff.SetTime(t_init);
   v_gf.ProjectCoefficient(v_coeff);
   v_gf_LO.ProjectCoefficient(v_coeff);
   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      v_gf(ess_vdofs[i]) = 0.0;
   }
   // Sync the data location of v_gf with its base, S
   v_gf.SyncAliasMemory(S);
   v_gf_LO.SyncAliasMemory(S_LO);

   // Initialize density and specific internal energy values. We interpolate in
   // a non-positive basis to get the correct values at the dofs. Then we do an
   // L2 projection to the positive basis in which we actually compute. The goal
   // is to get a high-order representation of the initial condition. Note that
   // this density is a temporary function and it will not be updated during the
   // time evolution.
   ParGridFunction rho0_gf(&L2FESpace);
   FunctionCoefficient rho0_coeff(rho0_static);
   rho0_coeff.SetTime(t_init);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes);

   L2_FECollection l2_fec_lo(order_e_lo, pmesh_lo->Dimension());
   ParFiniteElementSpace l2_fes_lo(pmesh_lo, &l2_fec_lo);
   ParGridFunction l2_e_LO(&l2_fes_lo);

   l2_rho0_gf.ProjectCoefficient(rho0_coeff);
   rho0_gf.ProjectGridFunction(l2_rho0_gf);
   rho_gf_LO.ProjectCoefficient(rho0_coeff);
   rho_gf.ProjectGridFunction(l2_rho0_gf);

   if (problem == 1)
   {
      // For the Sedov test, we use a delta function at the origin.
      DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                               blast_position[2], blast_energy);
      l2_e.ProjectCoefficient(e_coeff);
      l2_e_LO.ProjectCoefficient(e_coeff);
   }
   else
   {
      FunctionCoefficient sie_coeff(sie0_static);
      FunctionCoefficient ste_coeff(ste0_static);
      sie_coeff.SetTime(t_init);
      ste_coeff.SetTime(t_init);
      l2_e.ProjectCoefficient(sie_coeff);
      l2_e_LO.ProjectCoefficient(ste_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);
   ste_gf_LO.ProjectGridFunction(l2_e_LO);
   // Sync the data location of e_gf with its base, S
   e_gf.SyncAliasMemory(S);
   ste_gf_LO.SyncAliasMemory(S_LO);

   // Project low order sv
   FunctionCoefficient sv_coeff(sv0_static);
   sv_coeff.SetTime(t_init);
   sv_gf_LO.ProjectCoefficient(sv_coeff);
   sv_gf_LO.SyncAliasMemory(S_LO);


   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on function that's constant on the moving mesh.
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma_func_static);
   mat_coeff.SetTime(t_init);
   mat_gf.ProjectCoefficient(mat_coeff);

   // Additional details, depending on the problem.
   int source = 0; bool visc = true, vorticity = false;
   switch (problem)
   {
      case 0: if (pmesh->Dimension() == 2) { source = 1; } visc = false; break;
      case 1: visc = true; break;
      case 2: visc = true; break;
      case 3: visc = true; S.HostRead(); break;
      case 4: visc = false; break;
      case 5: visc = true; break;
      case 6: visc = true; break;
      case 7: source = 2; visc = true; vorticity = true;  break;
      case 8:
      case 9: visc = true; break;
      case 10: visc = true; break;
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 40:
      case 41:
      case 42:
      case 43:
      case 100: visc = true; break;
      default: MFEM_ABORT("Wrong problem specification!");
   }
   if (impose_visc) { visc = true; }

   /* Construct mass vectors */
   ParLinearForm *mHO = new ParLinearForm(&L2FESpace);
   mHO->AddDomainIntegrator(new DomainLFIntegrator(rho0_coeff));
   mHO->Assemble();
   HypreParVector *mHO_hpv = mHO->ParallelAssemble();

   /* Assemble initial masses for low order approximation */
   IntegrationRule LO_ir = IntRules.Get(pmesh_lo->GetElementBaseGeometry(0), 3*LO_H1FESpace.GetOrder(0) + LO_L2FESpace.GetOrder(0) - 1);;
   ParLinearForm *m = new ParLinearForm(&LO_L2FESpace);
   m->AddDomainIntegrator(new DomainLFIntegrator(rho0_coeff, &LO_ir));
   m->Assemble();

   if (idp_limit)
   {
      cout << "checking that masses match..." << endl;
      /* Check that the sum of the HO masses in each HO cell equal the mass in the LO cell */
      bool _mass_match = true;
      cout << "Checking that inital masses match..." << endl;
      /* Need to build coarse to fine table in the case that the meshes are not the same */
      Table coarse_to_fine;
      Array<int> tabrow;

      if (pmesh_lo->GetNE() != NE)
      {
         // MFEM_WARNING("Number of elements in the low order mesh does not match the number of elements in the high order mesh.");
         const CoarseFineTransformations &cf_tr = pmesh_lo->GetRefinementTransforms();
         cf_tr.MakeCoarseToFineTable(coarse_to_fine);
         // cout << "coarse_to_fine table:\n";
         // coarse_to_fine.Print(cout);
      }
      for (int e = 0; e < NE; e++)
      {
         /* Compute LO mass */
         double lo_mass = 0.;
         if (pmesh_lo->GetNE() != NE)
         {
            /* 
            LO mesh is not the same as the HO mesh. Will need to sum 
            up the masses from the LO cells that make up the HO cell
            */
            coarse_to_fine.GetRow(e, tabrow);
            for (int cell_dof_it = 0; cell_dof_it < tabrow.Size(); cell_dof_it++)
            {
               int j = tabrow[cell_dof_it];
               lo_mass += m->Elem(j);
            }
         }
         else {
            lo_mass = m->Elem(e);
         }

         /* Compute HO mass */
         double ho_mass = 0.0;
         Array<int> dofs;
         L2FESpace.GetElementDofs(e, dofs);
         for (int i = 0; i < dofs.Size(); i++)
         {
            const int dof = dofs[i];
            ho_mass += mHO_hpv->Elem(dof);
         }
         
         double val = fabs(lo_mass - ho_mass);
         if (val > 1e-6)
         {
            cout << "val: " << val << endl;
            cout << "!!!!!!!!!!!mass mismatch\n";
            cout << "el: " << e << " LO mass: " << lo_mass
               << " HO mass: " << ho_mass << endl;
            _mass_match = false;
            break;
         }
      }
      // if (!_mass_match)
      // {
      //    MFEM_ABORT("Masses do not initially match!");
      // }
   }

   // MFEM_WARNING("hydro instantiation does not depend on parameter for idp_limit. Hence the mass matrices will NEVER be updated.\n");
   /* Build Low order solver */
   /* Various other parameters */
   bool use_viscosity = true;
   bool mm = true;
   hydroLO::LagrangianLOOperator * hydro_LO = NULL;

   /*** Build limiter ***/
   H1_FECollection H1FEC_LO_t(1, dim);
   ParFiniteElementSpace H1FESpace_proj_LO(pmesh_lo, &H1FEC_LO_t);
   H1_FECollection H1FEC_HO_t(order_v, dim);
   ParFiniteElementSpace H1FESpace_proj_HO(pmesh, &H1FEC_HO_t); 
   IDPLimiter *idpl = NULL;
   if (idp_limit)
   {
      /*** Build Low-order solver */
      hydro_LO = new hydroLO::LagrangianLOOperator(
         dim, S_LO.Size(), LO_H1FESpace, LO_H1FESpace_L, LO_L2FESpace, 
         LO_L2VFESpace, LO_CRFESpace, rho0_coeff, rho_gf_LO, m, LO_ir, problem_class, 
         offset_LO, use_viscosity, 0, mm, cfl);
      
      hydro_LO->SetInitialMassesAndVolumes(S_LO);

      /* Set options for LO */
      hydro_LO->SetMVOption(-1);
      hydro_LO->SetMVLinOption(false);
      hydro_LO->SetFVOption(2);
      hydro_LO->SetProblem(problem);
      hydro_LO->SetDensityPP(true);
      hydro_LO->SetComputeMV(false);  

      GridTransfer *mv_gt = new InterpolationGridTransfer(H1FESpace, LO_H1FESpace);
      const Operator &P = mv_gt->ForwardOperator();
      idpl = new IDPLimiter(L2FESpace, H1FESpace_proj_LO, H1FESpace_proj_HO, *mHO_hpv, order_q);
   }
   

   hydrodynamics::LagrangianHydroOperator hydro(S.Size(),
                                                H1FESpace, L2FESpace, ess_tdofs,
                                                rho0_coeff, rho0_gf,
                                                idp_limit,
                                                problem_class,
                                                mat_gf, 
                                                hydro_LO, idpl,
                                                source, cfl,
                                                visc, vorticity, p_assembly,
                                                cg_tol, cg_max_iter, ftz_tol,
                                                order_q);

   socketstream vis_rho, vis_v, vis_e, vis_rho_limited;
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream vis_rho_LO, vis_v_LO, vis_ste_LO, vis_mc_LO;

   // This call is ok since it is just to initialize the grid function
   if (visualization || pview || visit) { hydro.ComputeDensity(rho_gf); }
   const double energy_init = hydro.InternalEnergy(e_gf) +
                              hydro.KineticEnergy(v_gf);
   hydro.GetRhoGFLim(rho_gf_limited);

   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      vis_rho_limited.precision(8);
      vis_rho_LO.precision(8);
      vis_v_LO.precision(8);
      vis_ste_LO.precision(8);
      vis_mc_LO.precision(8);
      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh + 45; // window offsets

      hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                    "Density", Wx, Wy, Ww, Wh);
      Wx += offx;
      if (idp_limit)
      {
         hydrodynamics::VisualizeField(vis_rho_limited, vishost, visport, rho_gf_limited,
                                       "Density limited", Wx, Wy, Ww, Wh);
      }
      Wx += offx;
      hydrodynamics::VisualizeField(vis_v, vishost, visport, v_gf,
                                    "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                    "Specific Internal Energy", Wx, Wy, Ww, Wh);
      
      if (idp_limit)
      {
         Wx = 0; Wy += offy;
         hydrodynamics::VisualizeField(vis_rho_LO, vishost, visport, rho_gf_LO,
                                       "LO Density", Wx, Wy, Ww, Wh);
         Wx += offx;
         hydrodynamics::VisualizeField(vis_v_LO, vishost, visport, v_gf_LO,
                                       "LO Velocity", Wx, Wy, Ww, Wh);
         Wx += offx;
         hydrodynamics::VisualizeField(vis_ste_LO, vishost, visport, ste_gf_LO,
                                       "LO Specific Total Energy", Wx, Wy, Ww, Wh);
                                       Wx += offx;
         hydrodynamics::VisualizeField(vis_mc_LO, vishost, visport, mc_gf_LO,
                                       "LO Mass loss", Wx, Wy, Ww, Wh);
      }
      
   }

   // Save data for VisIt visualization.
   VisItDataCollection visit_dc(basename, pmesh);
   VisItDataCollection visit_dc_LO(basename_LO, pmesh_lo);
   if (visit)
   {
      visit_dc.RegisterField("Density",  &rho_gf);
      visit_dc.RegisterField("Density limited", &rho_gf_limited);
      visit_dc.RegisterField("Velocity", &v_gf);
      visit_dc.RegisterField("Specific Internal Energy", &e_gf);
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();

      visit_dc_LO.RegisterField("Density", &rho_gf_LO);
      visit_dc_LO.RegisterField("Velocity", &v_gf_LO);
      visit_dc_LO.RegisterField("Specific Total Energy", &ste_gf_LO);
      visit_dc_LO.SetCycle(0);
      visit_dc_LO.SetTime(0.0);
      visit_dc_LO.Save();
   }

   ParaViewDataCollection paraview_dc(basename, pmesh);
   ParaViewDataCollection paraview_dc_LO(basename_LO, pmesh_lo);
   if (pview)
   {
      paraview_dc.SetDataFormat(VTKFormat::ASCII);
      paraview_dc.RegisterField("Density",  &rho_gf);
      paraview_dc.RegisterField("Density limited", &rho_gf_limited);
      paraview_dc.RegisterField("Velocity", &v_gf);
      paraview_dc.RegisterField("Specific Internal Energy", &e_gf);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.Save();

      paraview_dc_LO.SetDataFormat(VTKFormat::ASCII);
      paraview_dc_LO.RegisterField("Density", &rho_gf_LO);
      paraview_dc_LO.RegisterField("Velocity", &v_gf_LO);
      paraview_dc_LO.RegisterField("Specific Total Energy", &ste_gf_LO);
      paraview_dc_LO.SetCycle(0);
      paraview_dc_LO.SetTime(0.0);
      paraview_dc_LO.Save();
   }

   // Perform time-integration (looping over the time iterations, ti, with a
   // time-step dt). The object oper is of type LagrangianHydroOperator that
   // defines the Mult() method that used by the time integrators.
   ode_solver->Init(hydro);
   hydro.ResetTimeStepEstimate();

   double t = t_init, t_LO = t_init, dt = hydro.GetTimeStepEstimate(S), t_old;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S), S_old_LO(S_LO);
   long mem=0, mmax=0, msum=0;
   int checks = 0;
   //   const double internal_energy = hydro.InternalEnergy(e_gf);
   //   const double kinetic_energy = hydro.KineticEnergy(v_gf);
   //   if (mpi.Root())
   //   {
   //      cout << std::fixed;
   //      cout << "step " << std::setw(5) << 0
   //            << ",\tt = " << std::setw(5) << std::setprecision(4) << t
   //            << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
   //            << ",\t|IE| = " << std::setprecision(10) << std::scientific
   //            << internal_energy
   //            << ",\t|KE| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy
   //            << ",\t|E| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy+internal_energy;
   //      cout << std::fixed;
   //      if (mem_usage)
   //      {
   //         cout << ", mem: " << mmax << "/" << msum << " MB";
   //      }
   //      cout << endl;
   //   }
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      S_old_LO = S_LO;
      hydro.ResetTimeStepEstimate();

      /* Validate timestep and setup hydro for next step */
      if (idp_limit)
      {
         // hydro_LO->BuildDijMatrix(S_LO);
         // Check cfl restriction
         // hydro_LO->CalculateTimestep(S_LO);
         // double dt_LO = hydro_LO->GetTimestep();
         // if (dt > dt_LO)
         // {
         //    // cout << "dt: " << dt << ", lo dt: " << dt_LO << endl;
         //    dt = dt_LO;
         //    // MFEM_ABORT("Time step too large.\n");
         // }
         MFEM_WARNING("Check that lom satisfies cfl condition.\n");
      }

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(S, t, dt);
      hydro.GetSLO(S_LO);
      // Increment steps
      steps++;

      // Adaptive time step control.
      const double dt_est = hydro.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < std::numeric_limits<double>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         // if (idp_limit)
         // {
         //    S_LO = S_old_LO;
         // }
         hydro.ResetQuadratureData();
         if (Mpi::Root()) { cout << "Repeating step " << ti << endl; }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      // Ensure the sub-vectors x_gf, v_gf, and e_gf know the location of the
      // data in S. This operation simply updates the Memory validity flags of
      // the sub-vectors to match those of S.
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);

      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      pmesh->NewNodes(x_gf, false);

      // Do the same case for the low order approximation
      if (idp_limit)
      {
         x_gf_LO.SyncAliasMemory(S_LO);
         sv_gf_LO.SyncAliasMemory(S_LO);
         v_gf_LO.SyncAliasMemory(S_LO);
         ste_gf_LO.SyncAliasMemory(S_LO);
         pmesh_lo->NewNodes(x_gf_LO, false);
      }

      /* Compute cell masses */
      Vector el_mass(NE), el_vol(NE);
      hydro.ComputeDensity(rho_gf);
      if(idp_limit)
      {
         // cout << "setting rho_gf to limited vals\n";
         hydro.GetRhoGFLim(rho_gf_limited);
         rho_gf = rho_gf_limited;
      }
      MassesAndVolumesAtPosition(rho_gf, x_gf, el_mass, el_vol);
      double sum_current_masses = el_mass.Sum(), global_sum_om = mHO_hpv->GlobalVector()->Sum();
      double global_sum_cm;
      MPI_Allreduce(&sum_current_masses, &global_sum_cm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
      double _val = abs(global_sum_cm - global_sum_om) / global_sum_om;
      if (_val > 1.e-10)
      {
         cout << "|global_sum_cm - global_sum_om| = " << _val << endl;
         cout << setprecision(12) << "sum current masses: " << global_sum_cm << ", sum original: " << global_sum_om << endl;

         MFEM_ABORT("Not mass conservative.");
      }

      if (last_step || (ti % vis_steps) == 0)
      {
         // char ch;
         // fscanf(stdin, "%c", &ch);
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         if (mem_usage)
         {
            mem = GetMaxRssMB();
            MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
            MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
         }
         // const double internal_energy = hydro.InternalEnergy(e_gf);
         // const double kinetic_energy = hydro.KineticEnergy(v_gf);
         if (Mpi::Root())
         {
            const double sqrt_norm = sqrt(norm);

            cout << std::fixed;
            cout << "step " << std::setw(5) << ti
                 << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                 << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
                 << ",\t|e| = " << std::setprecision(10) << std::scientific
                 << sqrt_norm;
            //  << ",\t|IE| = " << std::setprecision(10) << std::scientific
            //  << internal_energy
            //   << ",\t|KE| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy
            //   << ",\t|E| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy+internal_energy;
            cout << std::fixed;
            if (mem_usage)
            {
               cout << ", mem: " << mmax << "/" << msum << " MB";
            }
            cout << endl;
         }

         // Fill grid function with mass information
         if (idp_limit)
         {
            double mass_loss;
            hydro_LO->ValidateMassConservation(S_LO, mc_gf_LO, mass_loss);
         }

         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || pview || visit || gfprint) { 
            hydro.ComputeDensity(rho_gf); 
            hydro.GetRhoGFLim(rho_gf_limited);
         }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10, offy = Wh + 45; // window offsets
            hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                          "Density", Wx, Wy, Ww, Wh);
            Wx += offx;
            if (idp_limit)
            {
               hydrodynamics::VisualizeField(vis_rho_limited, vishost, visport, rho_gf_limited,
                                             "Density limited", Wx, Wy, Ww, Wh);
            }
            Wx += offx;
            hydrodynamics::VisualizeField(vis_v, vishost, visport,
                                          v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                          "Specific Internal Energy",
                                          Wx, Wy, Ww,Wh);
            
            /* LO visualization */
            if (idp_limit)
            {
               Wx = 0; Wy += offy;
               hydrodynamics::VisualizeField(vis_rho_LO, vishost, visport, rho_gf_LO,
                                             "LO Density", Wx, Wy, Ww, Wh);
               Wx += offx;
               hydrodynamics::VisualizeField(vis_v_LO, vishost, visport, v_gf_LO,
                                             "LO Velocity", Wx, Wy, Ww, Wh);
               Wx += offx;
               hydrodynamics::VisualizeField(vis_ste_LO, vishost, visport, ste_gf_LO,
                                             "LO Specific Total Energy", Wx, Wy, Ww, Wh);
               Wx += offx;
               hydrodynamics::VisualizeField(vis_mc_LO, vishost, visport, mc_gf_LO,
                                             "LO Mass loss", Wx, Wy, Ww, Wh);
            }
            
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();

            visit_dc_LO.SetCycle(ti);
            visit_dc_LO.SetTime(t);
            visit_dc_LO.Save();
         }

         if (pview)
         {
            paraview_dc.SetCycle(ti);
            paraview_dc.SetTime(t);
            paraview_dc.Save();

            paraview_dc_LO.SetCycle(ti);
            paraview_dc_LO.SetTime(t);
            paraview_dc_LO.Save();
         }

         if (gfprint)
         {
            std::ostringstream mesh_name, rho_name, v_name, e_name;
            mesh_name << basename << "_" << ti << "_mesh";
            rho_name  << basename << "_" << ti << "_rho";
            v_name << basename << "_" << ti << "_v";
            e_name << basename << "_" << ti << "_e";

            std::ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            pmesh->PrintAsOne(mesh_ofs);
            mesh_ofs.close();

            std::ofstream rho_ofs(rho_name.str().c_str());
            rho_ofs.precision(8);
            rho_gf.SaveAsOne(rho_ofs);
            rho_ofs.close();

            std::ofstream v_ofs(v_name.str().c_str());
            v_ofs.precision(8);
            v_gf.SaveAsOne(v_ofs);
            v_ofs.close();

            std::ofstream e_ofs(e_name.str().c_str());
            e_ofs.precision(8);
            e_gf.SaveAsOne(e_ofs);
            e_ofs.close();

            if (idp_limit)
            {
               std::ostringstream rho_max_name, rho_min_name, rho_LO_name, rho_limited_name;
               rho_max_name << basename << "_" << ti << "_rho_max_idp";
               rho_min_name << basename << "_" << ti << "_rho_min_idp";
               rho_LO_name << basename << "_" << ti << "_rho_LO";
               rho_limited_name << basename << "_" << ti << "_rho_limited";

               ParGridFunction rho_gf_limited_max(&L2FESpace), rho_gf_limited_min(&L2FESpace);
               idpl->GetRhoMax(rho_gf_limited_max);
               idpl->GetRhoMin(rho_gf_limited_min);

               std::ofstream rho_max_ofs(rho_max_name.str().c_str());
               rho_max_ofs.precision(8);
               rho_gf_limited_max.SaveAsOne(rho_max_ofs);
               rho_max_ofs.close();

               std::ofstream rho_min_ofs(rho_min_name.str().c_str());
               rho_min_ofs.precision(8);
               rho_gf_limited_min.SaveAsOne(rho_min_ofs);
               rho_min_ofs.close();

               std::ofstream rho_LO_ofs(rho_LO_name.str().c_str());
               rho_LO_ofs.precision(8);
               rho_gf_LO.SaveAsOne(rho_LO_ofs);
               rho_LO_ofs.close();

               std::ofstream rho_limited_ofs(rho_limited_name.str().c_str());
               rho_limited_ofs.precision(8);
               rho_gf_limited.SaveAsOne(rho_limited_ofs);
               rho_limited_ofs.close();
            }
            
         }
      }

      // Problems checks
      if (check)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         const double e_norm = sqrt(norm);
         MFEM_VERIFY(rs_levels==0 && rp_levels==0, "check: rs, rp");
         MFEM_VERIFY(order_v==2, "check: order_v");
         MFEM_VERIFY(order_e==1, "check: order_e");
         MFEM_VERIFY(ode_solver_type==4, "check: ode_solver_type");
         MFEM_VERIFY(t_final == 0.6, "check: t_final");
         MFEM_VERIFY(cfl==0.5, "check: cfl");
         MFEM_VERIFY(strncmp(mesh_file, "default", 7) == 0, "check: mesh_file");
         MFEM_VERIFY(dim==2 || dim==3, "check: dimension");
         Checks(ti, e_norm, checks);
      }
   }
   MFEM_VERIFY(!check || checks == 2, "Check error!");

   switch (ode_solver_type)
   {
      case 2: steps *= 2; break;
      case 3: steps *= 3; break;
      case 4: steps *= 4; break;
      case 6: steps *= 6; break;
      case 7: steps *= 2;
   }

   hydro.PrintTimingData(Mpi::Root(), steps, fom);

   if (mem_usage)
   {
      mem = GetMaxRssMB();
      MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
      MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
   }

   const double energy_final = hydro.InternalEnergy(e_gf) +
                               hydro.KineticEnergy(v_gf);
   if (Mpi::Root())
   {
      cout << endl;
      cout << "Energy  diff: " << std::scientific << std::setprecision(2)
           << fabs(energy_init - energy_final) << endl;
      if (mem_usage)
      {
         cout << "Maximum memory resident set size: "
              << mmax << "/" << msum << " MB" << endl;
      }
   }

   // Print the error.
   // For problems 0 and 4 the exact velocity is constant in time.
   if (problem == 0 || problem == 4)
   {
      const double error_max = v_gf.ComputeMaxError(v_coeff),
                   error_l1  = v_gf.ComputeL1Error(v_coeff),
                   error_l2  = v_gf.ComputeL2Error(v_coeff);
      if (Mpi::Root())
      {
         cout << "L_inf  error: " << error_max << endl
              << "L_1    error: " << error_l1 << endl
              << "L_2    error: " << error_l2 << endl;
      }
   }

   /*
   For all test cases in which we have an exact solution,
   compute the error for convergence testing
   */
   if (problem_class->has_exact_solution())
   {
      if (idp_limit) {
         // No need to reacquire rho_gf_limited
         rho_gf = rho_gf_limited;
      } else {
         hydro.ComputeDensity(rho_gf);
      }

      ostringstream convergence_filename;
      convergence_filename << basename << "/convergence/np" << num_tasks;

      /* Coefficient to assist in computation of errors */
      ConstantCoefficient zero(0.0);
      
      /* Values to store numerators, to be computed on case by case basis since exact solutions vary */
      double rho_L1_error_n = 0., vel_L1_error_n = 0., ste_L1_error_n = 0.,
             rho_L2_error_n = 0., vel_L2_error_n = 0., ste_L2_error_n = 0.,
             rho_Max_error_n = 0., vel_Max_error_n = 0., ste_Max_error_n = 0.;

      /* Set coefficients to final time */
      FunctionCoefficient rho_coeff(rho0_static);
      rho_coeff.SetTime(t);
      v_coeff.SetTime(t);
      FunctionCoefficient sie_coeff(sie0_static);
      sie_coeff.SetTime(t);
      // FunctionCoefficient sv_coeff(sv0_static);
      // sv_coeff.SetTime(t);

      if (problem_class->get_indicator() == "Vdw1")
      {
         problem_class->update(x_gf, t);
      }

      // Compute errors
      ParGridFunction rho_ex_gf(&L2FESpace), vel_ex_gf(&H1FESpace), sie_ex_gf(&L2FESpace), sv_ex_gf(&L2FESpace);
      rho_ex_gf.ProjectCoefficient(rho_coeff);
      vel_ex_gf.ProjectCoefficient(v_coeff);

      // Similar to how the gridfunction is initialized, we need to interpolate in a non-positive 
      // basis to get the correct values at the dofs. Then we do an L2 projection to the positive
      // basis in which we actually compute. The goal is to get a high-order representation of the
      // exact solution.
      l2_e.ProjectCoefficient(sie_coeff);
      sie_ex_gf.ProjectGridFunction(l2_e);

      // In the case of the Noh Problem, project 0 on the boundary of approx and exact
      // if (problem_class->get_indicator() == "Noh")
      // {
      //    MFEM_ABORT("Issue with computing error for Noh problem.\n");
      //    cout << "[Noh] Projecting zero on the boundary cells.\n";
      //    ParGridFunction cell_bdr_flag_gf;
      //    hydro.GetCellBdrFlagGF(cell_bdr_flag_gf);

      //    for (int i = 0; i < pmesh->GetNE(); i++)
      //    {
      //       if (cell_bdr_flag_gf[i] != -1)
      //       {
      //          // We have a boundary cell
      //          rho_gf[i] = 0.;
      //          e_gf[i] = 0.;
      //          rho_ex_gf[i] = 0.;
      //          sie_ex_gf[i] = 0.;
      //          sv_ex_gf[i] = 0.;
      //          for (int j = 0; j < dim; j++)
      //          {
      //             int index = i + j*pmesh->GetNE();
      //             v_gf[index] = 0.;
      //             vel_ex_gf[index] = 0.;
      //          }
      //       }
      //    }
      // }

      /* Project 0 on all extrapolated cells, marked with attr = 99 */
      if (pmesh->attributes.Find(99) != -1)
      {
         if (Mpi::Root()) { cout << "Projecting zero on cells with attr 99\n"; }
         Vector _vec_zero(dim);
         _vec_zero = 0.;
         VectorConstantCoefficient _zero_vcc(_vec_zero);
         // onto approx
         rho_gf.ProjectCoefficient(_zero_vcc, 99);
         v_gf.ProjectCoefficient(_zero_vcc, 99);
         e_gf.ProjectCoefficient(_zero_vcc, 99);

         // onto exact
         rho_ex_gf.ProjectCoefficient(_zero_vcc, 99);
         vel_ex_gf.ProjectCoefficient(_zero_vcc, 99);
         sie_ex_gf.ProjectCoefficient(_zero_vcc, 99);
         sv_ex_gf.ProjectCoefficient(_zero_vcc, 99);
      }

      /* Exact grid function shows inf */
      // e_gf.Print(cout);
      // cout << "---\n";
      // sie_ex_gf.Print(cout);
      // sie_ex_gf[0] = e_gf[0];

      /* Compute relative errors */
      GridFunctionCoefficient rho_ex_coeff(&rho_ex_gf), vel_ex_coeff(&vel_ex_gf), ste_ex_coeff(&sie_ex_gf), sv_ex_coeff(&sv_ex_gf);
      
      // Velocity errors
      vel_L1_error_n = v_gf.ComputeL1Error(vel_ex_coeff) / vel_ex_gf.ComputeL1Error(zero);
      vel_L2_error_n = v_gf.ComputeL2Error(vel_ex_coeff) / vel_ex_gf.ComputeL2Error(zero);
      vel_Max_error_n = v_gf.ComputeMaxError(vel_ex_coeff) / vel_ex_gf.ComputeMaxError(zero);
      
      rho_L1_error_n = rho_gf.ComputeL1Error(rho_ex_coeff) / rho_ex_gf.ComputeL1Error(zero);
      rho_L2_error_n = rho_gf.ComputeL2Error(rho_ex_coeff) / rho_ex_gf.ComputeL2Error(zero);
      rho_Max_error_n = rho_gf.ComputeMaxError(rho_ex_coeff) / rho_ex_gf.ComputeMaxError(zero);

      ste_L1_error_n = e_gf.ComputeL1Error(ste_ex_coeff) / sie_ex_gf.ComputeL1Error(zero);
      ste_L2_error_n = e_gf.ComputeL2Error(ste_ex_coeff) / sie_ex_gf.ComputeL2Error(zero);
      ste_Max_error_n = e_gf.ComputeMaxError(ste_ex_coeff) / sie_ex_gf.ComputeMaxError(zero);

      /* Get composite errors values, will return 0 if exact solution is not known */
      const double L1_error = (rho_L1_error_n + vel_L1_error_n + ste_L1_error_n) / 3.;
      const double L2_error = (rho_L2_error_n + vel_L2_error_n + ste_L2_error_n) / 3.;
      const double Max_error = (rho_Max_error_n + vel_Max_error_n + ste_Max_error_n) / 3.;

      /* In either case, write convergence file. */
      if (Mpi::Root())
      {
         if (rs_levels != 0) {
            convergence_filename << "_s" << setfill('0') << setw(2) << rs_levels;
         }
         if (rp_levels != 0) {
            convergence_filename << "_p" << setfill('0') << setw(2) << rp_levels;
         }
         convergence_filename << "_refinement_"
                              << setfill('0') << setw(2)
                              << to_string(rp_levels + rs_levels)
                              << ".out";
         ofstream convergence_file(convergence_filename.str().c_str());
         convergence_file.precision(8);
         convergence_file << "Processor_Runtime " << "1." << "\n"
                           << "n_processes " << num_tasks << "\n"
                           << "n_refinements "
                           << to_string(rp_levels + rs_levels) << "\n"
                           << "n_Dofs " << glob_size_l2 << "\n"
                           << "h " << hmin << "\n"
                           // rho
                           << "rho_L1_Error " << rho_L1_error_n << "\n"
                           << "rho_L2_Error " << rho_L2_error_n << "\n"
                           << "rho_Linf_Error " << rho_Max_error_n << "\n"
                           // vel
                           << "vel_L1_Error " << vel_L1_error_n << "\n"
                           << "vel_L2_Error " << vel_L2_error_n << "\n"
                           << "vel_Linf_Error " << vel_Max_error_n << "\n"
                           // ste
                           << "ste_L1_Error " << ste_L1_error_n << "\n"
                           << "ste_L2_Error " << ste_L2_error_n << "\n"
                           << "ste_Linf_Error " << ste_Max_error_n << "\n"
                           // total
                           << "L1_Error " << L1_error << "\n"
                           << "L2_Error " << L2_error << "\n"
                           << "Linf_Error " << Max_error << "\n"
                           << "mass_loss " << 0. << "\n"
                           << "dt " << dt << "\n"
                           << "Endtime " << t << "\n";
                     
         convergence_file.close();
      }
   } // End error computation

   if (visualization)
   {
      vis_v.close();
      vis_e.close();
   }

   // Free the used memory.
   delete ode_solver;
   delete pmesh_lo;
   delete pmesh;
   delete idpl;
   delete problem_class;
   delete LO_CRFEC;
   delete m;
   // delete mv_gt;

   return 0;
}


double rho0(const Vector &x)
{
   switch (problem)
   {
      case 0: return 1.0;
      case 1: return 1.0;
      case 2: return (x(0) < 0.5) ? 1.0 : 0.1;
      case 3: return (dim == 2) ? (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0
                        : x(0) > 1.0 && ((x(1) < 1.5 && x(2) < 1.5) ||
                                         (x(1) > 1.5 && x(2) > 1.5)) ? 0.125 : 1.0;
      case 4: return 1.0;
      case 5:
      {
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.5313; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 0.8; }
         return 1.0;
      }
      case 6:
      {
         if (x(0) <  0.5 && x(1) >= 0.5) { return 2.0; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 3.0; }
         return 1.0;
      }
      case 7: return x(1) >= 0.0 ? 2.0 : 1.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

double gamma_func(const Vector &x)
{
   switch (problem)
   {
      case 0: return 5.0 / 3.0;
      case 1: return 1.4;
      case 2: return 1.4;
      case 3:
         if (dim == 1) { return (x(0) > 0.5) ? 1.4 : 1.5; }
         else { return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5; }
      case 4: return 5.0 / 3.0;
      case 5: return 1.4;
      case 6: return 1.4;
      case 7: return 5.0 / 3.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}


static double rad(double x, double y) { return sqrt(x*x + y*y); }


void v0(const Vector &x, Vector &v)
{
   const double atn = dim!=1 ? pow((x(0)*(1.0-x(0))*4*x(1)*(1.0-x(1))*4.0),
                                   0.4) : 0.0;
   switch (problem)
   {
      case 0:
         v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
         v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
         if (x.Size() == 3)
         {
            v(0) *= cos(M_PI*x(2));
            v(1) *= cos(M_PI*x(2));
            v(2) = 0.0;
         }
         break;
      case 1: v = 0.0; break;
      case 2: v = 0.0; break;
      case 3: v = 0.0; break;
      case 4:
      {
         v = 0.0;
         const double r = rad(x(0), x(1));
         if (r < 0.2)
         {
            v(0) =  5.0 * x(1);
            v(1) = -5.0 * x(0);
         }
         else if (r < 0.4)
         {
            v(0) =  2.0 * x(1) / r - 5.0 * x(1);
            v(1) = -2.0 * x(0) / r + 5.0 * x(0);
         }
         else { }
         break;
      }
      case 5:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=0.7276*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.7276*atn; return; }
         MFEM_ABORT("Error in problem 5!");
         return;
      }
      case 6:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=-0.5*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=-0.5*atn; return;}
         MFEM_ABORT("Error in problem 6!");
         return;
      }
      case 7:
      {
         v = 0.0;
         v(1) = 0.02 * exp(-2*M_PI*x(1)*x(1)) * cos(2*M_PI*x(0));
         break;
      }
      default: MFEM_ABORT("Bad number given for problem id!");
   }
}

double e0(const Vector &x)
{
   switch (problem)
   {
      case 0:
      {
         const double denom = 2.0 / 3.0;  // (5/3 - 1) * density.
         double val;
         if (x.Size() == 2)
         {
            val = 1.0 + (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) / 4.0;
         }
         else
         {
            val = 100.0 + ((cos(2*M_PI*x(2)) + 2) *
                           (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) - 2) / 16.0;
         }
         return val/denom;
      }
      case 1: return 0.0; // This case in initialized in main().
      case 2: return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma_func(x) - 1.0)
                        : 0.1 / rho0(x) / (gamma_func(x) - 1.0);
      case 3: return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma_func(x) - 1.0)
                        : 1.0 / rho0(x) / (gamma_func(x) - 1.0);
      case 4:
      {
         const double r = rad(x(0), x(1)), rsq = x(0) * x(0) + x(1) * x(1);
         const double gamma = 5.0 / 3.0;
         if (r < 0.2)
         {
            return (5.0 + 25.0 / 2.0 * rsq) / (gamma - 1.0);
         }
         else if (r < 0.4)
         {
            const double t1 = 9.0 - 4.0 * log(0.2) + 25.0 / 2.0 * rsq;
            const double t2 = 20.0 * r - 4.0 * log(r);
            return (t1 - t2) / (gamma - 1.0);
         }
         else { return (3.0 + 4.0 * log(2.0)) / (gamma - 1.0); }
      }
      case 5:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.4 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 5!");
         return 0.0;
      }
      case 6:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 6!");
         return 0.0;
      }
      case 7:
      {
         const double rho = rho0(x), gamma = gamma_func(x);
         return (6.0 - rho * x(1)) / (gamma - 1.0) / rho;
      }
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}


static void display_banner(std::ostream &os)
{
   os << endl
      << "       __                __                 " << endl
      << "      / /   ____  ____  / /_  ____  _____   " << endl
      << "     / /   / __ `/ __ `/ __ \\/ __ \\/ ___/ " << endl
      << "    / /___/ /_/ / /_/ / / / / /_/ (__  )    " << endl
      << "   /_____/\\__,_/\\__, /_/ /_/\\____/____/  " << endl
      << "               /____/                       " << endl << endl;
}

static long GetMaxRssMB()
{
   struct rusage usage;
   if (getrusage(RUSAGE_SELF, &usage)) { return -1; }
#ifndef __APPLE__
   const long unit = 1024; // kilo
#else
   const long unit = 1024*1024; // mega
#endif
   return usage.ru_maxrss/unit; // mega bytes
}

static void Checks(const int ti, const double nrm, int &chk)
{
   const double eps = 1.e-13;
   //printf("\033[33m%.15e\033[m\n",nrm);

   auto check = [&](int p, int i, const double res)
   {
      auto rerr = [](const double a, const double v, const double eps)
      {
         MFEM_VERIFY(fabs(a) > eps && fabs(v) > eps, "One value is near zero!");
         const double err_a = fabs((a-v)/a);
         const double err_v = fabs((a-v)/v);
         return fmax(err_a, err_v) < eps;
      };
      if (problem == p && ti == i)
      { chk++; MFEM_VERIFY(rerr(nrm, res, eps), "P"<<problem<<", #"<<i); }
   };

   const double it_norms[2][8][2][2] = // dim, problem, {it,norm}
   {
      {
         {{5, 6.546538624534384e+00}, { 27, 7.588576357792927e+00}},
         {{5, 3.508254945225794e+00}, { 15, 2.756444596823211e+00}},
         {{5, 1.020745795651244e+01}, { 59, 1.721590205901898e+01}},
         {{5, 8.000000000000000e+00}, { 16, 8.000000000000000e+00}},
         {{5, 3.446324942352448e+01}, { 18, 3.446844033767240e+01}},
         {{5, 1.030899557252528e+01}, { 36, 1.057362418574309e+01}},
         {{5, 8.039707010835693e+00}, { 36, 8.316970976817373e+00}},
         {{5, 1.514929259650760e+01}, { 25, 1.514931278155159e+01}},
      },
      {
         {{5, 1.198510951452527e+03}, {188, 1.199384410059154e+03}},
         {{5, 1.339163718592566e+01}, { 28, 7.521073677397994e+00}},
         {{5, 2.041491591302486e+01}, { 59, 3.443180411803796e+01}},
         {{5, 1.600000000000000e+01}, { 16, 1.600000000000000e+01}},
         {{5, 6.892649884704898e+01}, { 18, 6.893688067534482e+01}},
         {{5, 2.061984481890964e+01}, { 36, 2.114519664792607e+01}},
         {{5, 1.607988713996459e+01}, { 36, 1.662736010353023e+01}},
         {{5, 3.029858112572883e+01}, { 24, 3.029858832743707e+01}}
      }
   };

   for (int p=0; p<8; p++)
   {
      for (int i=0; i<2; i++)
      {
         const int it = it_norms[dim-2][p][i][0];
         const double norm = it_norms[dim-2][p][i][1];
         check(p, it, norm);
      }
   }
}

void MassesAndVolumesAtPosition(const ParGridFunction &u, const GridFunction &x,
                                Vector &el_mass, Vector &el_vol)
{
   // Only the order of the transformation matters.
   auto *Tr = x.FESpace()->GetMesh()->GetElementTransformation(0);
   const FiniteElement *fe = u.ParFESpace()->GetFE(0);
   const IntegrationRule &ir = MassIntegrator::GetRule(*fe, *fe, *Tr);
   const int nqp = ir.GetNPoints();
   const int NE = x.FESpace()->GetNE();

   GeometricFactors geom(x, ir, GeometricFactors::DETERMINANTS);
   auto qi_u = u.FESpace()->GetQuadratureInterpolator(ir);
   Vector u_qvals(nqp * NE);
   // As an L2 function, u has the correct EVector lexicographic ordering.
   qi_u->Values(u, u_qvals);

   for (int k = 0; k < NE; k++)
   {
      el_mass(k) = 0.0;
      el_vol(k)  = 0.0;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         el_mass(k) += ip.weight * geom.detJ(k*nqp + q) * u_qvals(k*nqp + q);
         el_vol(k)  += ip.weight * geom.detJ(k*nqp + q);
      }
   }
}
