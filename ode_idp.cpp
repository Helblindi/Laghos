#include "ode_idp.hpp"
#include "mfem/linalg/ode.hpp"

namespace mfem
{
namespace hydrodynamics
{

ODESolverIDP::ODESolverIDP() : ODESolver(), f_LO(NULL), f_HO(NULL) { 
   mem_type = Device::GetHostMemoryType(); 
}

void ODESolverIDP::Init(TimeDependentOperator &f_HO_, hydroLO::LagrangianLOOperator<2> &f_LO_)
{
   ODESolver::Init(f_HO_);
   this->f_HO = dynamic_cast<hydrodynamics::LagrangianHydroOperator*>(&f_HO_);
   MFEM_VERIFY(f_HO, "ODESolverIDP expect LagrangianHydroOperator.");
   this->f_LO = &f_LO_;
}

void ODESolverIDP::SetIDPOperator(hydroLO::LagrangianLOOperator<2> &f_LO_) 
{
   std::cout << "ODESolverIDP::SetIDPOperator" << std::endl;
   this->f_LO = &f_LO_; 
}

void ODESolverIDP::SetGridTransferOperator(const Operator &P_) 
{
   this->P = &P_;
   int height = P->Height(), width = P->Width();
   dx_gf_HO.SetSize(width, mem_type);
   dx_gf_LO.SetSize(height, mem_type);
}

/**
 * @brief Performs a single time step of the ODE solver with a limited high-order solution.
 * 
 * This method implements a low-order step followed by limiting the high-order solution
 * at each stage of the Runge-Kutta method. The following operations are performed:
 * 
 * 1. Set the time for the low-order operator.
 * 2. Update the low-order mesh velocity using the high-order mesh velocity.
 * 3. Enforce boundary conditions on the low-order mesh velocity.
 * 4. Step forward in time using the Forward Euler method for the low-order solution.
 * 5. Enforce boundary conditions on the low-order solution.
 * 6. Limit the high-order solution at the final time of the stage.
 * 
 * @param[in] x The current state vector.
 * @param[in] t The current time.
 * @param[in] dt The time step size.
 * @param[out] k The result vector after the step.
 */
void ODESolverIDP::StepLimited(const Vector &x, const double &t, const double &dt, Vector &k)
{
   /* 
   Low order must perform the following operations at each stage of the RungeKutta method
      1. Set the time of the low order operator
      2. Set the low order mesh velocity with the high order mesh velocity
      3. Enforce mesh velocity BCs
      4. Step forward in time using ForwardEuler
      5. Enforce boundary conditions on the low order solution
      6. Limit the high order solution at the final time of the stage
   */
   /* Compute IDP LO step */
   
}

void RK4SolverIDP::Init(TimeDependentOperator &f_HO_)
{
   MFEM_WARNING("Will need to set f_LO using ODESolverIDP::SetIDPOperator");
   this->f_LO = NULL;

   ODESolver::Init(f_HO_);
   this->f_HO = dynamic_cast<hydrodynamics::LagrangianHydroOperator*>(&f_HO_);
   int n = f->Width();
   y.SetSize(f->Width(), mem_type);
   k.SetSize(f->Width(), mem_type);
   z.SetSize(f->Width(), mem_type);
}

void RK4SolverIDP::Init(TimeDependentOperator &f_HO_, hydroLO::LagrangianLOOperator<2> &f_LO_)
{
   ODESolverIDP::Init(f_HO_, f_LO_);
   int n = f->Width();
   y.SetSize(n, mem_type);
   k.SetSize(n, mem_type);
   z.SetSize(n, mem_type);

   int nl = f_LO->Width();
   yl.SetSize(nl, mem_type);
   kl.SetSize(nl, mem_type);
   zl.SetSize(nl, mem_type);
}

void RK4SolverIDP::SetIDPOperator(hydroLO::LagrangianLOOperator<2> &f_LO_)
{
   ODESolverIDP::SetIDPOperator(f_LO_);
   int n = f_LO->Width();
   yl.SetSize(n, mem_type);
   kl.SetSize(n, mem_type);
   zl.SetSize(n, mem_type);
}

void RK4SolverIDP::Step(Vector &x, double &t, double &dt)
{
   //   0  |
   //  1/2 | 1/2
   //  1/2 |  0   1/2
   //   1  |  0    0    1
   // -----+-------------------
   //      | 1/6  1/3  1/3  1/6
   MFEM_ASSERT(f_LO != NULL, "f_LO not set. Use SetIDPOperator");

   // In each sub-step:
   // - Solve the HO stage of RK method
   // - Solve the LO stage of RK method
   // - Compute unlimited HO density
   // - Compute LO density
   // - Limit the HO density

   /**********  HO stage 1 **********/
   f_HO->SetTime(t);
   f_HO->Mult(x, k); // k1
   add(x, dt/2, k, y);
   add(x, dt/6, k, z);

   /**********  LO stage 1 **********/
   f_HO->GetMeshVelocity(dx_gf_HO);
   P->Mult(dx_gf_HO, dx_gf_LO);
   f_LO->SetMV(dx_gf_LO);

   f_LO->SetTime(t);
   f_LO->Mult(*S_LO, kl); // k1
   add(*S_LO, dt/2, kl, yl);
   add(*S_LO, dt/6, kl, zl);

   double pct_corrected, rel_mass_corrected;

   /* Limit HO Density */
   f_HO->Update(y);
   f_HO->ComputeDensity(*rho_gf_limited);
   f_LO->SetMassConservativeDensity(yl, pct_corrected, rel_mass_corrected);
   f_LO->ComputeDensity(yl, *rho_gf_LO);
   limiter->Limit(*rho_gf_LO, *rho_gf_limited);

   /**********  HO stage 2 **********/
   f_HO->SetTime(t + dt/2);
   f_HO->Mult(y, k); // k2
   add(x, dt/2, k, y);
   z.Add(dt/3, k);

   /**********  LO stage 2 **********/
   f_HO->GetMeshVelocity(dx_gf_HO);
   P->Mult(dx_gf_HO, dx_gf_LO);
   f_LO->SetMV(dx_gf_LO);

   f_LO->SetTime(t + dt/2);
   f_LO->Mult(yl, kl); // k2
   add(*S_LO, dt/2, kl, yl);
   zl.Add(dt/3, kl);

   /* Limit HO Density */
   f_HO->Update(y);
   f_HO->ComputeDensity(*rho_gf_limited);
   f_LO->SetMassConservativeDensity(yl, pct_corrected, rel_mass_corrected);
   f_LO->ComputeDensity(yl, *rho_gf_LO);
   limiter->Limit(*rho_gf_LO, *rho_gf_limited);

   /**********  HO stage 3 **********/
   f_HO->Mult(y, k); // k3
   add(x, dt, k, y);
   z.Add(dt/3, k);

   /**********  LO stage 3 **********/
   f_HO->GetMeshVelocity(dx_gf_HO);
   P->Mult(dx_gf_HO, dx_gf_LO);
   f_LO->SetMV(dx_gf_LO);

   f_LO->Mult(yl, kl); // k3
   add(*S_LO, dt, kl, yl);
   zl.Add(dt/3, kl);

   /* Limit HO Density */
   f_HO->Update(y);
   f_HO->ComputeDensity(*rho_gf_limited);
   f_LO->SetMassConservativeDensity(yl, pct_corrected, rel_mass_corrected);
   f_LO->ComputeDensity(yl, *rho_gf_LO);
   limiter->Limit(*rho_gf_LO, *rho_gf_limited);

   /**********  HO stage 4 **********/
   f_HO->SetTime(t + dt);
   f_HO->Mult(y, k); // k4
   add(z, dt/6, k, x);

   /**********  LO stage 4 **********/
   f_HO->GetMeshVelocity(dx_gf_HO);
   P->Mult(dx_gf_HO, dx_gf_LO);
   f_LO->SetMV(dx_gf_LO);

   f_LO->SetTime(t + dt);
   f_LO->Mult(yl, kl); // k4
   add(zl, dt/6, kl, *S_LO);
   /* No need to limit at this stage, just one final limit on the whole update*/

   /* Limit HO Density */
   f_HO->Update(x);
   f_HO->ComputeDensity(*rho_gf_limited);
   f_LO->SetMassConservativeDensity(*S_LO, pct_corrected, rel_mass_corrected);
   f_LO->ComputeDensity(*S_LO, *rho_gf_LO);
   limiter->Limit(*rho_gf_LO, *rho_gf_limited);

   t += dt;
}

void RK2SolverIDP::Init(TimeDependentOperator &f_HO_)
{
   MFEM_WARNING("Will need to set f_LO using ODESolverIDP::SetIDPOperator");
   this->f_LO = NULL;

   ODESolver::Init(f_HO_);
   this->f_HO = dynamic_cast<hydrodynamics::LagrangianHydroOperator*>(&f_HO_);
   int n = f->Width();
   dxdt.SetSize(n, mem_type);
   x1.SetSize(n, mem_type);
}

void RK2SolverIDP::Init(TimeDependentOperator &f_HO_, hydroLO::LagrangianLOOperator<2> &f_LO_)
{
   ODESolverIDP::Init(f_HO_, f_LO_);
   int n = f_HO->Width();
   dxdt.SetSize(n, mem_type);
   x1.SetSize(n, mem_type);

   int nl = f_LO->Width();
   dxdtl.SetSize(nl, mem_type);
   x1l.SetSize(nl, mem_type);
}

void RK2SolverIDP::SetIDPOperator(hydroLO::LagrangianLOOperator<2> &f_LO_)
{
   ODESolverIDP::SetIDPOperator(f_LO_);
   int n = f_LO->Width();
   dxdtl.SetSize(n, mem_type);
   x1l.SetSize(n, mem_type);
}

void RK2SolverIDP::Step(Vector &x, real_t &t, real_t &dt)
{
   //  0 |
   //  a |  a
   // ---+--------
   //    | 1-b  b      b = 1/(2a)

   MFEM_ASSERT(f_HO != NULL, "f_HO not set. Use SetIDPOperator");
   MFEM_ASSERT(f_LO != NULL, "f_LO not set. Use SetIDPOperator");
   MFEM_ASSERT(S_LO != NULL, "S_LO not set. Use SetLOStateVector");
   MFEM_ASSERT(P != NULL, "P not set. Use SetGridTransferOperator");
   MFEM_ASSERT(rho_gf_limited != NULL, "rho_gf_limited not set. Use SetRhoGFLimited");
   MFEM_ASSERT(limiter != NULL, "limiter not set. Use SetIDPLimiter");

   const real_t b = 0.5/a;

   /* HO stage 1 */
   f_HO->SetTime(t);
   f_HO->Mult(x, dxdt); // k1
   add(x, (1. - b)*dt, dxdt, x1);
   x.Add(a*dt, dxdt);

   /* LO stage 1 */
   f_HO->GetMeshVelocity(dx_gf_HO);
   P->Mult(dx_gf_HO, dx_gf_LO);
   f_LO->SetMV(dx_gf_LO);

   f_LO->SetTime(t);
   f_LO->Mult(*S_LO, dxdtl); // k1l
   add(*S_LO, (1. - b)*dt, dxdtl, x1l); 
   S_LO->Add(a*dt, dxdtl);

   double pct_corrected, rel_mass_corrected;

   /* Limiting */
   f_HO->Update(x);
   f_HO->ComputeDensity(*rho_gf_limited);
   f_LO->SetMassConservativeDensity(*S_LO, pct_corrected, rel_mass_corrected);
   f_LO->ComputeDensity(*S_LO, *rho_gf_LO);
   limiter->Limit(*rho_gf_LO, *rho_gf_limited);

   /* HO stage 2 */
   f_HO->SetTime(t + a*dt);
   f_HO->Mult(x, dxdt); // k2
   add(x1, b*dt, dxdt, x); // xn+1

   /* LO stage 2 */
   f_HO->GetMeshVelocity(dx_gf_HO);
   P->Mult(dx_gf_HO, dx_gf_LO);
   f_LO->SetMV(dx_gf_LO);
   f_LO->BuildDijMatrix(*S_LO); // Need to rebuild dij, no need to rebuild in first stage since it has been built already
   f_LO->SetTime(t + a*dt);
   f_LO->Mult(*S_LO, dxdtl);
   add(x1l, b*dt, dxdtl, *S_LO);

   /* Limiting */
   f_HO->Update(x);
   f_HO->ComputeDensity(*rho_gf_limited);
   f_LO->SetMassConservativeDensity(*S_LO, pct_corrected, rel_mass_corrected);
   f_LO->ComputeDensity(*S_LO, *rho_gf_LO);
   limiter->Limit(*rho_gf_LO, *rho_gf_limited);

   t += dt;
}

void ForwardEulerSolverIDP::Init(TimeDependentOperator &f_HO_)
{
   MFEM_WARNING("Will need to set f_LO using ODESolverIDP::SetIDPOperator");
   this->f_LO = NULL;

   ODESolver::Init(f_HO_);
   this->f_HO = dynamic_cast<hydrodynamics::LagrangianHydroOperator*>(&f_HO_);
   int n = f->Width();
   dxdt.SetSize(n, mem_type);
}

void ForwardEulerSolverIDP::Init(TimeDependentOperator &f_HO_, hydroLO::LagrangianLOOperator<2> &f_LO_)
{
   ODESolverIDP::Init(f_HO_, f_LO_);
   int n = f_HO->Width();
   dxdt.SetSize(n, mem_type);

   int nl = f_LO->Width();
   dxdtl.SetSize(nl, mem_type);
}

void ForwardEulerSolverIDP::SetIDPOperator(hydroLO::LagrangianLOOperator<2> &f_LO_)
{
   ODESolverIDP::SetIDPOperator(f_LO_);
   int n = f_LO->Width();
   dxdtl.SetSize(n, mem_type);
}

void ForwardEulerSolverIDP::Step(Vector &x, real_t &t, real_t &dt)
{
   MFEM_ASSERT(f_HO != NULL, "f_HO not set. Use SetIDPOperator");
   MFEM_ASSERT(f_LO != NULL, "f_LO not set. Use SetIDPOperator");
   MFEM_ASSERT(S_LO != NULL, "S_LO not set. Use SetLOStateVector");
   MFEM_ASSERT(P != NULL, "P not set. Use SetGridTransferOperator");
   MFEM_ASSERT(rho_gf_limited != NULL, "rho_gf_limited not set. Use SetRhoGFLimited");
   MFEM_ASSERT(limiter != NULL, "limiter not set. Use SetIDPLimiter");

   /* HO step */
   f_HO->SetTime(t);
   f_HO->Mult(x, dxdt);
   x.Add(dt, dxdt);

   /* LO step */
   f_HO->GetMeshVelocity(dx_gf_HO);
   P->Mult(dx_gf_HO, dx_gf_LO);
   f_LO->SetMV(dx_gf_LO);
   f_LO->SetTime(t);
   f_LO->Mult(*S_LO, dxdtl);
   S_LO->Add(dt, dxdtl);

   double pct_corrected, rel_mass_corrected;
   f_LO->SetMassConservativeDensity(*S_LO, pct_corrected, rel_mass_corrected);

   /* Limiting */
   f_HO->Update(x);
   f_HO->ComputeDensity(*rho_gf_limited);
   f_LO->ComputeDensity(*S_LO, *rho_gf_LO);
   limiter->Limit(*rho_gf_LO, *rho_gf_limited);

   t += dt;
}

} // ns hydrodynamics

} // ns mfem