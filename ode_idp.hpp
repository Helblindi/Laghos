#ifndef ODE_SOLVER_IDP
#define ODE_SOLVER_IDP

#include "mfem.hpp"
#include "laghos_solver.hpp"
#include "laglos_solver.hpp"
#include "limiter.h"
#include "mfem/fem/pgridfunc.hpp"
#include "mfem/linalg/vector.hpp"

namespace mfem
{
namespace hydrodynamics
{

class ODESolverIDP : public ODESolver
{
protected:
   hydrodynamics::LagrangianHydroOperator *f_HO;
   hydroLO::LagrangianLOOperator<2> *f_LO; // Could be LagrangianLOOperator, would simplify things
   ParGridFunction dx_gf_HO, dx_gf_LO;
   ParGridFunction *rho_gf_LO, *rho_gf_limited;
   const Operator *P;
   Vector *S_LO;
   IDPLimiter *limiter;
public:
   ODESolverIDP();

   /* Setters */
   virtual void SetIDPOperator(hydroLO::LagrangianLOOperator<2> &f_LO_);
   void SetIDPLimiter(IDPLimiter &limiter_) { this->limiter = &limiter_; }
   void SetLOStateVector(Vector &S_LO_) { this->S_LO = &S_LO_; }
   void SetGridTransferOperator(const Operator &P_);
   void SetRhoGFLimited(ParGridFunction &rho_gf_limited_) { this->rho_gf_limited = &rho_gf_limited_; }
   void SetRhoGFLO(ParGridFunction &rho_gf_LO_) { this->rho_gf_LO = &rho_gf_LO_; }

   /* Misc */
   virtual void Init(TimeDependentOperator &f_HO_, hydroLO::LagrangianLOOperator<2> &f_LO_);
   void StepLimited(const Vector &x, const double &t, const double &dt, Vector &k);
};

class RK4SolverIDP : public ODESolverIDP
{
private:
   Vector y, k, z;
   Vector yl, kl, zl;

public:
   void Init(TimeDependentOperator &f_) override;

   void Init(TimeDependentOperator &f_HO_, hydroLO::LagrangianLOOperator<2> &f_LO_) override;

   void SetIDPOperator(hydroLO::LagrangianLOOperator<2> &f_LO_) override;

   void Step(Vector &x, double &t, double &dt) override;
};

/** A family of explicit second-order RK2 methods. Some choices for the
parameter 'a' are:
a = 1/2 - the midpoint method
a =  1  - Heun's method
a = 2/3 - default, has minimal truncation error. */
class RK2SolverIDP : public ODESolverIDP
{
private:
   real_t a;
   Vector dxdt, x1;
   Vector dxdtl, x1l;

public:
   RK2SolverIDP(const real_t a_ = 2./3.) : a(a_) { }

   void Init(TimeDependentOperator &f_HO_) override;

   void Init(TimeDependentOperator &f_HO_, hydroLO::LagrangianLOOperator<2> &f_LO_) override;

   void SetIDPOperator(hydroLO::LagrangianLOOperator<2> &f_LO_) override;

   void Step(Vector &x, real_t &t, real_t &dt) override;
};

class ForwardEulerSolverIDP : public ODESolverIDP
{
private:
   Vector dxdt, dxdtl;

public:
   ForwardEulerSolverIDP() { }

   void Init(TimeDependentOperator &f_HO_) override;

   void Init(TimeDependentOperator &f_HO_, hydroLO::LagrangianLOOperator<2> &f_LO_) override;

   void SetIDPOperator(hydroLO::LagrangianLOOperator<2> &f_LO_) override;

   void Step(Vector &x, double &t, double &dt) override;
};

} // ns hydrodynamics

} // ns mfem

#endif // ODE_SOLVER_IDP