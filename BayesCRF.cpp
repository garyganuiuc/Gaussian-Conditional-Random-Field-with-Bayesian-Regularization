#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>
#include <math.h>
#include <cstring>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

struct Param{
  Param(double v0_y_,double v1_y_,
        double v0_xy_,double v1_xy_,
        double tau_, double beta_,
        double sigma_, int ls_max_iters_,int maxiter_,bool quiet_):\
  v0_y(v0_y_), v1_y(v1_y_), v0_xy(v0_xy_), \
  v1_xy(v1_xy_), tau(tau_), beta(beta_), \
  sigma(sigma_),ls_max_iters(ls_max_iters_), maxiter(maxiter_), quiet(quiet_){}
  double v0_y, v1_y, v0_xy, v1_xy, tau; // tuning parameters in SSLasso prior
  double beta, sigma; // line search parameters (shrinkage parameter, tolerance coeff in Armijo-rule)
  int ls_max_iters, maxiter; //line search max iterations, algo max iterations
  bool quiet; // display algo status or not
};

struct Covariance{
  Covariance(Mat<double>S_x_,Mat<double>S_y_,Mat<double>S_xy_):\
  S_x(S_x_), S_y(S_y_), S_xy(S_xy_){}
  Mat<double> S_x, S_y, S_xy;
};





// Soft thresholding function
int sgn(double v) {
  return (v < 0) ? -1 : 1;
}

double SoftThreshold(double a,double b,double c,double lambda){
  return -c+sgn(c-b/a)*fmax(fabs(c-b/a)-lambda/a,0.0);
}

// Calculating the log-likehood value
double Likelihood(
    Mat<double>& Theta_y,
    Mat<double>& Theta_xy,
    Mat<double>& S_y,
    Mat<double>& S_xy,
    Mat<double>& S_x,
    int N) {
  return real(N/2.0*(-log_det(Theta_y)+trace(S_y*Theta_y)+\
    2.0*trace(Theta_xy.t()*S_xy)+\
    trace(Theta_y.i()*Theta_xy.t()*S_x*Theta_xy)));
}


// Lasso penalty Function  (Used in M-step)
double Lasso_offdiag(Mat<double> &p,Mat<double> &Theta,
               double v0,double v1){
  return accu((p/v1+(1-p)/v0)%abs(Theta))-accu(diagmat((p/v1+(1-p)/v0)%abs(Theta)));
  //accu(log((1-p_y)%exp(-abs(Theta_y)/v0_y)+p_y%exp(-abs(Theta_y)/v1_y)));
}

double Lasso_diag(Mat<double> &Theta_y,double tau){
  return accu(tau*Theta_y.diag());
  //accu(log((1-p_y)%exp(-abs(Theta_y)/v0_y)+p_y%exp(-abs(Theta_y)/v1_y)));
}

double Lasso_subgrad_offdiag(Mat<double> &p,Mat<double> &Theta,
                     double v0,double v1){
  return accu((p/v1+(1-p)/v0)%sign(Theta))-accu(diagmat((p/v1+(1-p)/v0)%sign(Theta)));
  //accu(log((1-p_y)%exp(-abs(Theta_y)/v0_y)+p_y%exp(-abs(Theta_y)/v1_y)));
}

double Lasso_subgrad_diag(Mat<double> &Theta_y,double tau){
  return accu(tau*sign(Theta_y.diag()));
  //accu(log((1-p_y)%exp(-abs(Theta_y)/v0_y)+p_y%exp(-abs(Theta_y)/v1_y)));
}



//E-step Function
Mat<double> E_step(
  double v0,
  double v1,
  double eta,
  Mat<double> &Theta){
  Mat<double> logit=std::log(v0/v1)+std::log(eta/(1-eta))-abs(Theta)/v1+abs(Theta)/v0;
  return 1.0/(1.0+exp(-logit));
}


//Coordinate Descent/M-step Function
void Coordinate_Descent(
    int N,
    Param & params,
    Covariance & sample_cov,
    Mat<double> & p_y,
    Mat<double> & p_xy,
    Mat<double>& Theta_y,
    Mat<double>& Theta_xy,
    Mat<double> & D_y,
    Mat<double> & D_xy){
            
            Mat<double> S_x = sample_cov.S_x;
            Mat<double> S_xy = sample_cov.S_xy;
            Mat<double> S_y = sample_cov.S_y;  
            int q = S_x.n_rows;
            int p = S_y.n_rows;
            

              
            Mat<double> Sigma_y = Theta_y.i();
            Mat<double> U = mat(p, p, fill::zeros);
            D_y = mat(p, p, fill::zeros);
            D_xy = mat(q, p, fill::zeros);
            Mat<double> V = mat(q, p, fill::zeros);
            
            //M-step
 
           //update Theta_y off-diag
           //Mat<double> A = Theta_y.i();
           Mat<double> Phi = Sigma_y*Theta_xy.t()*S_x*Theta_xy*Sigma_y;
           Mat<double> B = Sigma_y + 2*Phi;
           Mat<double> C=Sigma_y*Theta_xy.t()*S_x;
           for(int i = 0; i < p - 1; i++){
             for(int j = (i+1); j < p; j++){
               double a=N*(pow(Sigma_y(i,j),2)+Sigma_y(i,i)*Sigma_y(j,j)+  \
                           Sigma_y(i,i)*Phi(j,j) + 2*Sigma_y(i,j)*Phi(i,j)+\
                           Sigma_y(j,j)*Phi(i,i));
               double b=N*(-Sigma_y(i,j)+S_y(i,j)-Phi(i,j)+sum(Sigma_y.row(i)*U.col(j)+Phi.row(i)*U.col(j)+\
                           Phi.row(j)*U.col(i)-C.row(i)*V.col(j)-C.row(j)*V.col(i)));
               double c=Theta_y(i,j)+U(i,j);
               double lambda=(1-p_y(i,j))/params.v0_y+p_y(i,j)/params.v1_y;
               double delta_y=SoftThreshold(a,b,c,lambda);
               D_y(i,j) = D_y(i,j)+delta_y;
               D_y(j,i) = D_y(j,i)+delta_y;
               U.row(i) = U.row(i)+Sigma_y.row(j)*delta_y;
               U.row(j) = U.row(j)+Sigma_y.row(i)*delta_y;
             }
           }
           
           //update Theta_y diag
           for(int i=0; i < p; i++){
             Mat<double> A = Phi*U;
             double a=2*N/4*Sigma_y(i,i)*B(i,i);
             double b=N/2*(-Sigma_y(i,i)+S_y(i,i)-Phi(i,i)+ sum(Sigma_y.row(i)*U.col(i) - 2*C.row(i)*V.col(i))+\
                           A(i,i)*2);
             double c=Theta_y(i,i);
             double delta_y=SoftThreshold(a,b,c,params.tau);
             U.row(i)=U.row(i)+Sigma_y.row(i)*delta_y;
             D_y(i,i)=D_y(i,i)+delta_y;
           }
           
            //Update Theta_xy
            for(int i = 0; i < q; i++){
              for(int j = 0; j < p; j++){
                Mat<double> A = Theta_xy*Sigma_y;
                Mat<double> B = S_x*Theta_xy*Sigma_y;
                double a = N*Sigma_y(j,j)*S_x(i,i);
                double b = N *(S_xy(i,j)+sum(S_x.row(i)*A.col(j)+S_x.row(i)*V.col(j)-B.row(i)*U.col(j)));
                double c = Theta_xy(i,j)+V(i,j);
                double lambda = (1-p_xy(i,j))/params.v0_xy+p_xy(i,j)/params.v1_xy;
                double delta_xy = SoftThreshold(a,b,c,lambda);
                D_xy(i,j) = D_xy(i,j)+delta_xy;
                V.row(i) = V.row(i)+Sigma_y.row(j)*delta_xy;
              }
            }
    return;
                
                
}

// Line_search/Back_tracking for the step size of the M-step (Check the Armijo Rule)
bool Line_search(
    int N,
    Param & params,
    Covariance & sample_cov,
    Mat<double> & p_y,
    Mat<double> & p_xy,   
    Mat<double> & Theta_y,
    Mat<double> & Theta_xy,
    Mat<double> & D_y, 
    Mat<double> & D_xy
){
  double alpha = 1.0;
  double Likeli_old = Likelihood(Theta_y, Theta_xy, sample_cov.S_y, sample_cov.S_xy, sample_cov.S_x, N);
  //double l1_norm = Lasso_offdiag(p_y,Theta_y, params.v0_y, params.v1_y) +\
  //              Lasso_offdiag(p_xy,Theta_xy, params.v0_xy, params.v1_xy) +\
  //              Lasso_diag(Theta_y, params.tau);
  
  //Mat<double> tmp = Theta_y+D_y;
  //double l1_diff = Lasso_offdiag(p_y,tmp, params.v0_y, params.v1_y) + Lasso_diag(tmp, params.tau);
  //tmp = Theta_xy+D_xy;
  //l1_diff += Lasso_offdiag(p_xy,tmp, params.v0_xy, params.v1_xy) - l1_norm;
  //l1_diff = fabs(l1_diff);
  
  double l1_diff = Lasso_subgrad_offdiag(p_y, Theta_y, params.v0_y, params.v1_y) +\
            Lasso_subgrad_offdiag(p_xy,Theta_xy, params.v0_xy, params.v1_xy) +\
            Lasso_subgrad_diag(Theta_y, params.tau);
  
  Mat<double> A = -Theta_y.i()+sample_cov.S_y-Theta_y.i()*Theta_xy.t()*sample_cov.S_x*Theta_xy*Theta_y.i();

  double delta_t=trace(A.t()*D_y) + 2*trace((sample_cov.S_xy+sample_cov.S_x*Theta_xy*Theta_y.i()).t()*D_xy);
  delta_t = fabs(delta_t);
  
  // Contine from here
  Mat<double> Theta_y_alpha, Theta_xy_alpha;
  int i;
  for(i = 0; i < params.ls_max_iters; i++){
    Theta_y_alpha = Theta_y + alpha*D_y;
    mat decomp;
    if(chol(decomp,Theta_y_alpha) == false){
      if(params.quiet == false)
        Rprintf("Warning: line search %d, alpha=%f\tnot PD\n", i, alpha);
      alpha *= params.beta;
      continue;
    }
    
    Theta_xy_alpha = Theta_xy + alpha*D_xy;


    double Likeli_new = Likelihood(Theta_y_alpha, Theta_xy_alpha, sample_cov.S_y, sample_cov.S_xy, sample_cov.S_x, N);

    if(Likeli_new < Likeli_old + alpha*params.sigma*(delta_t + l1_diff))
      break;
    alpha *= params.beta;
  }
  if(i >= params.ls_max_iters){
    //if(params.quiet == false)
      Rprintf("Failed to improve objective\n");
    return false;
  }
  Theta_xy = Theta_xy_alpha;
  Theta_y = Theta_y_alpha;
  return true;
}
    



// [[Rcpp::export]] 
//Main Function
field<mat> BayesCRF(int N,
              double v0_xy,
              double v1_xy,
              double v0_y,
              double v1_y,
              double tau,
              Mat<double> & S_x,
              Mat<double> & S_y,
              Mat<double> & S_xy,
              double eta_xy=0.5,
              double eta_y=0.5,
              double sigma=0.05,
              double beta=0.5,
              int maxiter=1000,
              int ls_max_iters = 50,
              int M_max_iters = 100,
              bool quiet = false){
//Input
//sigma is a constant between 0 and 0.5, default 0.3
//tuning paramters for Theta_xy: v0_xy,v1_xy,eta_xy
//tuning paramters for Theta_y: v0_y,v1_y,eta_y,tau
//N how many observations

  /// Check if inputs are valid
  mat dec;
  if(chol(dec,S_y) == false)
    //throw std::invalid_argument("Error: Sample Covariance for y is not PD");
    Rprintf("Error: Sample Covariance for y is not PD");
    
  if(S_x.n_rows != S_xy.n_rows || S_xy.n_cols != S_y.n_rows || S_x.n_cols != S_x.n_rows || S_y.n_cols != S_y.n_rows){
    //REprintf("Error: Dimensions for Sample Covariances Don't Match");
    throw std::invalid_argument("Error: Dimensions for Sample Covariances Don't Match");
  }
  
  


  Covariance sample_cov(S_x,S_y,S_xy);
  Param params(v0_y,v1_y,v0_xy,v1_xy,tau,beta,sigma, ls_max_iters,maxiter,quiet);
  Mat<double> p_xy, p_y;
  Mat<double> Theta_y_old2, Theta_xy_old2;
  //int q = S_x.n_rows;
  //int p = S_y.n_rows;
  Mat<double> Theta_y= eye<mat>(size(S_y));
  Mat<double> Theta_xy = mat(size(S_xy), fill::zeros);
  Mat<double> Theta_xy_old = mat(size(Theta_xy));
  Theta_xy_old.fill(datum::inf);
  Mat<double> Theta_y_old = mat(size(Theta_y));
  Theta_y_old.fill(datum::inf);
  int iter;
  
  
  for(iter = 0; iter<= params.maxiter; iter++){

    //break condition
    if(abs((Theta_y-Theta_y_old).max())<0.001 && abs((Theta_xy-Theta_xy_old).max())<0.001){
      //if(params.quiet == false)
        Rprintf("Algorithm Converged\n");
      break;
    }
    
    Theta_y_old = Theta_y;
    Theta_xy_old = Theta_xy;
    
    //E-step
    p_xy = E_step(params.v0_xy,params.v1_xy,eta_xy,Theta_xy);
    p_y = E_step(params.v0_y,params.v1_y,eta_y,Theta_y);

    Theta_y_old2 = mat(size(Theta_y));
    Theta_y_old2.fill(datum::inf);
    Theta_xy_old2 = mat(size(Theta_xy));
    Theta_xy_old2.fill(datum::inf);
    
    //M-step
    for(int j=0; j<= M_max_iters; j++){
      
        if(abs((Theta_y-Theta_y_old2).max())<0.001 && abs((Theta_xy-Theta_xy_old2).max())<0.001){
          //if(params.quiet == false)
            Rprintf("M-step Converged\n");
          break;
        }
        Theta_y_old2 = Theta_y;
        Theta_xy_old2 = Theta_xy;
  
        Mat<double> D_y, D_xy;
        Coordinate_Descent(N, params, sample_cov, p_y, p_xy,\
                           Theta_y, Theta_xy, D_y, D_xy);
        
        // Track the line search status, if state == false, break the current M-step 
        bool state = Line_search(N, params, sample_cov, p_y, p_xy,\
                                 Theta_y, Theta_xy, D_y, D_xy);
        if(state  == false)
          break;
        
      
    }

    
    
  }
  if(iter > params.maxiter){
    //if(params.quiet == false)
      Rprintf("Warning: Algorithm Didn't Converge\n"); 
  }
  
  field<arma::mat> returnlist(4);
  returnlist(0) = Theta_y;
  returnlist(1) = Theta_xy;
  returnlist(2) = p_y;
  returnlist(3) = p_xy;
  
  return returnlist;

}
 
