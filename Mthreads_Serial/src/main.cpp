#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "GeneralTools.h"
#include <boost/thread.hpp>
#include <boost/ref.hpp>

#include "occa.hpp"

typedef double hp;

void foo()
{
  occa::device Device;
  Device.setup("Serial",0,0);
  const size_t N_simple = 10;
  const size_t size_simple = N_simple*sizeof(hp);
 
  occa::kernelInfo args;
  args.addCompilerFlag(GeneralTools::get_occa_local_include());
  // build kernels
    
  occa::kernel simple = Device.buildKernelFromSource(GeneralTools::get_occaKernels_path("simple.occa"), "simple", args);

  // size of array
  int N = 99999;

  // set thread array for Jacobi iteration
  int T = 1024;
  int dims = 1;
  occa::dim inner(T);
  occa::dim outer((N+T-1)/T);
  simple.setWorkingDims(dims, inner, outer);


  size_t sz = N*sizeof(float);

  // allocate array on HOST
  float *h_x = (float*) malloc(sz);
  for(int n=0;n<N;++n)
    h_x[n] = 123;
  
  // allocate array on DEVICE (copy from HOST)
  occa::memory c_x = Device.malloc(sz, h_x);

  // queue kernel
  simple(N, c_x);
 

  // copy result to HOST
  c_x.copyTo(h_x);
  
  /* print out results */
  for(int n=0;n<N;++n)
    printf("h_x[%d] = %g\n", n, h_x[n]);

  c_x.free();

};

void foo_add()
{
  const bool verbose = true;
  occa::device Device;
  Device.setup("Serial",0,0);
  // arg
  int nn;
  cout << "input N: ";
  cin>>nn;
  const int N = nn;  
  const int threadsPerBlock = 16;
  const int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
 
  occa::kernelInfo args;
  args.addDefine("N",N);
  args.addDefine("bDIM",threadsPerBlock);
  args.addDefine("gDIM",blocksPerGrid);
  args.addCompilerFlag(GeneralTools::get_occa_local_include());

  occa::kernel Scan_add1_p1 = Device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Scan_serial.occa"), "Scan_add_Blelloch_p1", args);
  occa::kernel Scan_add1_p2 = Device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Scan_serial.occa"), "Scan_add_Blelloch_p2", args);

  int dims = 1;
  occa::dim inner(threadsPerBlock);
  occa::dim outer(blocksPerGrid);
  
  Scan_add1_p1.setWorkingDims(dims, inner, outer);
  Scan_add1_p2.setWorkingDims(dims, inner, outer);
  // allocate array on HOST  
  size_t sz = N*sizeof(int);
  
  int *a_cpu = (int*) malloc(sz);  
  int *b_cpu1 = (int*) malloc(sz); 
  int *b_ref = (int*) malloc(sz);

  for(int i=0;i<N;++i)
  {
    a_cpu[i] = i+1;
  }
  if(verbose)
  {
    cout << "a: "; 
    for(int i =0; i< N; ++i) cout << a_cpu[i] <<", ";
    cout << endl;
  }
  //--------------Ref-------------------------
 
  //--b2 ref
  
  b_ref[0]=1;
  for (int i =1; i < N; ++i) 
  {
    b_ref[i]= b_ref[i-1]+a_cpu[i];
  }

  if(verbose)
  {
    cout <<"b_ref: ";
    for(int i =0; i< N; ++i) cout << b_ref[i] <<", ";
    cout << endl;
    cout << "---------------Calculation----------------" << endl;
  }

  // allocate array on DEVICE (copy from HOST)
  occa::memory a_gpu = Device.malloc(sz,a_cpu);
  occa::memory b_gpu1 = Device.malloc(sz);
  occa::memory b_tp1_gpu = Device.malloc(blocksPerGrid*sizeof(int));

  Scan_add1_p1(a_gpu, b_gpu1, b_tp1_gpu);
  
  b_gpu1.copyTo(b_cpu1);
  if (verbose)
  {
    cout <<"b_1 p1: ";
    for(int i =0; i< N; ++i) cout << b_cpu1[i] <<", ";
    cout << endl;
  }
  
  Scan_add1_p2(b_tp1_gpu, b_gpu1);
  
  b_gpu1.copyTo(b_cpu1);
  if (verbose)
  {
    cout <<"b_1 p2: ";
    for(int i =0; i< N; ++i) cout << b_cpu1[i] <<", ";
    cout << endl;
  }

  bool res1(true);
  
  for (int i=0; i < N; ++i)
  {
    if (b_ref[i] != b_cpu1[i])
    {
      res1 = false;
      printf("id= %d not equal; b_ref=%d, b1=%d\n",i, b_ref[i], b_cpu1[i]);
      break;
    }
  }

  if (res1)
    {
      cout << "test pass :)" << endl;
    }
  else 
  {
    cout << "test failed :(" << endl;
  }

  Device.free();
  
};


int main(int argc, char **argv){

  /* hard code platform and device number */
  foo();

  //foo_add();
 
  return 0;
};
