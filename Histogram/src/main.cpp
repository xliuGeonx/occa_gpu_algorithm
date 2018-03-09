#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "GeneralTools.h"
#include <boost/progress.hpp>

#include "occa.hpp"

int main(int argc, char **argv){

  const bool verbose = true;
  /* hard code platform and device number */
  int plat = 0;
  int dev = 0;

  occa::device device;
  device.setup("CUDA", plat, dev);

  // build jacobi kernel from source file
  const char *functionName = "";

  // arg
  int nn;
  cout << "input N: ";
  cin>>nn;
  const int N = nn;
  const int threadsPerBlock = 1024;
  const int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
  const int bin_count = 10;

  occa::kernelInfo args;
  args.addDefine("N",N);
  args.addDefine("bin_count",bin_count);
  args.addDefine("bDIM",threadsPerBlock);
  args.addDefine("gDIM",blocksPerGrid);
  args.addCompilerFlag(GeneralTools::get_occa_local_include());

  // build Jacobi kernel
  occa::kernel Simple_histogram = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Histogram.occa"), "Simple_histogram", args);


  // set thread array for Jacobi iteration
  const int dims = 1;
  occa::dim inner(threadsPerBlock);
  occa::dim outer(blocksPerGrid);
  Simple_histogram.setWorkingDims(dims, inner, outer);


  size_t sz = N*sizeof(int);
  int *a_cpu = (int*) malloc(sz);

  int *b_cpu1 = (int*) malloc(sz);
  int *b_ref = (int*) malloc(sz);

  boost::progress_timer t_ref;
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
  cout <<"Scan_add_ref run time: " << t_ref.elapsed() << endl;

  if(verbose)
  {
    cout <<"b_ref: ";
    for(int i =0; i< N; ++i) cout << b_ref[i] <<", ";
    cout << endl;
    cout << "---------------Calculation----------------" << endl;
  }
 //------------------------------------------------
  occa::memory a_gpu = device.malloc(sz, a_cpu);
  occa::memory b_gpu1 = device.malloc(sz,b_cpu1);


  // queue kernel
  boost::progress_timer t_reduce1;
  Simple_histogram(a_gpu, b_gpu1);
  b_gpu1.copyTo(b_cpu1);
  if (verbose)
  {
    cout <<"b_1 p1: ";
    for(int i =0; i< N; ++i) cout << b_cpu1[i] <<", ";
    cout << endl;
  }
  cout <<"Scan_add1 run time: " << t_reduce1.elapsed() << endl; 
  
 
  // copy result to HOST
  Simple_histogram.free();


 //--------------Verification-----------------
  bool res1;

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
      return 1;
  }
  else 
  {
    cout << "test failed :(" << endl;
    return 0;
  }
}
