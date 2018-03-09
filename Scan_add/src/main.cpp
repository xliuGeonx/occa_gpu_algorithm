//---------------CUDA algorithm[Scan Inclusive] executated by OCCA2---------------------
//--------------------------by Xin at GeonX-----------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "GeneralTools.h"
#include <boost/progress.hpp>
#include <math.h>

#include "occa.hpp"

using namespace std;

int main(int argc, char **argv){

  const bool verbose = false;
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
    occa::kernelInfo args;
    args.addDefine("N",N);
    
    args.addDefine("bDIM",threadsPerBlock);
    args.addDefine("gDIM",blocksPerGrid);
    args.addCompilerFlag(GeneralTools::get_occa_local_include());
  // build kernels
    
  occa::kernel Scan_add1_p1 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Scan.occa"), "Scan_add_Blelloch_p1", args);
  occa::kernel Scan_add1_p2 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Scan.occa"), "Scan_add_Blelloch_p2", args);

  occa::kernel Scan_add2_p1 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Scan.occa"), "Scan_add_HS_p1", args);
  occa::kernel Scan_add2_p2 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Scan.occa"), "Scan_add_HS_p2", args);
  occa::kernel Scan_add_HS = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Scan.occa"), "Scan_add_HS", args);
  // set thread array 
  int dims = 1;
  occa::dim inner(threadsPerBlock);
  occa::dim outer(blocksPerGrid);
  
  Scan_add1_p1.setWorkingDims(dims, inner, outer);
  Scan_add1_p2.setWorkingDims(dims, inner, outer);
  Scan_add2_p1.setWorkingDims(dims, inner, outer);
  Scan_add2_p2.setWorkingDims(dims, inner, outer);
  Scan_add_HS.setWorkingDims(dims, inner, outer);
  // allocate array on HOST  
  size_t sz = N*sizeof(int);
  
  int *a_cpu = (int*) malloc(sz);
  
  int *b_cpu1 = (int*) malloc(sz);
  int *b_cpu2 = (int*) malloc(sz);
  int *b_cpuHS = (int*) malloc(sz);
  int *b_ref = (int*) malloc(sz);
  
  //---------------------------------

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
  
  
  // allocate array on DEVICE (copy from HOST)
  occa::memory a_gpu = device.malloc(sz,a_cpu);
  occa::memory b_gpu1 = device.malloc(sz);
  occa::memory b_gpu2 = device.malloc(sz);
  occa::memory b_gpuHS = device.malloc(sz);
  occa::memory b_tp1_gpu = device.malloc(blocksPerGrid*sizeof(int));
  occa::memory b_tp2_gpu = device.malloc(blocksPerGrid*sizeof(int));
  occa::memory b_tpHS_gpu = device.malloc(blocksPerGrid*sizeof(int));
  //---------------------------------------------------
  
  boost::progress_timer t_reduce1;
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
  
  cout <<"Scan_add1 run time: " << t_reduce1.elapsed() << endl; 

  //-----------------------b2--------------------
  

  boost::progress_timer t_reduce2;

  Scan_add2_p1(a_gpu, b_gpu2, b_tp2_gpu);
  
  b_gpu2.copyTo(b_cpu2);
  
  if (verbose)
  {
  cout <<"b_2 p1: ";
  for(int i =0; i< N; ++i) cout << b_cpu2[i] <<", ";
  cout << endl;
  }
  
  Scan_add2_p2(b_tp2_gpu, b_gpu2);
  
  b_gpu2.copyTo(b_cpu2);
  if (verbose)
  {
  cout <<"b_2 p2: ";
  for(int i =0; i< N; ++i) cout << b_cpu2[i] <<", ";
  cout << endl;
  }
  
  cout <<"Scan_add2 run time: " << t_reduce2.elapsed() << endl; 

//----------------------------------------------------
  boost::progress_timer t_HS;

  Scan_add_HS(a_gpu, b_gpuHS, b_tpHS_gpu);
  
  b_gpuHS.copyTo(b_cpuHS);
  //if (verbose)
  {
  cout <<"b_HS: ";
  for(int i =0; i< N; ++i) cout << b_cpuHS[i] <<", ";
  cout << endl;
  }
  
  cout <<"Scan_add_HS run time: " << t_HS.elapsed() << endl;

//----------------------------------------------------
  /*
  Scan_add1_p1.free();
  Scan_add1_p2.free();
  Scan_add2_p1.free();
  Scan_add2_p2.free();
  Scan_add_HS.free();
  */
  bool res1(true), res2(true), res_HS(true);

  
  
  for (int i=0; i < N; ++i)
  {
    if (b_ref[i] != b_cpu1[i])
    {
      res1 = false;
      printf("id= %d not equal; b_ref=%d, b1=%d\n",i, b_ref[i], b_cpu1[i]);
      break;
    }
  }
   
  for (int i=0; i < N; ++i)
  {
    if (b_ref[i] != b_cpu2[i])
    {
      res2 = false;
      printf("id= %d not equal; b_ref=%d, b2=%d\n",i, b_ref[i], b_cpu2[i]);
      break;
    }
  }

  for (int i=0; i < N; ++i)
  {
    if (b_ref[i] != b_cpuHS[i])
    {
      res_HS = false;
      printf("id= %d not equal; b_ref=%d, b_HS=%d\n",i, b_ref[i], b_cpuHS[i]);
      break;
    }
  }
  
  if (res1 && res2 && res_HS)
    {
      cout << "test pass :)" << endl;
      return 1;
    }
  else 
  {
    cout << "test failed :(" << endl;
    return 0;
  }

  device.free();
}
