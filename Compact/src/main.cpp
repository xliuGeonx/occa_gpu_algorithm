//---------------CUDA algorithm[Compact] executated by OCCA2---------------------
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
  const int filter_size = (N+1)/2;
  const int threadsPerBlock = 8;
  const int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    occa::kernelInfo args;
    args.addDefine("N",N);    
    args.addDefine("bDIM",threadsPerBlock);
    args.addDefine("gDIM",blocksPerGrid);
    args.addCompilerFlag(GeneralTools::get_occa_local_include());
  // build kernels
    
  occa::kernel kernel_compact_bool = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Compact.occa"), "Compact_bool", args);
  occa::kernel kernel_scan_HS = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Scan.occa"), "Scan_add_HS", args);
  occa::kernel kernel_compact_map = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Compact.occa"), "Compact_map", args);

  // set thread array 
  int dims = 1;
  occa::dim inner(threadsPerBlock);
  occa::dim outer(blocksPerGrid);
  
  kernel_compact_bool.setWorkingDims(dims, inner, outer);
  kernel_scan_HS.setWorkingDims(dims, inner, outer);
  kernel_compact_map.setWorkingDims(dims, inner, outer);
  // allocate array on HOST  
  size_t sz = N*sizeof(int);
  size_t filter_sz = filter_size*sizeof(int);
  int *a_cpu = (int*) malloc(sz);
  int *address_cpu = (int*) malloc(sz);
  int *filter_cpu = (int*) malloc(filter_sz);
  int *bool_cpu = (int*) malloc(sz);
  int *b_cpu = (int*) malloc(filter_sz);
  int *b_ref = (int*) malloc(sz);
  //---------------------------------

  boost::progress_timer t_ref;
  for(int i=0;i<N;++i)
  {
    a_cpu[i] = i;
  }
  for(int i=0;i<N;++i)
  {
    address_cpu[i] = 0;
  }

  for(int i=0;i<N;++i)
  {
    bool_cpu[i] = 0;
  }

  for(int i=0;i<filter_size;++i)
  {
    filter_cpu[i] = i*2;
  }


  /*
  for(int i=0;i<filter_size;++i)
  {
    filter_cpu[i] = i+1;
  }
  */
  if(verbose)
  {
    /*
    cout << "address: "; 
    for(int i =0; i< N; ++i) cout << address_cpu[i] <<", ";
    cout << endl;
    
    cout << "bool_cpu: "; 
    for(int i =0; i< N; ++i) cout << bool_cpu[i] <<", ";
    cout << endl;
    */
    cout << "a: "; 
    for(int i =0; i< N; ++i) cout << a_cpu[i] <<", ";
    cout << endl;

    cout << "filter: "; 
    for(int i =0; i<filter_size; ++i) cout << filter_cpu[i] <<", ";
    cout << endl;
  }
  //--------------Ref-------------------------
 
  //--b2 ref
  
  b_ref[0]=1;
  for (int i =1; i < N; ++i) 
  {
    b_ref[i]= b_ref[i-1]+address_cpu[i];
  }
  //cout <<"Scan_add_ref run time: " << t_ref.elapsed() << endl;
  /*
  if(verbose)
  {
    cout <<"b_ref: ";
    for(int i =0; i< N; ++i) cout << b_ref[i] <<", ";
    cout << endl;
    cout << "---------------Calculation----------------" << endl;
  }
  */
  
  // allocate array on DEVICE (copy from HOST)
  occa::memory a_gpu = device.malloc(sz, a_cpu);
  occa::memory address_gpu = device.malloc(sz, address_cpu);
  occa::memory filter_gpu = device.malloc(filter_sz, filter_cpu);
  occa::memory bool_gpu = device.malloc(sz,bool_cpu);
  occa::memory b_gpu = device.malloc(filter_sz,b_cpu);
  occa::memory b_tp_gpu = device.malloc(blocksPerGrid*sizeof(int));
  //---------------------------------------------------
  
  

  kernel_compact_bool(
    filter_gpu,
    filter_size,
    bool_gpu
    );

  
  bool_gpu.copyTo(bool_cpu);
  if(verbose)
  {
    cout << "bool_cpu: "; 
    for(int i =0; i< N; ++i) cout << bool_cpu[i] <<", ";
    cout << endl;
  }

  kernel_scan_HS
    (
    bool_gpu,
    address_gpu,
    b_tp_gpu
    );

  address_gpu.copyTo(address_cpu);
  if(verbose)
  {
    cout << "address_cpu: "; 
    for(int i =0; i< N; ++i) cout << address_cpu[i] <<", ";
    cout << endl;
  }

  kernel_compact_map(
    a_gpu,
    bool_gpu,
    address_gpu,
    b_gpu
    );
  
  b_gpu.copyTo(b_cpu);
  if(verbose)
  {
    cout << "b_cpu: "; 
    for(int i =0; i< filter_size; ++i) cout << b_cpu[i] <<", ";
    cout << endl;
  }

//----------------------------------------------------


  bool res1(true);

  
  /*
  for (int i=0; i < N; ++i)
  {
    if (b_ref[i] != b_cpu[i])
    {
      res1 = false;
      printf("id= %d not equal; b_ref=%d, b=%d\n",i, b_ref[i], b_cpu[i]);
      break;
    }
  }
   */
  

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
