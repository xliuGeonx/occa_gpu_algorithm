//---------------CUDA algorithm[Reduce] executated by OCCA2---------------------
//--------------------------by Xin at GeonX-------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "GeneralTools.h"
#include <boost/progress.hpp>

#include "occa.hpp"

using namespace std;

int main(int argc, char **argv){

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
    args.addDefine("bDim",threadsPerBlock);
    args.addDefine("gDim",blocksPerGrid);
    args.addCompilerFlag(GeneralTools::get_occa_local_include());
  // build Jacobi kernel
  occa::kernel inner_product_reduce0 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Reduce.occa"), "inner_product_ReduceDivergent", args);
  occa::kernel inner_product_reduce1 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Reduce.occa"), "inner_product_ReduceNoDivergent", args);
  occa::kernel inner_product_reduce2 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Reduce.occa"), "inner_product_ReduceSequential", args);
  occa::kernel inner_product_reduce3 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("Reduce.occa"), "inner_product_ReduceSequentialUnroll", args);
  // set thread array for Jacobi iteration
  int dims = 1;
  occa::dim inner(threadsPerBlock);
  occa::dim outer(blocksPerGrid);
  inner_product_reduce0.setWorkingDims(dims, inner, outer);
  inner_product_reduce1.setWorkingDims(dims, inner, outer);
  inner_product_reduce2.setWorkingDims(dims, inner, outer);
  inner_product_reduce3.setWorkingDims(dims, inner, outer);

  size_t sz = N*sizeof(int);

  // allocate array on HOST
  int *a_cpu = (int*) malloc(sz);
  int *b_cpu = (int*) malloc(sz);
  int *pproduct_cpu = (int*) malloc(blocksPerGrid*sizeof(int));
  //int product = 0;

  for(int i=0;i<N;++i)
  {
    a_cpu[i] = i;
    b_cpu[i] = i*2;
  }
  
  // allocate array on DEVICE (copy from HOST)
  occa::memory a_gpu = device.malloc(sz, a_cpu);
  occa::memory b_gpu = device.malloc(sz, b_cpu);
  occa::memory pproduct_gpu0 = device.malloc(blocksPerGrid*sizeof(int), pproduct_cpu);
  occa::memory pproduct_gpu1 = device.malloc(blocksPerGrid*sizeof(int), pproduct_cpu);
  occa::memory pproduct_gpu2 = device.malloc(blocksPerGrid*sizeof(int), pproduct_cpu);
  occa::memory pproduct_gpu3 = device.malloc(blocksPerGrid*sizeof(int), pproduct_cpu);
  // queue kernel
 
 
  
  // copy result to HOST
  
  boost::progress_timer t_reduce0;
  inner_product_reduce0(a_gpu, b_gpu,pproduct_gpu0);
  pproduct_gpu0.copyTo(pproduct_cpu);
  int product0 = 0;

  for (int i=0; i< blocksPerGrid; ++i)
  {
    product0 += pproduct_cpu[i];
  }

  cout << "Product0= "  << product0 << endl;

  cout <<"inner_product_reduce0 run time: " << t_reduce0.elapsed() << endl; 
  
  
  //-------------------------------------------------

  boost::progress_timer t_reduce1;
  inner_product_reduce1(a_gpu, b_gpu,pproduct_gpu1);
  pproduct_gpu1.copyTo(pproduct_cpu);
  int product1 = 0;

  for (int i=0; i< blocksPerGrid; ++i)
  {
    product1 += pproduct_cpu[i];
  }

  cout << "Product1= "  << product1 << endl;

  cout <<"inner_product_reduce1 run time: " << t_reduce1.elapsed() << endl; 
//----------------------------------------------------
  
  boost::progress_timer t_reduce2;
  inner_product_reduce2(a_gpu, b_gpu,pproduct_gpu2);
  pproduct_gpu2.copyTo(pproduct_cpu);
  int product2 = 0;

  for (int i=0; i< blocksPerGrid; ++i)
  {
    product2 += pproduct_cpu[i];
  }

  /* print out results */
  cout << "Product2= "  << product2 << endl;

  cout <<"inner_product_reduce2 run time: " << t_reduce2.elapsed() << endl;

  //-----------------------------------------------------

  boost::progress_timer t_reduce3;
  inner_product_reduce3(a_gpu, b_gpu,pproduct_gpu3);
  pproduct_gpu3.copyTo(pproduct_cpu);
  int product3 = 0;

  for (int i=0; i< blocksPerGrid; ++i)
  {
    product3 += pproduct_cpu[i];
  }

  /* print out results */
  cout << "Product3= "  << product3 << endl;

  cout <<"inner_product_reduce3 run time: " << t_reduce3.elapsed() << endl;
  

  //------------------------------------------------------

  inner_product_reduce0.free();
  inner_product_reduce1.free();
  inner_product_reduce2.free();
  inner_product_reduce3.free();
  
  boost::progress_timer t_reduce4;
  int ref_p= 0;
  for (int i=0; i< N; ++i) ref_p+= 2*i*i;
  cout << "ref_p= " << ref_p << endl;
  cout <<"inner_product_reduce4 run time: " << t_reduce4.elapsed() << endl;
  if (product0 == ref_p &&  product1 == ref_p &&  product2 == ref_p &&  product3 == ref_p)
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
