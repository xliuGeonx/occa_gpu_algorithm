//---------------CUDA algorithm[Bitonic Sort] executated by OCCA2---------------------
//--------------------------by Xin at GeonX-----------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "GeneralTools.h"
#include <boost/progress.hpp>
#include <math.h>
#include <boost/exception/detail/type_info.hpp>
#include <set>

#include "occa.hpp"

using namespace std;

typedef double hp;
typedef float lp;

//#define epsilon_datafloat std::numeric_limits<float>::epsilon()

void main(int argc, char **argv){

  const bool verbose = false;
  const bool verbose_simple = false;
  const bool verbose_mb = false;
  /* hard code platform and device number */
  int plat = 0;
  int dev = 0;

  occa::device device;
  device.setup("CUDA", plat, dev);

  // build jacobi kernel from source file
  const char *functionName = "";

  // arg
  bool res = true;
  const unsigned int nn=9999999;
  //cout << "input N: ";
  //cin>>nn;

  const unsigned int N = nn;
  const unsigned int N_block_mb = 128;
  const unsigned int threadsPerBlock = 512;
  const unsigned int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    occa::kernelInfo args;
    args.addDefine("datatype",boost::type_name<hp>());
    args.addDefine("size_t",boost::type_name<size_t>());
    args.addDefine("tiny",-LLONG_MAX); 
    args.addDefine("bDIM",threadsPerBlock);
    args.addCompilerFlag(GeneralTools::get_occa_local_include());
  // build kernels
    
  occa::kernel kernel_BSort_simple = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("BSort.occa"), "BSort_simple", args);

  occa::kernel kernel_BSort_mblocks_p1 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("BSort.occa"), "BSort_mblocks_p1", args);
  occa::kernel kernel_BSort_mblocks_p2 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("BSort.occa"), "BSort_mblocks_p2", args);

  occa::kernel kernel_BSort_random_p1 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("BSort.occa"), "BSort_random_p1", args);
  occa::kernel kernel_BSort_random_p2 = device.buildKernelFromSource(GeneralTools::get_occaKernels_path("BSort.occa"), "BSort_random_p2", args);

  // set thread array 
  int dims = 1;
  occa::dim inner(threadsPerBlock);
  occa::dim outer(blocksPerGrid);
  
  kernel_BSort_simple.setWorkingDims(dims,threadsPerBlock,1);

  kernel_BSort_mblocks_p1.setWorkingDims(dims,threadsPerBlock,N_block_mb);
  kernel_BSort_mblocks_p2.setWorkingDims(dims,threadsPerBlock,N_block_mb);



  //-----------------simple (only one block)------------------

  const size_t N_simple = threadsPerBlock;
  const size_t size_simple = N_simple*sizeof(hp);

  hp* arry_simple_cpu = (hp*) malloc(size_simple);
  vector<hp> arry_simple_ref(N_simple);

  for (unsigned int i=0; i<N_simple/2; ++i)
  {
    arry_simple_cpu[i] = (hp)(2*i);
  }

  for (unsigned int i=0; i<N_simple/2; ++i)
  {
    arry_simple_cpu[i+N_simple/2] = (hp)(2*i+1);
  }

  for (unsigned int i=0; i<N_simple; ++i)
  {
    arry_simple_ref[N_simple-i-1] = (hp)(N_simple-i-1);
  }

  if (verbose_simple)
  {
     cout << "arry_simple_initial:\n";
     for (unsigned int i=0; i<N_simple; ++i)
     {
       cout << arry_simple_cpu[i] <<" ";
     }
     cout << endl;

     cout << "arry_simple_ref:\n";
     for (unsigned int i=0; i<N_simple; ++i)
     {
       cout << arry_simple_ref[i] <<" ";
     }
     cout << endl;
  }

  occa::memory arry_simpel_gpu = device.malloc(size_simple, arry_simple_cpu);

  kernel_BSort_simple
    (
    arry_simpel_gpu
    );

  arry_simpel_gpu.copyTo(arry_simple_cpu);

  if (verbose_simple)
  {
     cout << "arry_simple_gpu:\n";
     for (unsigned int i=0; i<N_simple; ++i)
     {
       cout << arry_simple_cpu[i] <<" ";
     }
     cout << endl;
  }

  for (unsigned int i=0; i<N_simple; ++i)
  {
   if (abs(arry_simple_cpu[i]-arry_simple_ref[i]) > 1e-3 )
   {
     res = false;
     cout << "node at " << i << " is not equal\n";
     cout << "arry_simple_cpu = "<< arry_simple_cpu[i] << "; ref = " << arry_simple_ref[i] << endl;
     break;
   }
  }


  //---------------------mblocks---------------

  const size_t N_mb = threadsPerBlock * N_block_mb;
  const size_t size_mblocks = N_mb*sizeof(hp);

  hp* arry_mb_cpu = (hp*) malloc(size_mblocks);
  vector<hp> arry_mb_ref(N_mb);
  
  for (unsigned int i=0; i<N_mb/2; ++i)
  {
    arry_mb_cpu[i] = (hp)(2*i);
  }

  for (unsigned int i=0; i<N_mb/2; ++i)
  {
    arry_mb_cpu[i+N_mb/2] = (hp)(2*i+1);
  }
  
  for (unsigned int i=0; i<N_mb; ++i)
  {
    arry_mb_ref[N_mb-i-1] = (hp)(N_mb-i-1);
  }

  if (verbose_mb)
  {
     cout << "arry_mb_initial:\n";
     for (unsigned int i=0; i<N_mb; ++i)
     {
       cout << arry_mb_cpu[i] <<" ";
     }
     cout << endl;

     cout << "arry_mb_ref:\n";
     for (unsigned int i=0; i<N_mb; ++i)
     {
       cout << arry_mb_ref[i] <<" ";
     }
     cout << endl;
  }

  occa::memory arry_mb_gpu = device.malloc(size_mblocks, arry_mb_cpu);

  kernel_BSort_mblocks_p1
  (
    arry_mb_gpu
  );
  
  if (verbose_mb)
  {
     arry_mb_gpu.copyTo(arry_mb_cpu);
     cout << "arry_mb_gpu_p1:\n";
     for (unsigned int i=0; i<N_mb; ++i)
     {
       cout << arry_mb_cpu[i] <<" ";
     }
     cout << endl;
  }


  for(unsigned int i= threadsPerBlock*2; i <=N_mb; i<<=1)
  {
    for (unsigned int j=(i>>1); j>0; j>>=1)
    {
      kernel_BSort_mblocks_p2
      (
        i,
        j,
        arry_mb_gpu
      );
    }
  }

  arry_mb_gpu.copyTo(arry_mb_cpu);

  if (verbose_mb)
  {
     cout << "arry_mb_gpu:\n";
     for (unsigned int i=0; i<N_mb; ++i)
     {
       cout << arry_mb_cpu[i] <<" ";
     }
     cout << endl;
  }

  
  for (unsigned int i=0; i<N_mb; ++i)
  {
   if (abs(arry_mb_cpu[i]-arry_mb_ref[i]) > 1e-3 )
   {
     res = false;
     cout << "node at " << i << " is not equal\n";
     cout << "arry_mb_cpu = "<< arry_mb_cpu[i] << "; ref = " << arry_mb_ref[i] << endl;
     break;
   }
  }

  //-------------random arry---------------------
  const unsigned int p = (unsigned int)(log((hp)N)/log(2.));
  cout << "p_key= "<< p << endl;
  const unsigned int N_p = pow(2,p);
  const size_t size_ra = N*sizeof(hp);
  const size_t size_ra_p = N_p*sizeof(hp);
  hp* arry_ra_cpu = (hp*) malloc(size_ra);
  hp* arry_temp_cpu = (hp*) malloc(size_ra_p);
  vector<hp> arry_ra_ref(N);
  
  for (unsigned int i=0; i<N; ++i)
  {
    arry_ra_cpu[i] = (hp)(N-i-1);
  }
  
  for (unsigned int i=0; i<N; ++i)
  {
    arry_ra_ref[i] = (hp)(i);
  }

  if (verbose)
  {
     cout << "arry_ra_initial:\n";
     for (unsigned int i=0; i<N; ++i)
     {
       cout << arry_ra_cpu[i] <<" ";
     }
     cout << endl;

     cout << "arry_ra_ref:\n";
     for (unsigned int i=0; i<N; ++i)
     {
       cout << arry_ra_ref[i] <<" ";
     }
     cout << endl;
  }

  occa::memory arry_ra_gpu = device.malloc(size_ra, arry_ra_cpu);
  occa::memory arry_raR_gpu = device.malloc(size_ra_p);
  occa::memory arry_raL_gpu = device.malloc(size_ra_p);

  const unsigned int blocksPerGrid_ra_p1 = (N_p+threadsPerBlock-1)/threadsPerBlock;
  const unsigned int blocksPerGrid_ra_p2 = (N-N_p+threadsPerBlock-1)/threadsPerBlock;
  occa::dim inner_ra(threadsPerBlock);
  occa::dim outer_ra_p1(blocksPerGrid_ra_p1);
  occa::dim outer_ra_p2(blocksPerGrid_ra_p2);
  kernel_BSort_random_p1.setWorkingDims(1,inner_ra,outer_ra_p1);
  kernel_BSort_random_p2.setWorkingDims(1,inner_ra,outer_ra_p2);

  boost::progress_timer t_Bsort;

  kernel_BSort_random_p1
    (
    N,
    N_p,
    arry_ra_gpu,
    arry_raL_gpu,
    arry_raR_gpu
    );

  if (verbose)
  {
     arry_raL_gpu.copyTo(arry_temp_cpu);
     cout << "arry_raL_gpu:\n";
     for (unsigned int i=0; i<N_p; ++i)
     {
       cout << arry_temp_cpu[i] <<" ";
     }
     cout << endl;

     arry_raR_gpu.copyTo(arry_temp_cpu);
     cout << "arry_raR_gpu:\n";
     for (unsigned int i=0; i<N_p; ++i)
     {
       cout << arry_temp_cpu[i] <<" ";
     }
     cout << endl;
  }

  kernel_BSort_mblocks_p1.setWorkingDims(1,threadsPerBlock,outer_ra_p1);
  kernel_BSort_mblocks_p2.setWorkingDims(1,threadsPerBlock,outer_ra_p1);

  kernel_BSort_mblocks_p1
  (
    arry_raL_gpu
  );

  for(unsigned int i= threadsPerBlock*2; i <=N_p; i<<=1)
  {
    for (unsigned int j=(i>>1); j>0; j>>=1)
    {
      kernel_BSort_mblocks_p2
      (
        i,
        j,
        arry_raL_gpu
      );
    }
  }

  kernel_BSort_mblocks_p1
  (
    arry_raR_gpu
  );

  for(unsigned int i= threadsPerBlock*2; i <=N_p; i<<=1)
  {
    for (unsigned int j=(i>>1); j>0; j>>=1)
    {
      kernel_BSort_mblocks_p2
      (
        i,
        j,
        arry_raR_gpu
      );
    }
  }
  
  if (verbose)
  {
     arry_raL_gpu.copyTo(arry_temp_cpu);
     cout << "arry_raL_gpu_p1:\n";
     for (unsigned int i=0; i<N_p; ++i)
     {
       cout << arry_temp_cpu[i] <<" ";
     }
     cout << endl;

     arry_raR_gpu.copyTo(arry_temp_cpu);
     cout << "arry_raR_gpu_p1:\n";
     for (unsigned int i=0; i<N_p; ++i)
     {
       cout << arry_temp_cpu[i] <<" ";
     }
     cout << endl;
  }

  kernel_BSort_random_p2
    (
    N,
    N_p,
    arry_raL_gpu,
    arry_raR_gpu
    );

  if (verbose)
  {
     arry_raL_gpu.copyTo(arry_temp_cpu);
     cout << "arry_raL_gpu_p2:\n";
     for (unsigned int i=0; i<N_p; ++i)
     {
       cout << arry_temp_cpu[i] <<" ";
     }
     cout << endl;

     arry_raR_gpu.copyTo(arry_temp_cpu);
     cout << "arry_raR_gpu_p2:\n";
     for (unsigned int i=0; i<N_p; ++i)
     {
       cout << arry_temp_cpu[i] <<" ";
     }
     cout << endl;
  }

  kernel_BSort_mblocks_p1
  (
    arry_raL_gpu
  );

  for(unsigned int i= threadsPerBlock*2; i <=N_p; i<<=1)
  {
    for (unsigned int j=(i>>1); j>0; j>>=1)
    {
      kernel_BSort_mblocks_p2
      (
        i,
        j,
        arry_raL_gpu
      );
    }
  }

  kernel_BSort_mblocks_p1
  (
    arry_raR_gpu
  );

  for(unsigned int i= threadsPerBlock*2; i <=N_p; i<<=1)
  {
    for (unsigned int j=(i>>1); j>0; j>>=1)
    {
      kernel_BSort_mblocks_p2
      (
        i,
        j,
        arry_raR_gpu
      );
    }
  }

  cout << "Bsort time: " << t_Bsort.elapsed() << "s\n";

  arry_raL_gpu.copyTo(arry_temp_cpu);
  for (unsigned int i=0; i<N_p; ++i)
  {
      arry_ra_cpu[i] = arry_temp_cpu[i];
  }

  arry_raR_gpu.copyTo(arry_temp_cpu);
  for (unsigned int i=0; i<(N-N_p); ++i)
  {
      arry_ra_cpu[i+N_p] = arry_temp_cpu[2*N_p-N+i];
  }

  if (verbose)
  {
    cout << "arry_ra_result:\n";
    for (unsigned int i=0; i<N; ++i)
    {
       cout << arry_ra_cpu[i] <<" ";
    }
    cout << endl;
  }

  for (unsigned int i=0; i<N; ++i)
  {
   if (abs(arry_ra_cpu[i]-arry_ra_ref[i]) > 1e-3 )
   {
     res = false;
     cout << "node at " << i << " is not equal\n";
     cout << "arry_mb_cpu = "<< arry_ra_cpu[i] << "; ref = " << arry_ra_ref[i] << endl;
     break;
   }
  }

  //--------------RB tree-----------------
  set<hp> set_hp;
  vector<hp> set_temp(N);

  for (unsigned int i=0; i<N; ++i)
  {
   set_temp[i]=N-1-i;
  }

  boost::progress_timer t_set;
  for (unsigned int i=0; i<N; ++i)
  {
   set_hp.insert(set_temp[i]);
  }
  cout << "set time: " << t_set.elapsed() << "s\n";
  //------------------------------------------

  if (res)
    {
      cout << "test pass :)" << endl;
      //return 1;
    }
  else 
  {
    cout << "test failed :(" << endl;
    //return 0;
  }
}
