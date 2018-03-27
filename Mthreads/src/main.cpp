#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "GeneralTools.h"
#include <boost/thread.hpp>
#include <boost/ref.hpp>

#include "occa.hpp"

typedef double hp;

const size_t N_simple = 10;
const size_t size_simple = N_simple*sizeof(hp);

void foo(occa::memory_v* mv)
{
  occa::device d;
  d.setup("CUDA",0,0);
  const size_t N_simple = 10;
  const size_t size_simple = N_simple*sizeof(hp);
  occa::memory arry_simpel_gpu(mv);

  hp arry_simple_cpu[N_simple];
  arry_simpel_gpu.copyTo(arry_simple_cpu);
  cout << "foo: ";
   for (unsigned int i=0; i<N_simple; ++i)
  {
    cout<< arry_simple_cpu[i] << " " ;
  }
  cout << endl;
};

void foo2(occa::memory& vec_gpu)
{
  occa::device d;
  d.setup("CUDA",0,0);
  const size_t N_simple = 10;
  const size_t size_simple = N_simple*sizeof(hp);

  hp arry_simple_cpu[N_simple];
  vec_gpu.copyTo(arry_simple_cpu);
  cout << "foo2: ";
  for (unsigned int i=0; i<N_simple; ++i)
  {
    cout<< arry_simple_cpu[i] << " " ;
  }
  cout << endl;
};



int main(int argc, char **argv){

  /* hard code platform and device number */
  int plat = 0;
  int dev = 0;

  occa::device device;
  device.setup("CUDA", plat, dev);

  hp arry_simple_cpu[N_simple];
  for (unsigned int i=0; i<N_simple; ++i)
  {
    arry_simple_cpu[i] = (hp)(i);
  }

  occa::memory arry_simpel_gpu = device.malloc(size_simple, arry_simple_cpu);
  
  occa::memory_v* mv(arry_simpel_gpu.getMHandle());

  boost::thread t0(foo,boost::ref(mv));
  boost::thread t1(foo2,boost::ref(arry_simpel_gpu));


  t0.join();
  t1.join();
  return 0;
};
