#include "GeneralTools.h"

using namespace std;

namespace GeneralTools {
  
  const std::string get_occaKernels_path(const std::string s)
  {
    std::stringstream ss;
    std::string occa_kernels_directory("C:/Users/xliu/Documents/xin00/OCCA_X/occa_gpu_algorithm/OCCAKernels");
    ss<<occa_kernels_directory<<"/"<<s;
    return ss.str();
  };

  //-----------------------------------------

  const std::string get_occa_local_include()
  {
    return "-IC:/Users/xliu/Documents/xin00/OCCA_X/occa_gpu_algorithm/OCCAKernels";
  };
}