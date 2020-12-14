#include <vector>
#include <complex>
#include "mkl.h"

typedef std::complex<float> cf;
typedef std::complex<double> cd;
typedef MKL_Complex8 cf_MKL;
typedef MKL_Complex16 cd_MKL;

inline void _CountElements(size_t &kernel_length, size_t &data_in_length,
		size_t &data_out_length, const int dims, const int *const kernel_shape,
		const int *const data_in_shape, const int *const data_out_shape);

template<typename T_MKL, typename T>
inline void _ConversionFromComplexToComplexMKL(std::vector<T_MKL> & vkernel_MKL,
		std::vector<T_MKL> & vdata_in_MKL, const size_t kernel_length,
		const size_t data_in_length, const T *const kernel,
		const T *const data_in);

template<typename T_MKL, typename T>
inline void _ConversionFromComplexMKLToComplex(T *const data_out,
		const size_t data_out_length, const T_MKL *const data_out_MKL);

void Conv(const int mode, const int internal_precision, const int dims, 
		const float *const kernel, const float *const data_in, float *const data_out,
		const int *const kernel_shape, const int *const data_in_shape, const int *const data_out_shape);

void Conv(const int mode, const int internal_precision, const int dims, 
		const double *const kernel, const double *const data_in, double *const data_out,
		const int *const kernel_shape, const int *const data_in_shape, const int *const data_out_shape);

void Conv(const int mode, const int internal_precision, const int dims, 
		const cf *const kernel, const cf *const data_in, cf *const data_out,
		const int *const kernel_shape, const int *const data_in_shape, const int *const data_out_shape);

void Conv(const int mode, const int internal_precision, const int dims, 
		const cd *const kernel, const cd *const data_in, cd *const data_out,
		const int *const kernel_shape, const int *const data_in_shape, const int *const data_out_shape);