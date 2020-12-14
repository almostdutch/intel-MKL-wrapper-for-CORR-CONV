#include <iostream>
#include "intel_mkl_corr.h"
using namespace std;

inline void _CountElements(size_t &kernel_length, size_t &data_in_length,
		size_t &data_out_length, const int dims, const int *const kernel_shape,
		const int *const data_in_shape, const int *const data_out_shape) {

	for (int dims_No = 0; dims_No < dims; ++dims_No) {
		kernel_length *= *(kernel_shape + dims_No);
		data_in_length *= *(data_in_shape + dims_No);
		data_out_length *= *(data_out_shape + dims_No);
	}
}

template<typename T_MKL, typename T>
inline void _ConversionFromComplexToComplexMKL(vector<T_MKL> & vkernel_MKL,
		vector<T_MKL> & vdata_in_MKL, const size_t kernel_length,
		const size_t data_in_length, const T *const kernel,
		const T *const data_in) {

	T_MKL complex_MKL_temp;
	for (size_t el = 0; el < kernel_length; ++el) {		
		complex_MKL_temp.real = real(*(kernel + el));
		complex_MKL_temp.imag = imag(*(kernel + el));
		vkernel_MKL.at(el) = complex_MKL_temp;
	}

	for (size_t el = 0; el < data_in_length; ++el) {
		complex_MKL_temp.real = real(*(data_in + el));
		complex_MKL_temp.imag = imag(*(data_in + el));
		vdata_in_MKL.at(el) = complex_MKL_temp;
	}
}

template<typename T_MKL, typename T>
inline void _ConversionFromComplexMKLToComplex(T *const data_out,
		const size_t data_out_length, const T_MKL *const data_out_MKL) {

	for (size_t el = 0; el < data_out_length; ++el) {
		*(data_out + el) = T(data_out_MKL[el].real, data_out_MKL[el].imag);
	}
}

// correlation
// complex single precision
void Corr(const int mode, const int internal_precision, const int dims, 
		const cf *const kernel, const cf *const data_in, cf *const data_out,
		const int *const kernel_shape, const int *const data_in_shape, const int *const data_out_shape)
{
	
	// from complex to complex_MKL
	size_t kernel_length = 1, data_in_length = 1, data_out_length =
			1;
	_CountElements(kernel_length, data_in_length, data_out_length,
			dims, kernel_shape, data_in_shape, data_out_shape);

	cf_MKL *kernel_MKL, *data_in_MKL, *data_out_MKL, complex_MKL_temp;
	complex_MKL_temp.real = 0;
	complex_MKL_temp.imag = 0;
	vector<cf_MKL> vkernel_MKL(kernel_length, complex_MKL_temp), vdata_in_MKL(data_in_length, complex_MKL_temp), 
			vdata_out_MKL(data_out_length, complex_MKL_temp);
	
	_ConversionFromComplexToComplexMKL(vkernel_MKL, vdata_in_MKL, kernel_length, data_in_length, kernel, data_in);
	kernel_MKL = &vkernel_MKL.at(0);
	data_in_MKL = &vdata_in_MKL.at(0);
	data_out_MKL = &vdata_out_MKL.at(0);
		
	VSLCorrTaskPtr task;
	int status;

	status = vslcCorrNewTask(&task, mode, dims, kernel_shape, data_in_shape,
			data_out_shape);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslcCorrNewTask..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}
	
	status = vslCorrSetInternalPrecision (&task, internal_precision);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslCorrSetInternalPrecision..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}
	
	status = vslcCorrExec(task, kernel_MKL, NULL, data_in_MKL, NULL,
			data_out_MKL, NULL);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslcCorrExec..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}

	status = vslCorrDeleteTask(&task);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslCorrDeleteTask..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}

	// from complex_MKL to complex
	_ConversionFromComplexMKLToComplex(data_out, data_out_length,
			data_out_MKL);
}

// complex double precision
void Corr(const int mode, const int internal_precision, const int dims, 
		const cd *const kernel, const cd *const data_in, cd *const data_out,
		const int *const kernel_shape, const int *const data_in_shape, const int *const data_out_shape)
{
	
	// from complex to complex_MKL
	size_t kernel_length = 1, data_in_length = 1, data_out_length =
			1;
	_CountElements(kernel_length, data_in_length, data_out_length,
			dims, kernel_shape, data_in_shape, data_out_shape);

	cd_MKL *kernel_MKL, *data_in_MKL, *data_out_MKL, complex_MKL_temp;
	complex_MKL_temp.real = 0;
	complex_MKL_temp.imag = 0;
	vector<cd_MKL> vkernel_MKL(kernel_length, complex_MKL_temp), vdata_in_MKL(data_in_length, complex_MKL_temp), 
			vdata_out_MKL(data_out_length, complex_MKL_temp);
	
	_ConversionFromComplexToComplexMKL(vkernel_MKL, vdata_in_MKL, kernel_length, data_in_length, kernel, data_in);
	kernel_MKL = &vkernel_MKL.at(0);
	data_in_MKL = &vdata_in_MKL.at(0);
	data_out_MKL = &vdata_out_MKL.at(0);

	VSLCorrTaskPtr task;
	int status;

	status = vslzCorrNewTask(&task, mode, dims, kernel_shape, data_in_shape,
			data_out_shape);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vsldCorrNewTask..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}

	status = vslCorrSetInternalPrecision (&task, internal_precision);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslCorrSetInternalPrecision..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}
	
	status = vslzCorrExec(task, kernel_MKL, NULL, data_in_MKL, NULL,
			data_out_MKL, NULL);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vsldCorrExec..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}

	status = vslCorrDeleteTask(&task);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslCorrDeleteTask..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}

	// from complex_MKL to complex
	_ConversionFromComplexMKLToComplex(data_out, data_out_length,
			data_out_MKL);
}

// single precision
void Corr(const int mode, const int internal_precision, const int dims, 
		const float *const kernel, const float *const data_in, float *const data_out,
		const int *const kernel_shape, const int *const data_in_shape, const int *const data_out_shape)
{

	VSLCorrTaskPtr task;
	int status;

	status = vslsCorrNewTask(&task, mode, dims, kernel_shape, data_in_shape,
			data_out_shape);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslcCorrNewTask..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}
	
	status = vslCorrSetInternalPrecision (&task, internal_precision);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslCorrSetInternalPrecision..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}
	
	status = vslsCorrExec(task, kernel, NULL, data_in, NULL,
			data_out, NULL);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslcCorrExec..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}

	status = vslCorrDeleteTask(&task);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslCorrDeleteTask..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}
}

// double precision
void Corr(const int mode, const int internal_precision, const int dims, 
		const double *const kernel, const double *const data_in, double *const data_out,
		const int *const kernel_shape, const int *const data_in_shape, const int *const data_out_shape)
{

	VSLCorrTaskPtr task;
	int status;

	status = vsldCorrNewTask(&task, mode, dims, kernel_shape, data_in_shape,
			data_out_shape);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslcCorrNewTask..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}
		
	status = vsldCorrExec(task, kernel, NULL, data_in, NULL,
			data_out, NULL);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslcCorrExec..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}

	status = vslCorrDeleteTask(&task);
	if (status != VSL_STATUS_OK) {
		cout << "ERROR: during the call vslCorrDeleteTask..." << endl;
		cout << "ERROR: exit with status: " << status << endl;
		exit(1);
	}
}