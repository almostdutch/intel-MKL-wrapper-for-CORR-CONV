/*
 * Content:
 * Example of 2-dimensional linear convolution operation on double precision data
 * 
 * mode: VSL_CORR_MODE_AUTO, VSL_CORR_MODE_DIRECT, VSL_CORR_MODE_FFT
 * internal_precision: VSL_CORR_PRECISION_SINGLE, VSL_CORR_PRECISION_DOUBLE
 */

#include <iostream> 
#include "intel_mkl_conv.h"
using namespace std;

int main() {
	const int mode = VSL_CORR_MODE_FFT; 
	const int internal_precision = VSL_CORR_PRECISION_DOUBLE;
	const int dims = 2;

	const double kernel[2*2] = {1, 2,  3, 4}; // matrix 2x2 column-major format
	const double data_in[3*3] = {1, 2, 3,  1, 2, 3,  1, 2, 3}; // matrix 3x3 column-major format
	double data_out[4*4] = {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0}; // matrix 4x4 column-major format
	
	const int kernel_shape[] = { 2, 2 };
	const int data_in_shape[] = { 3, 3 };
	const int data_out_shape[] = { 4, 4 };

	Conv(mode, internal_precision, dims, kernel, 
			data_in, data_out, kernel_shape, data_in_shape, data_out_shape);

	cout << endl;
	cout << "2d convolution test" << endl;
	cout << "Input data: " << endl;
	for (int i = 0; i < data_in_shape[0]; i++) {
		for (int j = 0; j < data_in_shape[1]; j++) {
			cout << data_in[i + data_in_shape[0] * j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "Kernel: " << endl;
	for (int i = 0; i < kernel_shape[0]; i++) {
		for (int j = 0; j < kernel_shape[1]; j++) {
			cout << kernel[i + kernel_shape[0] * j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "Output data: " << endl;
	for (int i = 0; i < data_out_shape[0]; i++) {
		for (int j = 0; j < data_out_shape[1]; j++) {
			cout << data_out[i + data_out_shape[0] * j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	return 0;
}
