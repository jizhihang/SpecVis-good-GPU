#include <stdio.h>

#include "parameters.h"
extern parameterStruct P;

texture<_precision, 3, cudaReadModeElementType> texDataSet;

#include <cublas_v2.h>
#include <cudaHandleError.h>

#define BLOCK_SIZE				16
#include "gpuCreateTextures.h"
#include "gpuComputeSpectrum.h"
#include "gpuComputeHistogram.h"
#include "gpuComputeMetric.h"
#include "gpuComputeTF.h"

