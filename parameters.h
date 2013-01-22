#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <string>
#include "rtsVector3d.h"

#include <cuda_runtime_api.h>

#include "metrics.h"
#include "datatypes.h"
#include "transferfunc.h"

void UpdateSpatialWindow();
void UpdateSpectralWindow();
void UpdateWindows();
void UpdateUI();

static float clamp(float x, float a, float b)
{
    return x < a ? a : (x > b ? b : x);
}

struct parameterStruct
{
	//data file parameters
	precision* cpuData;
	precision* gpuData;
	cudaArray* gpuDataArray;
	vector3D<unsigned int> dim;
	string filename;

	//histogram parameters
	spectrumModeType spectrumMode;
	histoPrecision* gpuHistogram;     //actual histogram on the GPU
	unsigned int gpu_glHistBuffer;
	cudaGraphicsResource_t gpu_cudaHistResource;
	unsigned int histBins;
	unsigned int histWindow;

	//transfer functions
	vector<transferFuncStruct> tfList;
	int selectedTF;

	//system parameters
	unsigned int gpuTotalMemory;
	unsigned int gpuAvailableMemory;

	//display mode
	displayModeType displayMode;

	//OpenGL display parameters
	unsigned int gpu_glBuffer;
	cudaGraphicsResource_t gpu_cudaResource;

	//raw data visualization
	precision scaleMin;
	precision scaleMax;
	scaleModeType scaleMode;

	//spectral window
	precision spectralMin;
	precision spectralMax;

	//widget parameters
	unsigned int currentBand;
	int currentX;
	int currentY;

	//metrics
	vector<metricStruct> metricList;
	unsigned int selectedMetric;
	unsigned int selectedBaselinePoint;

	//specify default parameters
	parameterStruct(){
		cpuData = NULL;
		gpuData = NULL;
		gpuDataArray = NULL;

		displayMode = displayRaw;
		gpu_glBuffer = 0;

		currentBand = 100;
		scaleMode = automatic;
		scaleMin = 0.0;
		scaleMax = 1.0;

		spectralMin = 0.0;
		spectralMax = 1.0;

        spectrumMode = spectrum;
		gpuHistogram = NULL;
		histBins = 100;
		histWindow = 20;

		selectedMetric = 0;
		selectedTF = -1;
	}


};


#endif
