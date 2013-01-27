void gpuFindBandExtrema(int &iMin, int &iMax, unsigned int band)
{
	//create status variables
    cublasStatus_t stat;
    cublasHandle_t handle;

	//find the memory location for the start of the band
	_precision* gpuBandPtr;
	gpuBandPtr = P.gpuData + P.dim.x * P.dim.y * band;

	//create a CUBLAS handle
	stat = cublasCreate(&handle);
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("CUBLAS initialization failed\n");
		return;
	}

	//perform the computations via CUBLAS
	int size = P.dim.x * P.dim.y;
	stat = cublasIsamin(handle, size, gpuBandPtr, 1, &iMin);
	stat = cublasIsamax(handle, size, gpuBandPtr, 1, &iMax);

	//correct for the 1-based indexing used in CUBLAS
	iMin--;
	iMax--;

	//delete resources
	stat = cublasDestroy(handle);

}

__global__ void kernelRawToBuffer(_precision* gpuRaw, unsigned char* buffer,
					unsigned int samples, unsigned int lines, unsigned int band,
					_precision vMin, _precision vMax)
{
	//get the coordinate of the thread
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	if(u >= samples || v >= lines) return;

	//calculate the indices into the buffer and the raw image
	int iBuffer = v*samples*BUFFER_PIXEL_SIZE + u*BUFFER_PIXEL_SIZE;
	int iBand = band * (samples * lines);
	int iRaw = iBand +  v*samples + u;

	_precision val = gpuRaw[iRaw];
	_precision valScaled = (val - vMin) / (vMax - vMin);

	int I = valScaled * 255;
	if(I < 0) I = 0;
	if(I > 254) I = 254;

	buffer[iBuffer + 0] = I;
	buffer[iBuffer + 1] = I;
	buffer[iBuffer + 2] = I;

}

void gpuRawToBuffer()
{
	//find the band extrema
	_precision vMin, vMax;
	if(P.scaleMode == automatic)
	{
		//use CUBLAS to compute the indices of the extrema
		int iMin, iMax;
		gpuFindBandExtrema(iMin, iMax, P.currentBand);

		//copy the extrema values from the GPU to the CPU
		unsigned int iBand = P.currentBand * P.dim.x * P.dim.y;
		cudaMemcpy(&vMin, P.gpuData + iBand + iMin, sizeof(_precision), cudaMemcpyDeviceToHost);
		cudaMemcpy(&vMax, P.gpuData + iBand + iMax, sizeof(_precision), cudaMemcpyDeviceToHost);

		//store the results in the parameter structure
		P.scaleMin = vMin;
		P.scaleMax = vMax;
	}
	else
	{
		vMin = P.scaleMin;
		vMax = P.scaleMax;
	}


	//calculate the size of the buffer
	size_t size = P.dim.x * P.dim.y * BUFFER_PIXEL_SIZE;

	//map the buffer and create a pointer to GPU memory
	cudaGraphicsMapResources(1, &P.gpu_cudaResource, NULL);
	unsigned char* gpuPtr;
	cudaGraphicsResourceGetMappedPointer((void**)&gpuPtr, &size, P.gpu_cudaResource);

	//set the buffer to a specified value

	//set up the thread grid
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(P.dim.x / dimBlock.x + 1, P.dim.y / dimBlock.y + 1);
	kernelRawToBuffer<<<dimGrid, dimBlock>>>(P.gpuData, gpuPtr,
									P.dim.x, P.dim.y, P.currentBand,
									vMin, vMax);

	//unmap the buffer so that it can be rendered using OpenGL
	cudaGraphicsUnmapResources(1, &P.gpu_cudaResource, NULL);
}

void __global__ kernelMetricToBuffer(_precision* gpuMetric, unsigned char* gpuBuffer, unsigned int lines, unsigned int samples,
									 _precision scaleLow, _precision scaleHigh)
{
	//get the coordinate of the thread
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iv = blockIdx.y * blockDim.y + threadIdx.y;
	if(iu >= samples || iv >= lines) return;

	int i = iv * samples + iu;

	_precision val = gpuMetric[i];
	_precision scaledVal = (val - scaleLow)/(scaleHigh - scaleLow) * 255;
	if(scaledVal < 0)
		scaledVal = 0;
	if(scaledVal > 254)
		scaledVal = 254;

	gpuBuffer[i * BUFFER_PIXEL_SIZE + 0] = scaledVal;
	gpuBuffer[i * BUFFER_PIXEL_SIZE + 1] = scaledVal;
	gpuBuffer[i * BUFFER_PIXEL_SIZE + 2] = scaledVal;


}

void gpuMetricToBuffer(unsigned int m)
{

	//calculate the size of the buffer
	size_t size = P.dim.x * P.dim.y * BUFFER_PIXEL_SIZE;

	//get a pointer to the metric (should already be computed)
	_precision* gpuMetric = P.metricList[m].gpuMetric;

	//map the buffer and create a pointer to GPU memory
	cudaGraphicsMapResources(1, &P.gpu_cudaResource, NULL);
	unsigned char* gpuBuffer;
	cudaGraphicsResourceGetMappedPointer((void**)&gpuBuffer, &size, P.gpu_cudaResource);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(P.dim.x / dimBlock.x + 1, P.dim.y / dimBlock.y + 1);
	//cudaMemset(gpuBuffer, 128, size);
	kernelMetricToBuffer<<<dimGrid, dimBlock>>>(gpuMetric, gpuBuffer, P.dim.x, P.dim.y, 
												P.metricList[m].scaleLow, P.metricList[m].scaleHigh);

	//unmap the buffer so that it can be rendered using OpenGL
	cudaGraphicsUnmapResources(1, &P.gpu_cudaResource, NULL);

}


