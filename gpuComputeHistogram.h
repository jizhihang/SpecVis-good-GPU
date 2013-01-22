void gpuFindHistogramExtrema(int &iMin, int &iMax)
{
    //create status variables
    cublasStatus_t stat;
    cublasHandle_t handle;

	//create a CUBLAS handle
	stat = cublasCreate(&handle);
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("CUBLAS initialization failed\n");
		return;
	}

	//perform the computations via CUBLAS
	int size = P.dim.z * P.histBins;
	stat = cublasIsamin(handle, size, (float*)P.gpuHistogram, 1, &iMin);
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("CUBLAS initialization failed\n");
		return;
	}

	stat = cublasIsamax(handle, size, (float*)P.gpuHistogram, 1, &iMax);
    if(stat != CUBLAS_STATUS_SUCCESS){
		printf("CUBLAS initialization failed\n");
		return;
	}

    //correct for the 1-based indexing used in CUBLAS
	iMin--;
	iMax--;

	//delete resources
	stat = cublasDestroy(handle);

}

__global__ void kernelComputeHistogram(precision* data, histoPrecision* histogram,
							unsigned int nBins, unsigned int nSamples, unsigned int nLines, unsigned int nBands,
							precision vMin, precision vMax, unsigned int histWindow, int px, int py)
{
    int H[100];
    for(int si=0; si<100; si++)
        H[si] = 0;

	//compute the index into the data set
	int band = blockIdx.x * blockDim.x + threadIdx.x;
	if(band >= nBands) return;


	//compute the index to the band
	unsigned int iBand = band * nSamples * nLines;

	//determine the bin size
	precision v;
	precision binSize = (vMax - vMin) / (nBins + 1);

    //determine the rectangle along which the histogram is computed
    unsigned int x, y, i;
    int xStart = px - (int)histWindow/2;
    int xEnd = xStart + histWindow;
    int yStart = py - (int)histWindow/2;
    int yEnd = yStart + histWindow;

    //clip the rectangle to the image boundary
    if(xStart < 0) xStart = 0;
    if(yStart < 0) yStart = 0;
    if(xEnd > nSamples) xEnd = nSamples - 1;
    if(yEnd > nLines) yEnd = nLines - 1;

    for(x=xStart; x<xEnd; x++)
        for(y=yStart; y<yEnd; y++)
        {
            i = y * nSamples + x;
            //get the value
            v = data[iBand + i];

            //compute the appropriate bin
            if(v >= vMin && v < vMax)
            {
                unsigned int bin = (v - vMin)/binSize;
                H[bin]++;
            }
        }

	for(int si=0; si<100; si++)
        histogram[si * nBands + band] = H[si];

}

void gpuComputeHistogram()
{
	//this function computes the spectral histogram

	//set all histogram bins to zero
	cudaMemset(P.gpuHistogram, 0, sizeof(histoPrecision) * P.histBins * P.dim.z);

	//Initial test: Assign one thread per band, where each thread computes the histogram for that band
	gpuStartTimer();
	dim3 dimBlock(BLOCK_SIZE, 1);
	dim3 dimGrid(P.dim.z / dimBlock.x + 1, 1);
	kernelComputeHistogram<<<dimGrid, dimBlock>>>(P.gpuData, P.gpuHistogram,
												P.histBins, P.dim.x, P.dim.y, P.dim.z,
												P.spectralMin, P.spectralMax,
												P.histWindow, P.currentX, P.currentY);

	float t = gpuStopTimer();
	printf("Time to compute histogram: %f\n", t);

}

__global__ void kernelHistogramToBuffer(histoPrecision* histogram, unsigned char* buffer, unsigned int nBands, unsigned int nBins, histoPrecision vMin, histoPrecision vMax)
{
	//get the coordinate of the thread
	int band = blockIdx.x * blockDim.x + threadIdx.x;
	int bin = blockIdx.y * blockDim.y + threadIdx.y;
	if(band >= nBands || bin >= nBins) return;

	unsigned int i = bin * nBands + band;

	histoPrecision histVal = histogram[i];

	//log scaling

	double dhistVal;
	if(histVal <= 0)
        dhistVal = 0;
    else
        dhistVal = log10((double)histVal);

	double dvMin, dvMax;

	if(vMin <= 0)
        dvMin = 0;
    else
        dvMin = log10((double)vMin);

	dvMax = log10((double)vMax);


	//double scaledVal = (histVal - dvMin)/(dvMax - dvMin);
	double scaledVal = (dhistVal - dvMin) / (dvMax - dvMin);

    unsigned char I = scaledVal * 255;

	buffer[i * BUFFER_PIXEL_SIZE + 0] = I;
	buffer[i * BUFFER_PIXEL_SIZE + 1] = I;
	buffer[i * BUFFER_PIXEL_SIZE + 2] = I;

}

void gpuHistogramToBuffer()
{
    //compute the extrema
    int iMin, iMax;
    gpuFindHistogramExtrema(iMin, iMax);

    //copy the extrema values from the GPU to the CPU
    histoPrecision vMin, vMax;
    cudaMemcpy(&vMin, P.gpuHistogram + iMin, sizeof(histoPrecision), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vMax, P.gpuHistogram + iMax, sizeof(histoPrecision), cudaMemcpyDeviceToHost);

	//calculate the size of the buffer
	size_t size = P.dim.z * P.histBins * BUFFER_PIXEL_SIZE;

	//map the buffer and create a pointer to GPU memory
	cudaGraphicsMapResources(1, &P.gpu_cudaHistResource, NULL);
	unsigned char* gpuPtr;
	cudaGraphicsResourceGetMappedPointer((void**)&gpuPtr, &size, P.gpu_cudaHistResource);


	//set up the thread grid
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(P.dim.z / dimBlock.x + 1, P.histBins / dimBlock.y + 1);

	//apply a kernel to transform the histogram into a visible image
	kernelHistogramToBuffer<<<dimGrid, dimBlock>>>(P.gpuHistogram, gpuPtr,
									P.dim.z, P.histBins, vMin, vMax);

	//unmap the buffer so that it can be rendered using OpenGL
	cudaGraphicsUnmapResources(1, &P.gpu_cudaHistResource, NULL);
}
