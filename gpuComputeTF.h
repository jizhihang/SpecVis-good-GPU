void __global__ kernelTFToBuffer(_precision* gpuMetric, unsigned char* gpuBuffer,
								 unsigned int samples, unsigned int lines,
								 transferFuncType tfType, _precision tfMin, _precision tfMax,
								 float r, float g, float b)
{
	//get the coordinate of the thread
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iv = blockIdx.y * blockDim.y + threadIdx.y;
	if(iu >= samples || iv >= lines) return;

	int i = iv * samples + iu;

	//get the metric value
	_precision val = gpuMetric[i];

	float a = 0;
	//if the metric is outside of the selected region, the TF will be zero
	if(val < tfMin || val > tfMax)
		a = 0;
	else
	{
		if(tfType == tfConstant)
			a = 1;
		else if(tfType == tfLinearUp)
			a = (val - tfMin) / (tfMax - tfMin);
		else if(tfType == tfLinearDown)
			a = (tfMax - val) / (tfMax - tfMin);
		else if(tfType == tfGaussian)
		{
			_precision meanVal = (tfMax + tfMin)/2;
			_precision width = (tfMax - tfMin)/6;

			a = expf(- powf(val - meanVal, 2) / (2 * width * width));
		}
	}


	gpuBuffer[i * BUFFER_PIXEL_SIZE + 0] = a * r;
	gpuBuffer[i * BUFFER_PIXEL_SIZE + 1] = a * g;
	gpuBuffer[i * BUFFER_PIXEL_SIZE + 2] = a * b;
}

void gpuTFToBuffer()
{
	//This function renders the selected transfer function to the display buffer

	//simplify the info for the transfer function
	int tf = P.selectedTF;
	int m = P.tfList[tf].sourceMetric;
	transferFuncType tfType = P.tfList[tf].type;

	//calculate the size of the buffer
	size_t size = P.dim.x * P.dim.y * BUFFER_PIXEL_SIZE;

	//map the buffer and create a pointer to GPU memory
	cudaGraphicsMapResources(1, &P.gpu_cudaResource, NULL);
	unsigned char* gpuPtr;
	cudaGraphicsResourceGetMappedPointer((void**)&gpuPtr, &size, P.gpu_cudaResource);

	//set the buffer to a specified value
	//cudaMemset(gpuPtr, 128, size);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(P.dim.x / dimBlock.x + 1, P.dim.y / dimBlock.y + 1);
	kernelTFToBuffer<<<dimGrid, dimBlock>>>(P.metricList[m].gpuMetric, gpuPtr,
											P.dim.x, P.dim.y,
											tfType, P.tfList[tf].tfMin, P.tfList[tf].tfMax,
											P.tfList[tf].r, P.tfList[tf].g, P.tfList[tf].b);

	//unmap the buffer so that it can be rendered using OpenGL
	cudaGraphicsUnmapResources(1, &P.gpu_cudaResource, NULL);
}
