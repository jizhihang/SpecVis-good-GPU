void __global__ kernelComputeMetric(_precision* gpuMetric, _precision* gpuData, _precision* gpuReference, _precision refEpsilon,
									unsigned int startBand, unsigned int endBand,
									unsigned int* basePts, unsigned int nBasePts,
									unsigned int lines, unsigned int samples, unsigned int bands)
{
	//get the coordinate of the thread
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iv = blockIdx.y * blockDim.y + threadIdx.y;
	if(iu >= samples || iv >= lines) return;

	int iMetric = iv * samples + iu;

	//compute the reference (1.0 if no reference)
	_precision vRef = 1.0;
	if(gpuReference != NULL)
		vRef = gpuReference[iMetric];

	//find the initial baseline function
	int iNextBase = -1;
	_precision vBase, vSlope;
	float a = 0;
	if(nBasePts == 0)
	{
		vBase = 0;
		vSlope = 0;
	}
	else
	{
		//handle the boundary conditions
		if(startBand <= basePts[0])
		{
			vBase = gpuData[basePts[0] * samples * lines + iv * samples + iu];
			vSlope = 0;
			iNextBase = 0;
		}
		else if(startBand >= basePts[nBasePts-1])
		{
			vBase = gpuData[basePts[nBasePts-1] * samples * lines + iv * samples + iu];
			vSlope = 0;
		}
		//compute the linear baseline function
		else
		{
			iNextBase = 0;
			while(startBand >= basePts[iNextBase]) iNextBase++;
			a = (float)((float)basePts[iNextBase] - (float)startBand)/(float)((float)basePts[iNextBase] - (float)basePts[iNextBase-1]);
			_precision lowBase = gpuData[basePts[iNextBase-1] * samples * lines + iv * samples + iu];
			_precision highBase = gpuData[basePts[iNextBase] * samples * lines + iv * samples + iu];
			vBase = a * lowBase + (1.0 - a) * highBase;
			vSlope = (highBase - lowBase)/(float)(basePts[iNextBase] - basePts[iNextBase-1]);
		}
	}


	//iterate through each band, summing the results
	_precision vSum = 0.0;
	int ib;
	int nextBase = -1;
	if(iNextBase != -1)
		nextBase = basePts[iNextBase];
	for(int b = startBand; b <= endBand; b++)
	{
		//compute the index for the source pixel
		ib = b * samples * lines + iv * samples + iu;

		//integrate
		if(abs(vRef) > refEpsilon)
            vSum += (gpuData[ib] - vBase)/vRef;
		else
			vSum = nanf("_precision");

		//increment the baseline function
		vBase += vSlope;

		//adjust the baseline if necessary
		if(b == nextBase)
		{
			iNextBase++;
			if(iNextBase == nBasePts)
			{
				nextBase = -1;
				vSlope = 0;
			}
			else
			{
				a = (float)(basePts[iNextBase] - startBand)/(float)(basePts[iNextBase] - basePts[iNextBase-1]);
				_precision lowBase = gpuData[basePts[iNextBase-1] * samples * lines + iv * samples + iu];
				_precision highBase = gpuData[basePts[iNextBase] * samples * lines + iv * samples + iu];
				vBase = a * lowBase + (1.0 - a) * highBase;
				vSlope = (highBase - lowBase)/(float)(basePts[iNextBase] - basePts[iNextBase-1]);
			}
		}
	}

	gpuMetric[iMetric] =  vSum / (float)(endBand - startBand + 1);
}

void gpuComputeMetric(unsigned int m)
{
	size_t size = sizeof(_precision) * P.dim.x * P.dim.y;

	//allocate memory if necessary
	if(P.metricList[m].gpuMetric == NULL)
		HANDLE_ERROR(cudaMalloc(&P.metricList[m].gpuMetric, size));

	//simplify
	_precision* gpuMetricPtr = P.metricList[m].gpuMetric;

	//copy the current metric's baseline points to the GPU
	unsigned int* gpuBasePts;
	unsigned int nBasePts;
	gpuUploadBasePts(m, &gpuBasePts, nBasePts);

	//compute the starting and ending band for the metric
	unsigned int startBand = P.metricList[m].band - P.metricList[m].bandwidth/2;
	unsigned int endBand = startBand + P.metricList[m].bandwidth - 1;
	_precision refEpsilon = P.metricList[m].refEpsilon;

	//get the reference pointer (if there is one)
	_precision* gpuRef = NULL;
	int r = P.metricList[m].reference;
	if(r != -1)
		gpuRef = P.metricList[r].gpuMetric;

	//create the thread geometry
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(P.dim.x / dimBlock.x + 1, P.dim.y / dimBlock.y + 1);
	kernelComputeMetric<<<dimGrid, dimBlock>>>(gpuMetricPtr, P.gpuData, gpuRef, refEpsilon,
											   startBand, endBand,
											   gpuBasePts, nBasePts,
											   P.dim.x, P.dim.y, P.dim.z);

	//free memory
	HANDLE_ERROR(cudaFree(gpuBasePts));
	//HANDLE_ERROR(cudaMemset(gpuMetricPtr, 0, size));

}
