void gpuUploadBasePts(int m, unsigned int** gpuBasePts, unsigned int &nBasePts)
{
    //get the current metric
    //int m = P.selectedMetric;

	//store the number of baseline points
	nBasePts = P.metricList[m].baselinePoints.size();

    //sort the baseline points
    vector<unsigned int> sortedPts = P.metricList[m].baselinePoints;
    sort(sortedPts.begin(), sortedPts.end());

	//allocate memory
	unsigned int* cpuBasePts = (unsigned int*)malloc(sizeof(unsigned int) * nBasePts);
	HANDLE_ERROR(cudaMalloc(gpuBasePts, sizeof(unsigned int) * nBasePts));

	//copy the baseline points into a linear array
	for(unsigned int i=0; i<nBasePts; i++)
		cpuBasePts[i] = sortedPts[i];

	//copy to the gpu
	HANDLE_ERROR(cudaMemcpy(*gpuBasePts, cpuBasePts, sizeof(unsigned int) * nBasePts, cudaMemcpyHostToDevice));

}

__global__ void kernelGetSpectrum(_precision* gpuSpectrum, _precision* gpuData, int x, int y,
								int samples, int lines, int bands,
								unsigned int* gpuBasePts, unsigned int nBasePts, _precision* gpuReference)
{
	int ib = blockIdx.x * blockDim.x + threadIdx.x;
	if(ib >= bands) return;

	int i = ib * samples * lines + y * samples + x;

	//compute the reference (1.0 if no reference)
	_precision vRef = 1.0;
	if(gpuReference != NULL)
		vRef = gpuReference[y * samples + x];

	_precision vBaseline;
	//spectrum is unchaged if there are no baseline points
	if(nBasePts == 0)
		vBaseline = 0;
	//otherwise handle the boundary cases
	else if(ib <= gpuBasePts[0])
		vBaseline = gpuData[gpuBasePts[0] * samples * lines + y * samples + x];
	else if(ib >= gpuBasePts[nBasePts-1])
		vBaseline = gpuData[gpuBasePts[nBasePts-1] * samples * lines + y * samples + x];
	//interpolate anything in between two baseline points
	else
	{
		//find the highest neighboring baseline point (in)
		unsigned int in = 0;
		while(gpuBasePts[in] <= ib) in++;

		_precision vUpper = gpuData[gpuBasePts[in] * samples * lines + y * samples + x];
		_precision vLower = gpuData[gpuBasePts[in-1] * samples * lines + y * samples + x];
		float a = (float)(ib - gpuBasePts[in-1])/(float)(gpuBasePts[in] - gpuBasePts[in-1]);
		vBaseline = a * vUpper + (1.0 - a) * vLower;
	}

	gpuSpectrum[ib] = (gpuData[i] - vBaseline)/vRef;
}

void gpuGetSpectrum(_precision* cpuSpectrum)
{
    //return an array of zeros if the current point is out of bounds
    if(P.currentX >= P.dim.x || P.currentY >= P.dim.y)
    {
        memset(cpuSpectrum, 0, sizeof(_precision) * P.dim.z);
        return;
    }

	//draw the selected metric if in metric mode
	int m;
	if(P.displayMode == displayMetrics)
		m = P.selectedMetric;
	else if(P.displayMode == displayTF)
	{
		int tf = P.selectedTF;
		m = P.tfList[tf].sourceMetric;
	}

    //handle baseline points
    unsigned int nBasePts;
    unsigned int* gpuBasePts = NULL;
    //if the raw data is being displayed, there are no baseline points
    if(P.displayMode == displayRaw)
        nBasePts = 0;
    //if metrics are being displayed, there may be baseline points
    else
        //call the function to upload them
        gpuUploadBasePts(m, &gpuBasePts, nBasePts);

	//allocate GPU space for the spectrum
	_precision* gpuSpectrum;
	cudaMalloc(&gpuSpectrum, sizeof(_precision) * P.dim.z);

	//get the reference pointer (if there is one)
	_precision* gpuRef = NULL;
	if(P.displayMode == displayMetrics || P.displayMode == displayTF)
	{
		int r = P.metricList[m].reference;
		if(r != -1)
			gpuRef = P.metricList[r].gpuMetric;
	}

	//grab the spectrum from the data set
	dim3 dimBlock(BLOCK_SIZE, 1);
	dim3 dimGrid(P.dim.z / dimBlock.x + 1, 1);
	kernelGetSpectrum<<<dimGrid, dimBlock>>>(gpuSpectrum, P.gpuData, P.currentX, P.currentY,
								P.dim.x, P.dim.y, P.dim.z,
								gpuBasePts, nBasePts, gpuRef);

	//copy the spectrum to the CPU
	cudaMemcpy(cpuSpectrum, gpuSpectrum, sizeof(_precision) * P.dim.z, cudaMemcpyDeviceToHost);
	cudaFree(gpuSpectrum);
	cudaFree(gpuBasePts);
}
