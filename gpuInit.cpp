#include "qtSpatialWindow.h"
#include "qtSpectralWindow.h"

#ifdef WIN32
#include <Windows.h>
#endif

#include <GL/glew.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cudaHandleError.h>
#include <cuda_gl_interop.h>
#include "CHECK_OPENGL_ERROR.h"

extern qtSpatialWindow* gpSpatialWindow;
extern qtSpectralWindow* gpSpectralWindow;

void gpuInitGLEW()
{
	GLenum err = glewInit();
	if(GLEW_OK != err)
	{
		printf("Error starting GLEW.");
	}
	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
	if(glewIsSupported("GL_VERSION_3_0"))
		printf("OpenGL 3.0 supported\n");
	else
		printf("OpenGL 3.0 is NOT supported\n");
}

void gpuQueryMemory(unsigned int &availableMem, unsigned int &totalMem)
{
    size_t free;
    size_t total;

    cudaMemGetInfo(&free, &total);


    totalMem = total/1048576;
    availableMem = free/1048576;
}

void gpuSetCUDADevice()
{
	cudaDeviceProp prop;
	int dev;

	//find a CUDA device that can handle an offscreen buffer
	int num_gpu;
	HANDLE_ERROR(cudaGetDeviceCount(&num_gpu));
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major=1;
	prop.minor=3;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));
	HANDLE_ERROR(cudaGLSetGLDevice(dev));
}

void gpuCreateRenderBuffer(GLuint &buffer, cudaGraphicsResource_t &resource, int width, int height)
{
	//This function creates a texture map for rendering to the spatial window
	gpSpatialWindow->makeCurrent();

	//delete any previous texture
	if(buffer != 0)
		glDeleteBuffers(1, &buffer);

	//create a new buffer
	unsigned int size = width * height * BUFFER_PIXEL_SIZE;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW);

	//unbind the buffer so we don't mess stuff up later
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	//register the buffer with CUDA
	cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsRegisterFlagsNone);
}

void gpuCreateHistogram(histoPrecision** gpuHistogram, GLuint &buffer, cudaGraphicsResource_t &resource, int bins, int bands)
{
    //this function allocates space for the histogram, including a color buffer for display
    //called any time the number of bins is changed

    //activate the context for the spectral window
    gpSpectralWindow->makeCurrent();

    //delete any previous histogram
    if(*gpuHistogram != NULL)
        cudaFree(*gpuHistogram);
    //delete any previous pixel buffer
    if(buffer != 0)
        glDeleteBuffers(1, &buffer);

    //create a new histogram
    unsigned int sizeHist = sizeof(unsigned int) * bins * bands;
    cudaMalloc(gpuHistogram, sizeHist);

    //create a new pixel buffer
    unsigned int sizeBuff = bins * bands * BUFFER_PIXEL_SIZE;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sizeBuff, NULL, GL_DYNAMIC_DRAW);
	CHECK_OPENGL_ERROR

    //unbind the buffer so we don't mess stuff up later
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    //register the buffer with CUDA
    cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsRegisterFlagsNone);

}

void gpuUploadData(_precision** gpuData, _precision* cpuData, unsigned int samples, unsigned int lines, unsigned int bands)
{
	unsigned int size = sizeof(_precision)*samples*lines*bands;
	HANDLE_ERROR(cudaMalloc(gpuData, size));
	HANDLE_ERROR(cudaMemcpy(*gpuData, cpuData, size, cudaMemcpyHostToDevice));

	/*
	//upload the data to a CUDA array

	//create a channel descriptor (one 32-bit floating point channel)
	cudaChannelFormatDesc channelFormat;
	channelFormat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	//create an extend definining the size of the data set
	cudaExtent dims;
	dims = make_cudaExtent(samples, lines, bands);

	//allocate memory on the GPU
	HANDLE_ERROR(cudaMalloc3DArray(&P.gpuDataArray, &channelFormat, dims));

	//create a pitched pointer to host memory
	cudaPitchedPtr hostPtr;
	hostPtr = make_cudaPitchedPtr((void*)cpuData, samples*sizeof(precision), samples, lines);

	//create a copy parameter structure
	cudaMemcpy3DParms copyParms = {0};
	copyParms.srcPtr = hostPtr;
	copyParms.dstArray = P.gpuDataArray;
	copyParms.extent = dims;
	copyParms.kind = cudaMemcpyHostToDevice;

	//perform the copy
	HANDLE_ERROR(cudaMemcpy3D(&copyParms));

	//set the parameters for the texture structure
	texDataSet.normalized = false;
	texDataSet.filterMode = cudaFilterModePoint;
	texDataSet.addressMode[0] = cudaAddressModeClamp;
	texDataSet.addressMode[1] = cudaAddressModeClamp;

	//bind the texture to the array
	HANDLE_ERROR( cudaBindTextureToArray(texDataSet, P.gpuDataArray, channelFormat) );
    */
}
