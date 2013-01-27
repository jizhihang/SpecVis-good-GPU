#ifdef WIN32
#include <Windows.h>
#endif
#include <GL/gl.h>

void gpuInitGLEW();
void gpuQueryMemory(unsigned int &availableMem, unsigned int &totalMem);
void gpuSetCUDADevice();
void gpuCreateRenderBuffer(GLuint &buffer, cudaGraphicsResource_t &resource, int width, int height);
void gpuUploadData(_precision** gpuData, _precision* cpuData, unsigned int samples, unsigned int lines, unsigned int bands);
void gpuCreateHistogram(histoPrecision** gpuHistogram, GLuint &buffer, cudaGraphicsResource_t &resource, int bins, int bands);
