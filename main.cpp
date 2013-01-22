#include "GL/glew.h"

#include <QApplication>
#include "qtUI.h"
#include "qtSpatialWindow.h"
#include "qtSpectralWindow.h"

#include "parameters.h"
#include "rtsGUIConsole.h"

#include "gpuInit.h"
#include "project.h"

parameterStruct P;
QtUI* gpUIWindow;
qtSpatialWindow* gpSpatialWindow;
qtSpectralWindow* gpSpectralWindow;

void UpdateSpatialWindow()
{
	gpSpatialWindow->updateGL();
}

void UpdateSpectralWindow()
{
	gpSpectralWindow->updateGL();
}

void UpdateWindows()
{
	UpdateSpatialWindow();
	UpdateSpectralWindow();
}

void UpdateUI()
{
	gpUIWindow->UpdateUI();
}

void positionWindows()
{
    //get the size and position information from the UI window
    int px = gpUIWindow->x();
    int py = gpUIWindow->y();
	QRect uiFrame = gpUIWindow->frameGeometry();
	QRect uiNoFrame = gpUIWindow->geometry();

	//compute the frame dimensions (this is behaving unexpectedly in Linux)
	int frameHeight = uiFrame.height() - uiNoFrame.height();
	int frameWidth = uiFrame.width() - uiNoFrame.width();

    //compute the desired size of the spatial window
	int spatialWinHeight = uiFrame.height() - frameHeight;
	int spatialWinWidth = spatialWinHeight;

	if(P.dim.x > P.dim.y){
		float aspect = (float)P.dim.y / (float)P.dim.x;
		spatialWinHeight = aspect * spatialWinHeight;
	}
	if(P.dim.y > P.dim.x){
	float aspect = (float)P.dim.x / (float)P.dim.y;
		spatialWinWidth = aspect * spatialWinWidth;
	}

	gpSpatialWindow->resize(spatialWinWidth, spatialWinHeight);
	gpSpatialWindow->move(px + uiFrame.width(), py);
	gpSpatialWindow->show();


	gpSpectralWindow->resize(uiFrame.width() + spatialWinWidth, uiFrame.height()/2);
	gpSpectralWindow->move(px, py + uiFrame.height());
	gpSpectralWindow->show();
}



int main(int argc, char *argv[])
{
	//create a console for debugging
	RedirectIOToConsole();

	//start the Qt application (also initializes opengl)
	QApplication a(argc, argv);

	gpUIWindow = new QtUI();
	gpUIWindow->move(0, 0);
	gpUIWindow->show();

	//create the spatial and spectral display windows (not displayed at startup)
	gpSpectralWindow = new qtSpectralWindow();
    gpSpatialWindow = new qtSpatialWindow();

	//initialize OpenGL and CUDA interoperability in the spatial window
	gpSpatialWindow->makeCurrent();
	gpuSetCUDADevice();
	gpuInitGLEW();
	//gpuCreateRenderBuffer(P.gpu_glTexture, P.dim.x, P.dim.y);

	//update the UI with the default parameters
	gpUIWindow->UpdateUI();



	a.exec();
}
