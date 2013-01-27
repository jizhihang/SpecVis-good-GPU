#include "qtSpectralWindow.h"
#include "rts_glBrewer.h"
#include "CHECK_OPENGL_ERROR.h"

void gpuGetSpectrum(_precision* cpuSpectrum);
void gpuComputeHistogram();
void gpuHistogramToBuffer();

//constructor
qtSpectralWindow::qtSpectralWindow(QWidget *parent)
    : QGLWidget(parent)
{


}
//destructor
qtSpectralWindow::~qtSpectralWindow()
{
	makeCurrent();


}

QSize qtSpectralWindow::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize qtSpectralWindow::sizeHint() const
{
    return QSize(400, 400);
}

void qtSpectralWindow::initializeGL()
{


    makeCurrent();
	glClearColor(0.0, 0.0, 0.0, 0.0);


}

void qtSpectralWindow::drawHUD()
{
	//display the picked band
	int pickedBand = 0;
	if(P.displayMode == displayRaw)
		pickedBand = P.currentBand;
	else if(P.displayMode == displayMetrics || P.displayMode == displayTF)
		pickedBand = P.metricList[P.selectedMetric].band;

	//draw the metric bandwidth (if rendering the metric)
	if(P.displayMode == displayMetrics)
	{
		int m = P.selectedMetric;
		int lowBand = P.metricList[m].band - P.metricList[m].bandwidth/2;
		int highBand = lowBand + P.metricList[m].bandwidth;

		glColor3f(0.1f, 0, 0.1f);
		glBegin(GL_QUADS);
			glVertex2f(lowBand, P.spectralMin);
			glVertex2f(lowBand, P.spectralMax);
			glVertex2f(highBand, P.spectralMax);
			glVertex2f(highBand, P.spectralMin);
		glEnd();
	}

	//draw the selection region and band line (if rendering transfer functions)
	if(P.displayMode == displayTF)
	{
		int tf = P.selectedTF;
		int m = P.tfList[tf].sourceMetric;
		int lowBand = P.metricList[m].band - P.metricList[m].bandwidth/2;
		int highBand = lowBand + P.metricList[m].bandwidth;
		_precision lowVal = P.tfList[tf].tfMin;
		_precision highVal = P.tfList[tf].tfMax;
		_precision midVal = (highVal - lowVal)/2 + lowVal;

		//draw two boxes, one along the high range and one along the low range
		
		glBegin(GL_QUADS);
			glColor3f(0.2f, 0, 0);
			glVertex2f(lowBand, midVal);
			glVertex2f(lowBand, highVal);
			glVertex2f(highBand, highVal);
			glVertex2f(highBand, midVal);

			glColor3f(0, 0, 0.2f);
			glVertex2f(lowBand, lowVal);
			glVertex2f(lowBand, midVal);
			glVertex2f(highBand, midVal);
			glVertex2f(highBand, lowVal);
		glEnd();

		//draw band lines to indicate source metric position		
		glBegin(GL_LINES);
			glColor3f(1, 0, 0);
			glVertex2f(P.metricList[m].band, midVal);
			glVertex2f(P.metricList[m].band, highVal);
			glColor3f(0, 0, 1);
			glVertex2f(P.metricList[m].band, midVal);
			glVertex2f(P.metricList[m].band, lowVal);
		glEnd();
	}
	//draw the band line
	if(P.displayMode == displayRaw || P.displayMode == displayMetrics)
	{
		glColor3f(1, 0, 1);
		glBegin(GL_LINES);
			glVertex2f(pickedBand, P.spectralMin);
			glVertex2f(pickedBand, P.spectralMax);
		glEnd();
	}

		

	//draw the intensity extrema
	_precision scaleMin, scaleMax;
	if(P.displayMode == displayRaw)
	{
		scaleMin = P.scaleMin;
		scaleMax = P.scaleMax;
	}
	else if(P.displayMode == displayMetrics)
	{
		scaleMin = P.metricList[P.selectedMetric].scaleLow;
		scaleMax = P.metricList[P.selectedMetric].scaleHigh;
	}
	if(P.displayMode == displayRaw || P.displayMode == displayMetrics)
	{
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
			glVertex2f(0, scaleMax);
			glVertex2f(P.dim.z, scaleMax);
		glEnd();

		glColor3f(0.0, 0.0, 1.0);
		glBegin(GL_LINES);
			glVertex2f(0, scaleMin);
			glVertex2f(P.dim.z, scaleMin);
		glEnd();
	}

	//draw baseline points
	if(P.displayMode == displayMetrics || P.displayMode == displayTF)
	{
		//select the metric used to display baseline points
		int m;
		if(P.displayMode == displayMetrics)
			m = P.selectedMetric;
		else if(P.displayMode == displayTF)
			m = P.tfList[P.selectedTF].sourceMetric;

		//exit if the current metric has no baseline points
		if(P.metricList[m].baselinePoints.size() == 0)
			return;

		float winHeight = P.spectralMax - P.spectralMin;
		float winWidth = P.dim.z;
		int nBasePts = P.metricList[m].baselinePoints.size();
			
		glColor3f(0.0, 1.0, 0.0);
		glBegin(GL_LINES);
		for(int b=0; b<nBasePts; b++)
		{
			glVertex2f(P.metricList[m].baselinePoints[b], -0.1 * winHeight);
			glVertex2f(P.metricList[m].baselinePoints[b], 0.1 * winHeight);
		}
		glEnd();

		//draw the selected baseline point with triangles
		int b = P.selectedBaselinePoint;
		glBegin(GL_TRIANGLES);
			glVertex2f(P.metricList[m].baselinePoints[b], 0.05 * winHeight);
			glVertex2f(P.metricList[m].baselinePoints[b] + 0.005 * winWidth, 0.1 * winHeight);
			glVertex2f(P.metricList[m].baselinePoints[b] - 0.005 * winWidth, 0.1 * winHeight);

			glVertex2f(P.metricList[m].baselinePoints[b], -0.05 * winHeight);
			glVertex2f(P.metricList[m].baselinePoints[b] + 0.005 * winWidth, -0.1 * winHeight);
			glVertex2f(P.metricList[m].baselinePoints[b] - 0.005 * winWidth, -0.1 * winHeight);
		glEnd();

	}
}

void qtSpectralWindow::drawSpectrum()
{
	//allocate space for the spectral data
	_precision* cpuSpectrum;
	cpuSpectrum = (_precision*)malloc(sizeof(_precision) * P.dim.z);

	//get the spectrum from the GPU-based data set
	gpuGetSpectrum(cpuSpectrum);

	//draw the spectrum to the window
	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_LINE_STRIP);
	for(unsigned int b=0; b<P.dim.z; b++)
		glVertex2f(b, cpuSpectrum[b]);
	glEnd();
}

void qtSpectralWindow::drawHistogram()
{
	//compute the histogram
	gpuComputeHistogram();
	gpuHistogramToBuffer();

    //create a texture map from the histogram pixel buffer
	glEnable(GL_TEXTURE_2D);
	GLuint texRender;
	glGenTextures(1, &texRender);
	glBindTexture(GL_TEXTURE_2D, texRender);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	//bind the histogram buffer and create a texture
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, P.gpu_glHistBuffer);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);		//important for handling textures of different sizes
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, P.dim.z, P.histBins, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);


    //draw a quad
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0);		glVertex2f(0, P.spectralMin);
        glTexCoord2f(1, 0);		glVertex2f(P.dim.z, P.spectralMin);
        glTexCoord2f(1, 1);		glVertex2f(P.dim.z, P.spectralMax);
        glTexCoord2f(0, 1);		glVertex2f(0, P.spectralMax);
    glEnd();

	//disable texture mapping and delete the created texture
    glDisable(GL_TEXTURE_2D);
	glDeleteTextures(1, &texRender);

	CHECK_OPENGL_ERROR
}

void qtSpectralWindow::paintGL()
{
	//cout<<"Inspection refresh"<<endl;
	CHECK_OPENGL_ERROR
	makeCurrent();

	//specify the viewport parameters based on the specified spectral amplitude
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, P.dim.z, P.spectralMin, P.spectralMax);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawHUD();

	if(P.spectrumMode == spectrum)
        drawSpectrum();
    else if(P.spectrumMode == histogram)
        drawHistogram();

	
	CHECK_OPENGL_ERROR



}

void qtSpectralWindow::resizeGL(int width, int height)
{
	makeCurrent();
	glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
}

void qtSpectralWindow::mouseAction(QMouseEvent *event)
{

    if(event->buttons() == Qt::LeftButton)
	{

        //if SHIFT is pressed, modify the intensity bounds
        if(event->modifiers().testFlag(Qt::ShiftModifier))
        {
            //get the intensity value referenced by the current mouse pointer
            float a = ( height() - (float)event->pos().y() ) / (float)height();
            float pI = P.spectralMin + a * (P.spectralMax - P.spectralMin);

            //set the intensity stretching mode to manual
            P.scaleMode = manual;

            //set the maximum intensity
			if(P.displayMode == displayRaw)
				P.scaleMax = pI;
			else if(P.displayMode == displayMetrics || P.displayMode == displayTF)
				P.metricList[P.selectedMetric].scaleHigh = pI;
        }
        //otherwise select a new band
        else
        {
            //calculate the band clicked by the user
            float a = (float)event->pos().x() / (float)width();
			if(a < 0) a = 0;

			unsigned int pickedBand;
			pickedBand = (int)( a * P.dim.z );
            if(pickedBand >= P.dim.z) pickedBand = P.dim.z - 1;
            

			if(P.displayMode == displayRaw)
				P.currentBand = pickedBand;
			else if(P.displayMode == displayMetrics)
			{
				int m = P.selectedMetric;
				P.metricList[m].band = pickedBand;
			}
			else if(P.displayMode == displayTF)
			{
				//set the band value for the source metric
				int tf = P.selectedTF;
				int m = P.tfList[tf].sourceMetric;
				P.metricList[m].band = pickedBand;

				//set the amplitude range for the transfer function
				float b = ((float)height() - (float)event->pos().y()) / (float)height();
				//if(b < 0) b = 0;
				_precision pickedAmplitude;
				pickedAmplitude = b * (P.spectralMax - P.spectralMin) + P.spectralMin;
				_precision ampRange = (P.tfList[tf].tfMax - P.tfList[tf].tfMin)/2;
				P.tfList[tf].tfMin = pickedAmplitude - ampRange;
				P.tfList[tf].tfMax = pickedAmplitude + ampRange;
			}
		}
		//update the visualization
		UpdateWindows();
		UpdateUI();

	}
	else if(event->buttons() == Qt::RightButton)
	{
        if(event->modifiers().testFlag(Qt::ShiftModifier))
        {
            //get the intensity value referenced by the current mouse pointer
            float a = ( height() - (float)event->pos().y() ) / (float)height();
            float pI = P.spectralMin + a * (P.spectralMax - P.spectralMin);

            //set the intensity stretching mode to manual
            P.scaleMode = manual;

            //set the minimum intensity value
            if(P.displayMode == displayRaw)
				P.scaleMin = pI;
			else if(P.displayMode == displayMetrics || P.displayMode == displayTF)
				P.metricList[P.selectedMetric].scaleLow = pI;

        }
		else
		{
			//calculate the band clicked by the user
            float a = (float)event->pos().x() / (float)width();
			if(a < 0) a = 0;

			unsigned int pickedBand;
			pickedBand = (int)( a * P.dim.z );
            if(pickedBand >= P.dim.z) pickedBand = P.dim.z - 1;

			int m = P.selectedMetric;
			int b = P.selectedBaselinePoint;
			P.metricList[m].baselinePoints[b] = pickedBand;
		}

        //update the windows and UI
        UpdateWindows();
        UpdateUI();

	}
	if(event->buttons() == Qt::MiddleButton)
	{
        float dy = -( 1.0/(float)height() ) * (P.spectralMax - P.spectralMin);
        P.spectralMax += dyMouse * dy;
        P.spectralMin += dyMouse * dy;
        UpdateWindows();
        UpdateUI();
	}
}

void qtSpectralWindow::mousePressEvent(QMouseEvent *event)
{
	//get the current mouse position
	prevMouse = event->pos();
	dxMouse = 0;
	dyMouse = 0;

    //call the general mouse event handler
    mouseAction(event);

}

void qtSpectralWindow::mouseMoveEvent(QMouseEvent *event)
{
	//find the change in mouse position
	dxMouse = prevMouse.x() - event->pos().x();
	dyMouse = prevMouse.y() - event->pos().y();
	prevMouse = event->pos();

	//call the general mouse event handler
	mouseAction(event);


}
void qtSpectralWindow::wheelEvent(QWheelEvent *event)
{
	float dz = 0.1f;

	if(event->modifiers().testFlag(Qt::ShiftModifier) &&
	   P.displayMode == displayMetrics)
	{
		int m = P.selectedMetric;
		if(event->delta() < 0)
		{
			P.metricList[m].bandwidth--;
		}
		else
		{
			P.metricList[m].bandwidth++;
		}
		UpdateWindows();
	}
	else if(event->modifiers().testFlag(Qt::ShiftModifier) &&
	   P.displayMode == displayTF)
	{
		int tf = P.selectedTF;
		_precision ampStep = 0.01 * (P.spectralMax - P.spectralMin);
		if(event->delta() < 0)
		{
			P.tfList[tf].tfMin += ampStep;
			P.tfList[tf].tfMax -= ampStep;
		}
		else
		{
			P.tfList[tf].tfMin -= ampStep;
			P.tfList[tf].tfMax += ampStep;
		}
		UpdateWindows();
	}
	else
	{
		if(event->delta() < 0)
		{
			P.spectralMin -= dz;
			P.spectralMax += dz;
		}
		else
		{
			P.spectralMin += dz;
			P.spectralMax -= dz;
		}
		UpdateSpectralWindow();
	}

	
	UpdateUI();
}
