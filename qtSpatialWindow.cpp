#include "qtSpatialWindow.h"
#include "CHECK_OPENGL_ERROR.h"

void gpuRawToBuffer();
void gpuComputeMetric(unsigned int m);
void gpuMetricToBuffer(unsigned int m);

//constructor
qtSpatialWindow::qtSpatialWindow(QWidget *parent)
    : QGLWidget(parent)
{

}

//destructor
qtSpatialWindow::~qtSpatialWindow()
{
	makeCurrent();
}


QSize qtSpatialWindow::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize qtSpatialWindow::sizeHint() const
{
    return QSize(400, 400);
}

void qtSpatialWindow::initializeGL()
{
	makeCurrent();
	glClearColor(1.0, 0.0, 0.0, 0.0);
}
void qtSpatialWindow::renderBuffer()
{
	//This function renders the display buffer to the window
	CHECK_OPENGL_ERROR

	//no initial transformations
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//create a texture map from the spatial pixel buffer
	glEnable(GL_TEXTURE_2D);
	GLuint texRender;
	glGenTextures(1, &texRender);
	CHECK_OPENGL_ERROR
	glBindTexture(GL_TEXTURE_2D, texRender);
	CHECK_OPENGL_ERROR
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	CHECK_OPENGL_ERROR
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, P.gpu_glBuffer);
	CHECK_OPENGL_ERROR
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, P.dim.x, P.dim.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	CHECK_OPENGL_ERROR

	//draw the texture in the spatial window
	float px = P.dim.x;
	float py = P.dim.y;
	glColor3f(0.0, 1.0, 0.0);
	glBegin(GL_QUADS);
		glTexCoord2f(0, 0);
		glVertex2f(0, 0);

		glTexCoord2f(1, 0);
		glVertex2f(px, 0);

		glTexCoord2f(1, 1);
		glVertex2f(px, py);

		glTexCoord2f(0, 1);
		glVertex2f(0, py);
	glEnd();
	CHECK_OPENGL_ERROR



	glFlush();

	glDisable(GL_TEXTURE_2D);
	glDeleteTextures(1, &texRender);

}

void qtSpatialWindow::drawHUD()
{
    //draw the cross-hairs that indicate the currently selected position
	glColor3f(1.0, 0.0, 1.0);
	glBegin(GL_LINES);
		glVertex2f(0 + 0.5, P.currentY + 0.5);
		glVertex2f(P.dim.x + 0.5, P.currentY + 0.5);
	glEnd();
	glBegin(GL_LINES);
		glVertex2f(P.currentX + 0.5, 0 + 0.5);
		glVertex2f(P.currentX + 0.5, P.dim.y + 0.5);
	glEnd();

	//draw the bounding volume for the histogram window
	if(P.spectrumMode == histogram)
	{
		glColor3f(0.0, 1.0, 0.0);
		float minX = P.currentX - (int)P.histWindow/2 + 0.5;
		float maxX = minX + P.histWindow;
		float minY = P.currentY - (int)P.histWindow/2 + 0.5;
		float maxY = minY + P.histWindow;
		glBegin(GL_LINE_STRIP);
			glVertex2f(minX, minY);
			glVertex2f(minX, maxY);
			glVertex2f(maxX, maxY);
			glVertex2f(maxX, minY);
			glVertex2f(minX, minY);
		glEnd();
	}


}
void qtSpatialWindow::paintGL()
{
	//activate the OpenGL context for this (spatial) window
	CHECK_OPENGL_ERROR
	makeCurrent();
    CHECK_OPENGL_ERROR

	if(P.displayMode == displayRaw)
		gpuRawToBuffer();
	if(P.displayMode == displayMetrics)
	{
		gpuComputeMetric(P.selectedMetric);
		gpuMetricToBuffer(P.selectedMetric);
	}

	renderBuffer();

	drawHUD();
}

void qtSpatialWindow::resizeGL(int width, int height)
{
	makeCurrent();
	glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

	//this window will be in the image domain (in pixels)
	gluOrtho2D(0, P.dim.x, 0, P.dim.y);

    glMatrixMode(GL_MODELVIEW);
}

void qtSpatialWindow::mouseAction(QMouseEvent *event)
{
    if(event->buttons() == Qt::LeftButton)
	{
		float a = (float)(event->pos().x()) / (float)width();
		P.currentX = clamp(a * P.dim.x, 0, P.dim.x-1);
		float b = (float)(height() - event->pos().y()) / (float)height();
		P.currentY = clamp(b * P.dim.y, 0, P.dim.y-1);
		UpdateWindows();

	}
	else if(event->buttons() == Qt::RightButton)
	{

	}

}

void qtSpatialWindow::mousePressEvent(QMouseEvent *event)
{
	prevMouse = event->pos();

	mouseAction(event);
}

void qtSpatialWindow::mouseMoveEvent(QMouseEvent *event)
{
	//find the change in mouse position
	int dx = prevMouse.x() - event->pos().x();
	int dy = prevMouse.y() - event->pos().y();
	prevMouse = event->pos();

    //call the generic mouse controller
	mouseAction(event);
}

void qtSpatialWindow::wheelEvent(QWheelEvent *event)
{
	float dz = 0.1;
	if(event->delta() < 0)
	{
		P.histWindow -= P.histWindow * 0.25;
	}
	else
	{
		P.histWindow += 1 + P.histWindow * 0.25;
	}
	UpdateSpatialWindow();
	UpdateSpectralWindow();
	UpdateUI();
}
