 #ifndef QT_INSPECTION_WINDOW_H
 #define QT_INSPECTION_WINDOW_H

#include "GL/glew.h"
#include <QtOpenGL/QGLWidget>
#include <QMouseEvent>

#include "parameters.h"
extern parameterStruct P;
#include "GL/gl.h"

void UpdateWindows();

 class qtSpectralWindow : public QGLWidget
 {
     Q_OBJECT

 private:
	//previous mouse position for camera movement
	QPoint prevMouse;
	int dxMouse;
	int dyMouse;
	void drawHUD();
	void drawSpectrum();
	void drawHistogram();


 public:



	qtSpectralWindow(QWidget *parent = 0);
	~qtSpectralWindow();

	QSize minimumSizeHint() const;
	QSize sizeHint() const;

	void mouseAction(QMouseEvent *event);

 public slots:


 signals:


 protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void wheelEvent(QWheelEvent *event);

 private:

 };

 #endif
