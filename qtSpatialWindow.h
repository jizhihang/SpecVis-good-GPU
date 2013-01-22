 #ifndef QT_FARFIELD_WINDOW_H
 #define QT_FARFIELD_WINDOW_H

#include "GL/glew.h"

#include <QtOpenGL/QGLWidget>
#include <QMouseEvent>

#include "rtsCamera.h"

#include "GL/gl.h"

#include "parameters.h"
extern parameterStruct P;

void UpdateWindows();

 class qtSpatialWindow : public QGLWidget
 {
     Q_OBJECT

 private:
    //previous mouse position for camera movement
	QPoint prevMouse;
	void renderBuffer();
	void drawHUD();

 public:
     qtSpatialWindow(QWidget *parent = 0);
     ~qtSpatialWindow();

     QSize minimumSizeHint() const;
     QSize sizeHint() const;



 public slots:


 signals:


 protected:
     void initializeGL();
     void paintGL();
     void resizeGL(int width, int height);
     void mousePressEvent(QMouseEvent *event);
     void mouseMoveEvent(QMouseEvent *event);
     void wheelEvent(QWheelEvent *event);
     void mouseAction(QMouseEvent *event);

 private:

 };

 #endif
