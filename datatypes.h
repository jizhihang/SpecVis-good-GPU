#ifndef DATATYPES_H
#define DATATYPES_H

//specify the precision (currently 32-bit floating point)
typedef float precision;
typedef int histoPrecision;

#define BUFFER_PIXEL_SIZE		3

enum displayModeType {displayRaw, displayMetrics, displayTF};
enum scaleModeType {automatic, manual};
enum spectrumModeType {spectrum, histogram};
enum metricType {metricMean, metricCentroid};

#endif
