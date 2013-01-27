#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <algorithm>
#include "datatypes.h"

struct metricStruct
{
        string name;
        metricType type;

        vector<unsigned int> baselinePoints;

        //metric position along the spectrum
        unsigned int band;
        //number of bands for metrics that incorporate several (mean, centroid)
        unsigned int bandwidth;

		//visualization
		_precision scaleLow;
		_precision scaleHigh;

        //id of the reference metric
        int reference;

        //minimum reference value that sets the normalized value to zero
        _precision refEpsilon;

        //gpu data
        _precision* gpuMetric;

        metricStruct()
        {
            name = "metric";
            type = metricMean;
            band = 0;
            bandwidth = 1;
            reference = -1;
            refEpsilon = (_precision)0.001;

			scaleLow = 0;
			scaleHigh = 1;

            gpuMetric = NULL;

        }

};

#endif
