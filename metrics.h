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
		precision scaleLow;
		precision scaleHigh;

        //id of the reference metric
        int reference;

        //gpu data
        precision* gpuMetric;

        metricStruct()
        {
            name = "metric";
            type = metricMean;
            band = 0;
            bandwidth = 1;
            reference = -1;

			scaleLow = 0;
			scaleHigh = 1;

            gpuMetric = NULL;

        }

};

#endif
