#ifndef TRANSFERFUNC_H
#define TRANSFERFUNC_H

enum transferFuncType {tfConstant, tfLinearUp, tfLinearDown, tfGaussian};

struct transferFuncStruct{

	string name;
	transferFuncType type;
	float r, g, b;
	unsigned int sourceMetric;
	_precision tfMin;
	_precision tfMax;

	transferFuncStruct()
	{
		r = 255.0;
		g = 0.0;
		b = 0.0;

		tfMin = 0;
		tfMax = 1;
		sourceMetric = 0;

		type = tfConstant;
	}


};


#endif