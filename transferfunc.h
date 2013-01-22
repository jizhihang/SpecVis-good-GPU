#ifndef TRANSFERFUNC_H
#define TRANSFERFUNC_H

enum transferFuncType {tfConstant, tfLinearUp, tfLinearDown, tfGaussian};

struct transferFuncStruct{

	string name;
	transferFuncType type;
	float r, g, b;

	transferFuncStruct()
	{
		r = 0.0;
		g = 0.0;
		b = 0.0;
	}


};


#endif