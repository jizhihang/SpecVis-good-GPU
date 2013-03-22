#include "project.h"
#include "parameters.h"
#include "gpuInit.h"

#include "rtsENVI.h"

#include <fstream>
#include <stdio.h>
#include <string.h>

extern parameterStruct P;

void gpuComputeMetric(unsigned int m);

void LoadData(string filename)
{
    enviHeaderStruct header;
    if(!enviLoadf(header, &P.cpuData, filename))
    {
	    cout<<"Error loading ENVI file."<<endl;
	    return;
    }
    P.dim = vector3D<unsigned int>(header.samples, header.lines, header.bands);
    P.filename = filename;
    P.currentX = P.dim.x/2;
    P.currentY = P.dim.y/2;

    gpuUploadData(&P.gpuData, P.cpuData, P.dim.x, P.dim.y, P.dim.z);

    //create a buffer for the spatial window
    gpuCreateRenderBuffer(P.gpu_glBuffer, P.gpu_cudaResource, P.dim.x, P.dim.y);
}

loadStatus LoadProject(string filename)
{
    //open the project file as text
    ifstream infile(filename.c_str());

    string line, token;
    string strData;
    int intData;
    int m = -1; //metric counter
	int t = -1;	//tf counter

    //make sure that this is a valid project file
    char prjTest[8];
    infile.getline(prjTest, 8);
    cout<<prjTest<<endl;
    if(strcmp(prjTest, "SPECVIS") != 0)
        return loadStatusInvalidProject;

	enum inputType {typeMetric, typeTF};
	inputType inType;

    while(!infile.eof())
    {
        //read a line from the project file
        getline(infile, line);

        //if the line is a comment, continue on to the next one
        if(line[0] == '#') continue;

        //create another stream to perform conversion between data types
        stringstream convert(line);
        //grab the line token
        convert>>token;

        //if the line specifies the data file, load it
        if(token.find("data") != string::npos){
            convert>>strData;
            LoadData(strData);
        }
        //if the line specifies a metric
        if(token.find("metric") != string::npos){
			inType = typeMetric;
            //create a new metric
            metricStruct newMetric;
            //get the metric type
            convert>>strData;
            if(token.find("mean") != string::npos)
                newMetric.type = metricMean;
            if(token.find("centroid") != string::npos)
                newMetric.type = metricCentroid;
            //get the metric band and bandwidth
            convert>>newMetric.band;
            convert>>newMetric.bandwidth;
            P.metricList.push_back(newMetric);
            m++;    //increment the metric counter
        }
		 //if the line specifies a transfer function
        if(token.find("transferfunc") != string::npos){
			inType = typeTF;
            //create a new metric
            transferFuncStruct newTF;
            //get the metric type
            convert>>strData;
            if(token.find("constant") != string::npos)
                newTF.type = tfConstant;
            if(token.find("gaussian") != string::npos)
                newTF.type = tfGaussian;
			if(token.find("linearup") != string::npos)
                newTF.type = tfLinearUp;
            if(token.find("lineardown") != string::npos)
                newTF.type = tfLinearDown;
            //get the metric band and bandwidth
            convert>>newTF.tfMin;
            convert>>newTF.tfMax;
			convert>>newTF.sourceMetric;
			convert>>newTF.r;
			convert>>newTF.g;
			convert>>newTF.b;
            P.tfList.push_back(newTF);
			P.selectedTF = 0;
            t++;    //increment the metric counter
        }
        //add a name to the current metric
        if(token.find("name") != string::npos)
		{
			if(inType == typeMetric)
				convert>>P.metricList[m].name;
			else
				convert>>P.tfList[t].name;
		}

        //add baseline points
        if(token.find("baseline") != string::npos)
            while(!convert.eof())
            {
                convert>>intData;
                P.metricList[m].baselinePoints.push_back(intData);
            }
        //get a reference parameter
        if(token.find("reference") != string::npos)
		{
            convert>>P.metricList[m].reference;
			convert>>P.metricList[m].refEpsilon;
		}

        token = "";
    }

    //compute every metric so that you can work with TFs and referenced metrics without problems
	for(unsigned int m = 0; m<P.metricList.size(); m++)
		gpuComputeMetric(m);

    return loadStatusOK;


}

void SaveProject(string filename)
{
    //create a project file
    ofstream outfile(filename.c_str());

    //output the verification string indicating that the file is a SpecVis project
    outfile<<"SPECVIS"<<endl;

    //output the data filename
    outfile<<"#Data file"<<endl;
    outfile<<"data "<<P.filename;

    //output each metric
    unsigned int nMetrics = P.metricList.size();
    for(unsigned int m=0; m<nMetrics; m++){
        //output the token
        outfile<<endl<<"metric ";
        //output the metric type
        if(P.metricList[m].type == metricMean)
            outfile<<"mean ";
        if(P.metricList[m].type == metricCentroid)
            outfile<<"centroid ";
        //output the band and bandwidth
        outfile<<P.metricList[m].band<<" "<<P.metricList[m].bandwidth;

        //output the metric name
        if(P.metricList[m].name.length())
            outfile<<endl<<"     name "<<P.metricList[m].name;

        //output any metric baseline points
        unsigned int nBasePts = P.metricList[m].baselinePoints.size();
        if(nBasePts > 0)
            outfile<<endl<<"     baseline";
        for(unsigned int b=0; b<nBasePts; b++)
            outfile<<" "<<P.metricList[m].baselinePoints[b];

        //output any reference metric
        if(P.metricList[m].reference > -1)
            outfile<<endl<<"     reference "<<P.metricList[m].reference<<" "<<P.metricList[m].refEpsilon;

    }
    
    //output each transfer function
    unsigned int nTF = P.tfList.size();
    for(unsigned int tf=0; tf<nTF; tf++){
        //output the token
        outfile<<endl<<"transferfunc ";
        //output the metric type
        if(P.tfList[tf].type == tfConstant)
            outfile<<"constant ";
        if(P.tfList[tf].type == tfGaussian)
            outfile<<"gaussian ";
	   if(P.tfList[tf].type == tfLinearDown)
            outfile<<"lineardown ";
	   if(P.tfList[tf].type == tfLinearUp)
            outfile<<"linearup ";
        //output the band and bandwidth
        outfile<<P.tfList[tf].tfMin<<" "<<P.tfList[tf].tfMax<<" "<<P.tfList[tf].sourceMetric<<" "<<P.tfList[tf].r<<" "<<P.tfList[tf].g<<" "<<P.tfList[tf].b;

        //output the metric name
        if(P.metricList[tf].name.length())
            outfile<<endl<<"     name "<<P.tfList[tf].name;

    }

    outfile.close();

}
