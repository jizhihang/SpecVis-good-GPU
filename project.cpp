#include "project.h"
#include "parameters.h"
#include "gpuInit.h"

#include "rtsENVI.h"

#include <fstream>
#include <stdio.h>
#include <string.h>

extern parameterStruct P;

void LoadData(string filename)
{
    enviHeaderStruct header;
    header = enviLoadf(&P.cpuData, filename);
    P.dim = vector3D<unsigned int>(header.samples, header.lines, header.bands);
    P.filename = filename;
    P.currentX = P.dim.x/2;
    P.currentY = P.dim.y/2;

    //copy the data to the GPU
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
    char charData;
    int m = -1; //metric counter

    //make sure that this is a valid project file
    char prjTest[8];
    infile.getline(prjTest, 8);
    cout<<prjTest<<endl;
    if(strcmp(prjTest, "SPECVIS") != 0)
        return loadStatusInvalidProject;

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
        //add a name to the current metric
        if(token.find("name") != string::npos)
            convert>>P.metricList[m].name;
        //add baseline points
        if(token.find("baseline") != string::npos)
            while(!convert.eof())
            {
                convert>>intData;
                P.metricList[m].baselinePoints.push_back(intData);
            }
        //get a reference parameter
        if(token.find("reference") != string::npos)
            convert>>P.metricList[m].reference;

        token = "";
    }

    //char dataFile[256];
    //infile.getline(dataFile, 256);

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
    for(int m=0; m<nMetrics; m++){
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
        for(int b=0; b<nBasePts; b++)
            outfile<<" "<<P.metricList[m].baselinePoints[b];

        //output any reference metric
        if(P.metricList[m].reference > -1)
            outfile<<endl<<"     reference "<<P.metricList[m].reference;

    }

    outfile.close();

}
