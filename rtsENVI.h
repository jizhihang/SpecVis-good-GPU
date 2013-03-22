//This file contains functions for file IO and management of ENVI files

#ifndef RTS_ENVI_H
#define RTS_ENVI_H

#include<fstream>
#include<string>
#include<sstream>
#include<sys/stat.h>

using namespace std;

enum InterleaveType {bsq, bip, bil};

struct enviHeaderStruct{
    unsigned int samples;
    unsigned int lines;
    unsigned int bands;
    unsigned int offset;
    unsigned int datatype;
    InterleaveType interleave;
    
    enviHeaderStruct()
    {
	    samples = 0;
	    lines = 0;
	    bands = 0;
	    offset = 0;
	    datatype = 0;
    }
};

static void enviPrintHeader(enviHeaderStruct h)
{
    cout<<"ENVI Header data***************"<<endl;
    cout<<"Samples: "<<h.samples<<endl;
    cout<<"Lines: "<<h.lines<<endl;
    cout<<"Bands: "<<h.bands<<endl;
    cout<<"Offset: "<<h.offset<<endl;
    cout<<"Data Type: "<<h.datatype<<endl;
    cout<<"Interleave: "<<h.interleave<<endl;
    cout<<"*******************************"<<endl;

}

static bool enviLoadHeader(enviHeaderStruct &header, string filename)
{
    //open the header file for reading
    ifstream infile(filename.c_str());
    if(!infile)
    {
	    cout<<"Error loading header file."<<endl;
	    return false;
    }

    //search for relavant tokens
    string test, tmp;
    char c;
    while(!infile.eof())
    {

        //get a string
        getline(infile, test);

        //find tokens and set the relevant values in the header structure
        stringstream convert(test);
        if(test.find("samples") != string::npos)
            convert>>tmp>>c>>header.samples;
        if(test.find("lines") != string::npos)
            convert>>tmp>>c>>header.lines;
        if(test.find("bands") != string::npos)
            convert>>tmp>>c>>header.bands;
        if(test.find("data type") != string::npos){
            convert>>tmp>>tmp>>c>>header.datatype;
            if(header.datatype != 4)
                cout<<"Error, we can only handle 32-bit floating point data."<<endl;
        }
        if(test.find("header offset") != string::npos)
            convert>>tmp>>tmp>>c>>header.offset;
        if(test.find("interleave") != string::npos)
        {
            convert>>tmp>>c>>tmp;
            if(tmp == "bsq") header.interleave = bsq;
            if(tmp == "bip") header.interleave = bip;
            if(tmp == "bil") header.interleave = bil;
            if(header.interleave != bsq)
                cout<<"Error, we can only handle BSQ interleaved files at this time."<<endl;
        }

    }

    //close the file
    infile.close();

    //return the header structure
    return true;

}


static bool enviLoadf(enviHeaderStruct &header, float** cpuPtr, string filename, string headername)
{
    if(!enviLoadHeader(header, headername))
	    return false;

    //check the file to make sure everything matches up
    struct stat filestats;
    if(stat(filename.c_str(), &filestats) != 0){
        cout<<"File not found..."<<endl; return false;
    }
    unsigned int dataSize = sizeof(float) * header.samples * header.lines * header.bands;
    int fileSize = dataSize + sizeof(float) * header.offset;

    //warn the user if the file size doesn't match up
    if(filestats.st_size < fileSize){
        cout<<"Error: the binary file is of insufficient size."<<endl; return false;}
    if(filestats.st_size > fileSize){
        cout<<"Warning: the binary file is larger than expected."<<endl;}

    //allocate memory on the CPU
    (*cpuPtr) = (float*)malloc(dataSize);

    //load the binary data
	cout<<filename<<endl;
    ifstream infile(filename.c_str(), ios::in | ios::binary);   //load the file
	if(!infile) cout<<"Error: there was a problem opening the file for reading."<<endl;

    infile.seekg(sizeof(float) * header.offset);            //seek to the data starting point
    infile.read((char*)(*cpuPtr), dataSize);                    //read the data
    infile.close();


    cout<<headername<<endl;
    enviPrintHeader(header);
    return true;
}

static bool enviLoadf(enviHeaderStruct &header, float** cpuPtr, string filename)
{
    //this function is a wrapper which assumes that the name of the ENVI header is the same as the file

    //guess the header file name (check both .hdr and .HDR)
    struct stat filestats;
    string headername = filename;
    headername += ".hdr";    
    if(stat(headername.c_str(), &filestats) != 0){
        headername = filename;
	   headername += ".HDR";
    }
    if(stat(headername.c_str(), &filestats) != 0)
    {
	    cout<<"Header file not found."<<endl;
	    return false;
    }

    return enviLoadf(header, cpuPtr, filename, headername);


}


#endif
