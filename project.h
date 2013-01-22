#ifndef PROJECT_H
#define PROJECT_H
#include <string>
using namespace std;
enum loadStatus {loadStatusOK, loadStatusInvalidProject, loadStatusInvalidData};
loadStatus LoadProject(string filename);
void SaveProject(string filename);

#endif
