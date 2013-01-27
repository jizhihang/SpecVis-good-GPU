#include "ui_qtUI.h"
#include <iostream>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <qcolordialog.h>

#include "rtsENVI.h"
#include "parameters.h"
#include "gpuInit.h"
#include "project.h"

extern parameterStruct P;
using namespace std;

void positionWindows();
//void LoadProject(string filename);
//void SaveProject(string filename);

class QtUI : public QMainWindow
{
    Q_OBJECT
private:
    Ui::qtUIClass ui;
	bool updating;

public:
    QtUI(QWidget *parent = 0, Qt::WindowFlags flags = 0)
    {
        ui.setupUi(this);
		updating = true;

		//add the metric types to the type combo box
		ui.cmbMetricType->addItem("Mean", metricMean);
		ui.cmbMetricType->addItem("Centroid", metricCentroid);

		ui.cmbReferenceMetric->addItem("None", -1);

		//add transfer function types
		ui.cmbTransferFuncType->addItem("Constant", tfConstant);
		ui.cmbTransferFuncType->addItem("Linear - Up", tfLinearUp);
		ui.cmbTransferFuncType->addItem("Linear - Down", tfLinearDown);
		ui.cmbTransferFuncType->addItem("Gaussian", tfGaussian);

		updating = false;
    }
    ~QtUI()
	{

	};

	void closeEvent(QCloseEvent* event)
	{
		exit(0);
	}

	void UpdateMetricList()
	{
        updating = true;
        ui.cmbSelectedMetric->clear();
		ui.cmbReferenceMetric->clear();

		//add "None" to the reference list
		ui.cmbReferenceMetric->addItem("None", -1);

        //add each metric to the combo box
        unsigned int nMetrics = P.metricList.size();
        for(unsigned int m=0; m<nMetrics; m++)
		{
            ui.cmbSelectedMetric->addItem(P.metricList[m].name.c_str(), m);

			if(P.selectedMetric != m)
				ui.cmbReferenceMetric->addItem(P.metricList[m].name.c_str(), m);
		}

        //make the selected metric active
        if(nMetrics > 0)
            ui.cmbSelectedMetric->setCurrentIndex(P.selectedMetric);

        updating = false;
        UpdateUI();
	}

	void UpdateTFList()
	{
		updating = true;

		ui.cmbSelectedTF->clear();
		ui.cmbTFSourceMetric->clear();

		//add "Raw" to the TF metric list
		//ui.cmbTFSourceMetric->addItem("Raw", -1);

		//add each transfer function to the combo box
        unsigned int nTF = P.tfList.size();
        for(unsigned int tf=0; tf<nTF; tf++)
		{
            ui.cmbSelectedTF->addItem(P.tfList[tf].name.c_str(), tf);
		}

        //make the selected TF active
        if(nTF > 0)
            ui.cmbSelectedTF->setCurrentIndex(P.selectedTF);

		//add metrics to the source metric combo box
		unsigned int nMetrics = P.metricList.size();
		for(unsigned int m=0; m<nMetrics; m++)
			ui.cmbTFSourceMetric->addItem(P.metricList[m].name.c_str(), m);

		updating = false;
        UpdateUI();

	}

    void UpdateUI()
    {
		updating = true;
        //update the memory statistics
        gpuQueryMemory(P.gpuAvailableMemory, P.gpuTotalMemory);
        ui.txtTotalMemory->setText(QString::number(P.gpuTotalMemory));
        ui.txtAvailableMemory->setText(QString::number(P.gpuAvailableMemory));
        unsigned int memoryPercent = ( ( (float)P.gpuTotalMemory - (float)P.gpuAvailableMemory )/(float)P.gpuTotalMemory ) * 100;
        ui.barMemoryAvailable->setValue(memoryPercent);

        //update the file name and size
        ui.txtFileName->setText(QString(P.filename.c_str()));
        ui.txtSamples->setText(QString::number(P.dim.x));
        ui.txtLines->setText(QString::number(P.dim.y));
        ui.txtBands->setText(QString::number(P.dim.z));

        //enable relevant widgets
        //if there is a metric, enable metric widgets
        bool metricControls = false;
        if(P.metricList.size() != 0) metricControls = true;
        ui.btnRemoveMetric->setEnabled(metricControls);
        ui.btnDuplicateMetric->setEnabled(metricControls);
        ui.cmbMetricType->setEnabled(metricControls);
		ui.cmbSelectedMetric->setEnabled(metricControls);
        ui.btnRenameMetric->setEnabled(metricControls);
        ui.spinMetricBand->setEnabled(metricControls);
        ui.spinMetricBandwidth->setEnabled(metricControls);
        ui.cmbReferenceMetric->setEnabled(metricControls);
        ui.spinReferenceEpsilon->setEnabled(metricControls);
        ui.btnAddBaselinePoint->setEnabled(metricControls);
        ui.btnRemoveBaselinePoint->setEnabled(metricControls);
        ui.radDisplayMetrics->setEnabled(metricControls);
		ui.btnAddTF->setEnabled(metricControls);

        bool tfControls = false;
		if(P.tfList.size() != 0) tfControls = true;
		ui.cmbSelectedTF->setEnabled(tfControls);
		ui.cmbTransferFuncType->setEnabled(tfControls);
		ui.spinMaxTFValue->setEnabled(tfControls);
		ui.spinMinTFValue->setEnabled(tfControls);
		ui.chkDisplayTF->setEnabled(tfControls);
		ui.btnColorTF->setEnabled(tfControls);
		ui.btnRemoveTF->setEnabled(tfControls);
		ui.btnRenameTF->setEnabled(tfControls);
		ui.cmbTFSourceMetric->setEnabled(tfControls);


		if(P.displayMode == displayRaw)
		{
			ui.spinScaleMin->setValue(P.scaleMin);
			ui.spinScaleMax->setValue(P.scaleMax);
		}
		else if(P.displayMode == displayMetrics)
		{
			ui.spinScaleMin->setValue(P.metricList[P.selectedMetric].scaleLow);
			ui.spinScaleMax->setValue(P.metricList[P.selectedMetric].scaleHigh);
			ui.spinReferenceEpsilon->setValue(P.metricList[P.selectedMetric].refEpsilon);
		}
		ui.spinSpectralMin->setValue(P.spectralMin);
		ui.spinSpectralMax->setValue(P.spectralMax);

		if(P.scaleMode == automatic)
			ui.chkMinMax->setChecked(true);
		else
			ui.chkMinMax->setChecked(false);


		ui.spinHistogramBins->setValue(P.histBins);

		//metrics
		ui.spinMetricBand->setMaximum(P.dim.z - 1);
		ui.spinBaselinePointPosition->setMaximum(P.dim.z-1);
        if(P.metricList.size() > 0)
        {
            int m = P.selectedMetric;
            ui.spinMetricBand->setValue(P.metricList[m].band);
            ui.spinMetricBandwidth->setValue(P.metricList[m].bandwidth);

			//set the metric type in a combo box
            int typeIndex = ui.cmbMetricType->findData(P.metricList[m].type);
            ui.cmbMetricType->setCurrentIndex(typeIndex);

			//set the reference combo box to the correct metric
			int refID = P.metricList[m].reference;
			int refIndex = ui.cmbReferenceMetric->findData(refID);
			ui.cmbReferenceMetric->setCurrentIndex(refIndex);

			ui.spinMetricBand->setValue(P.metricList[m].band);

            //baseline points
            int b = P.selectedBaselinePoint;

            //if the current metric has no baseline points, disable the selection spinner
            if(P.metricList[m].baselinePoints.size() == 0)
			{
                ui.spinSelectedBaselinePoint->setEnabled(false);
				ui.spinBaselinePointPosition->setEnabled(false);
			}
            else
            {
                //otherwise enable the selection spinner
				ui.spinSelectedBaselinePoint->setEnabled(true);
				ui.spinBaselinePointPosition->setEnabled(true);
				//set the maximum value for the selection spinner
                ui.spinSelectedBaselinePoint->setMaximum(P.metricList[m].baselinePoints.size() - 1);
				//set the selection spinner to the currently selected baseline point
				ui.spinSelectedBaselinePoint->setValue(b);
				//set the band of the selected baseline point
				ui.spinBaselinePointPosition->setValue(P.metricList[m].baselinePoints[b]);
            }
        }

		//transfer functions
		if(P.tfList.size() != 0)
		{
			int tf = P.selectedTF;
			QString style = "background-color: rgb(%1, %2, %3);";
			ui.btnColorTF->setStyleSheet(style.arg(P.tfList[tf].r).arg(P.tfList[tf].g).arg(P.tfList[tf].b));
			ui.spinMaxTFValue->setValue(P.tfList[P.selectedTF].tfMax);
			ui.spinMinTFValue->setValue(P.tfList[P.selectedTF].tfMin);
		}
		

		updating = false;

    }

public slots:
    void on_mnuLoadENVI_triggered(){
        QString qFileName = QFileDialog::getOpenFileName();

        //return if the user presses cancel
        if(qFileName.size() == 0) return;

        //if a file is currently loaded, free the memory
        if(P.cpuData != NULL) free(P.cpuData);

        string filename = qFileName.toUtf8().constData();
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

		//position the spatial and spectral visualization windows
		positionWindows();

		//update the UI to reflect the new file
        UpdateUI();
    }
    void on_mnuLoadProject_triggered(){

        //get the project filename
        QString qFileName = QFileDialog::getOpenFileName(this, tr("Open Project"), QString(), tr("SpecVis Projects (*.svp)"));

        //return if the user presses cancel
        if(qFileName.size() == 0) return;

        string filename = qFileName.toUtf8().constData();

        loadStatus status = LoadProject(filename);
        if(status != loadStatusOK)
        {
            QMessageBox msgBox;
            if(status == loadStatusInvalidProject)
                msgBox.setText("The selected file was not a valid SpecVis project.");

            msgBox.exec();
            return;
        }

        //position the spatial and spectral visualization windows
		positionWindows();

		//update the UI to reflect the new file
        UpdateMetricList();
    }

    void on_mnuSaveProject_triggered(){
        //get the project filename
        QString qFileName = QFileDialog::getSaveFileName(this, tr("Open Project"), QString(), tr("SpecVis Projects (*.svp)"));

        //return if the user presses cancel
        if(qFileName.size() == 0) return;

        string filename = qFileName.toUtf8().constData();

        SaveProject(filename);
    }


    void on_radDisplayRaw_clicked(bool b)
    {
		if(updating) return;

        P.displayMode = displayRaw;
		UpdateWindows();
    }

    void on_radDisplayMetrics_clicked(bool b)
    {
		if(updating) return;

        P.displayMode = displayMetrics;
		UpdateWindows();
    }
	void on_radDisplayTF_clicked(bool b)
	{
		if(updating) return;
		P.displayMode = displayTF;
		UpdateWindows();
	}
	void on_chkMinMax_clicked(bool b)
	{
		if(updating) return;

		if(b) P.scaleMode = automatic;
		else P.scaleMode = manual;
		UpdateSpatialWindow();
	}
	void on_spinSpectralMin_valueChanged(double d)
	{
		if(updating) return;

		P.spectralMin = d;
		UpdateSpectralWindow();
	}
	void on_spinSpectralMax_valueChanged(double d)
	{
		if(updating) return;

		P.spectralMax = d;
		UpdateSpectralWindow();
	}
	void on_spinScaleMin_valueChanged(double d)
	{
		if(updating) return;

		P.scaleMode = manual;
		if(P.displayMode == displayRaw)
			P.scaleMin = d;
		else
			P.metricList[P.selectedMetric].scaleLow = d;

		UpdateUI();
		UpdateWindows();
	}
	void on_spinScaleMax_valueChanged(double d)
	{
		if(updating) return;

		P.scaleMode = manual;
		if(P.displayMode == displayRaw)
			P.scaleMax = d;
		else
			P.metricList[P.selectedMetric].scaleHigh = d;

		UpdateUI();
		UpdateWindows();
	}
	void on_chkDisplayHistogram_clicked(bool b)
	{
        if(updating) return;

        ui.spinHistogramBins->setEnabled(b);

        if(b)
        {
            P.spectrumMode = histogram;

            //create a histogram
            gpuCreateHistogram(&P.gpuHistogram, P.gpu_glHistBuffer, P.gpu_cudaHistResource, P.histBins, P.dim.z);
        }
        else
            P.spectrumMode = spectrum;

        UpdateUI();
        UpdateWindows();
	}
	void on_spinHistogramBins_valueChanged(int i)
	{
		if(updating) return;

		P.histBins = i;
		gpuCreateHistogram(&P.gpuHistogram, P.gpu_glHistBuffer, P.gpu_cudaHistResource, P.histBins, P.dim.z);

		UpdateUI();
        UpdateSpectralWindow();
	}

	//metrics
	void on_btnAddMetric_clicked()
	{
        metricStruct newMetric;

        //ask the user for a metric name
        bool ok;
        QString text = QInputDialog::getText(this, tr("Add Metric"), tr("Metric Name:"), QLineEdit::Normal, "metric", &ok);
        if (!ok && text.isEmpty())
            return;
        newMetric.name = text.toUtf8().constData();
        P.metricList.push_back(newMetric);
        P.selectedMetric = P.metricList.size() - 1;

		P.displayMode = displayMetrics;
		ui.radDisplayMetrics->setChecked(true);

        //update the metric list
        UpdateMetricList();

	}
	void on_cmbSelectedMetric_currentIndexChanged(int i)
	{
        if(updating) return;

        P.selectedMetric = i;
		P.selectedBaselinePoint = 0;
		//UpdateMetricList();
		UpdateWindows();
        UpdateMetricList();
	}
	void on_cmbSelectedTF_currentIndexChanged(int i)
	{
		if(updating) return;

		P.selectedTF = i;
		UpdateWindows();
		UpdateTFList();
	}
	void on_cmbTFSourceMetric_currentIndexChanged(int i)
	{
		if(updating) return;

		unsigned int tf = P.selectedTF;
		P.tfList[tf].sourceMetric = i;
		UpdateWindows();
		UpdateUI();

	}
	void on_cmbTransferFuncType_currentIndexChanged(int i)
	{
		if(updating) return;
		int tf = P.selectedTF;
		P.tfList[tf].type = (transferFuncType)ui.cmbTransferFuncType->itemData(i).toInt();
		UpdateSpatialWindow();
	}
	void on_cmbMetricType_currentIndexChanged(int i)
	{
        if(updating || P.metricList.size() == 0) return;

        int m = P.selectedMetric;
        P.metricList[m].type = (metricType)ui.cmbMetricType->itemData(i).toInt();
		UpdateWindows();
        UpdateUI();
    }
    void on_btnRenameMetric_clicked()
    {
        if(P.metricList.size() == 0) return;

        int m = P.selectedMetric;
        //ask the user for a metric name
        bool ok;
        QString text = QInputDialog::getText(this, tr("Add Metric"), tr("Metric Name:"), QLineEdit::Normal, P.metricList[m].name.c_str(), &ok);
        if (!ok && text.isEmpty())
            return;

        P.metricList[m].name = text.toUtf8().constData();
        UpdateMetricList();
    }
	void on_btnDuplicateMetric_clicked()
	{
		if(P.metricList.size() == 0) return;

        int m = P.selectedMetric;
        //ask the user for a metric name
        bool ok;
        QString text = QInputDialog::getText(this, tr("Add Metric"), tr("Metric Name:"), QLineEdit::Normal, P.metricList[m].name.c_str(), &ok);
        if (!ok && text.isEmpty())
            return;

		//create the new metric
		metricStruct newMetric = P.metricList[m];
		newMetric.name = text.toUtf8().constData();
        P.metricList.push_back(newMetric);
        P.selectedMetric = P.metricList.size() - 1;

        //update the metric list
        UpdateMetricList();
	}
    void on_spinSelectedBaselinePoint_valueChanged(int i)
    {
        if(updating) return;

        P.selectedBaselinePoint = i;
        UpdateUI();
		UpdateSpectralWindow();
    }
	void on_spinBaselinePointPosition_valueChanged(int i)
	{
		if(updating) return;

		int m = P.selectedMetric;
		int b = P.selectedBaselinePoint;

		//set the new position of the selected baseline point
		P.metricList[m].baselinePoints[b] = i;
		UpdateWindows();
	}

	void on_btnAddBaselinePoint_clicked()
	{
		//if there are no metrics, return
		if(P.metricList.size() == 0) return;

		unsigned int m = P.selectedMetric;
		P.metricList[m].baselinePoints.push_back(0);
		P.selectedBaselinePoint = P.metricList[m].baselinePoints.size() - 1;
		UpdateUI();
	}

	void on_spinMetricBand_valueChanged(int i)
	{
		if(updating) return;

		int m = P.selectedMetric;

		//update the band position
		P.metricList[m].band = i;
	}

	void on_spinMetricBandwidth_valueChanged(int i)
	{
		if(updating) return;

		int m = P.selectedMetric;

		//update the metric bandwidth
		P.metricList[m].bandwidth = i;
		UpdateWindows();
	}

	void on_cmbReferenceMetric_currentIndexChanged(int i)
	{
		if(updating) return;

		int m = P.selectedMetric;

		//update the metric reference
		P.metricList[m].reference = i - 1;

		UpdateWindows();
	}
	void on_spinReferenceEpsilon_valueChanged(double d)
	{
        if(updating) return;
        if(P.metricList.size() == 0) return;

        int m = P.selectedMetric;
        P.metricList[m].refEpsilon = d;
        UpdateWindows();

	}

	void on_btnColorTF_clicked()
	{
		if(updating) return;
		if(P.tfList.size() == 0) return;

		//set the swatch color based on the user-selected color
		QColor selectedColor = QColorDialog::getColor();
		QString style = "background-color: rgb(%1, %2, %3);";
		ui.btnColorTF->setStyleSheet(style.arg(selectedColor.red()).arg(selectedColor.green()).arg(selectedColor.blue()));

		//change the stored color to match the user selection
		int tf = P.selectedTF;
		P.tfList[tf].r = selectedColor.red();
		P.tfList[tf].g = selectedColor.green();
		P.tfList[tf].b = selectedColor.blue();

		UpdateSpatialWindow();

	}
	void on_btnAddTF_clicked()
	{
		//if there are no metrics, say so
		if(P.metricList.size() == 0)
		{
			QMessageBox msgBox;
            msgBox.setText("Transfer Functions are based on metrics. There are currently no metrics defined in this project.");

            msgBox.exec();
            return;
		}
		transferFuncStruct newTF;

        //ask the user for a metric name
        bool ok;
        QString text = QInputDialog::getText(this, tr("Add Transfer Function"), tr("TF Name:"), QLineEdit::Normal, "transferFunc", &ok);
        if (!ok && text.isEmpty())
            return;
        newTF.name = text.toUtf8().constData();
        P.tfList.push_back(newTF);
        P.selectedTF = P.tfList.size() - 1;

		//set the display mode to displayTF and update the radio button
		P.displayMode = displayTF;
		ui.radDisplayTF->setEnabled(true);
		ui.radDisplayTF->setChecked(true);

        //update the metric list
        UpdateTFList();
		UpdateWindows();
	}

};
