#ifndef CAPTUREGAMEDIALOG_H
#define CAPTUREGAMEDIALOG_H

#include <QDialog>
#include <QTimer>
#include <QFileDialog>

//#include <opencv2/opencv.hpp>
#include "cameramng.h"
#include "facelandmarker.h"
#include "headinggame.h"
#include "imagesave.h"

namespace Ui {
class CaptureGameDialog;
}

class CaptureGameDialog : public QDialog
{
    Q_OBJECT

private:
    Ui::CaptureGameDialog *ui;

    void showEvent(QShowEvent * event);
    void closeEvent(QCloseEvent *event);

    CameraMng *cm;
    FaceLandmarker *flm;
    HeadingGame *hg;
    ImageSave * is;

    QImage qImgL, qImgR;
    std::string dirPath;

    int test;

    QTimer* tmrTimer;

private slots:
    void getFrame();

    //void on_pushButton_clicked();

    void on_dirSetupButton_clicked();

public:
    //explicit CaptureGameDialog(cv::cuda::GpuMat &cvGpuMatL, cv::cuda::GpuMat &cvGpuMatR, QWidget *parent = 0);
    explicit CaptureGameDialog(CameraMng *cm, QWidget *parent = 0);
    ~CaptureGameDialog();
};

#endif // CAPTUREGAMEDIALOG_H
