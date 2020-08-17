#include "capturegamedialog.h"
#include "ui_capturegamedialog.h"
#include <opencv2/opencv.hpp>

//CaptureGameDialog::CaptureGameDialog(QImage &qImgL, QImage &qImgR, QWidget *parent) :
CaptureGameDialog::CaptureGameDialog(CameraMng *cm, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CaptureGameDialog)
{
    ui->setupUi(this);
    /*this->qImgL = qImgL;
    this->qImgR = qImgR;*/
    this->cm = cm;
    //this->flm = new FaceLandmarker();
    //this->hg = new HeadingGame();
    //this->is = new ImageSave();
}

CaptureGameDialog::~CaptureGameDialog()
{
    delete ui;
    //delete flm;
    //delete hg;
    //delete is;
}

void CaptureGameDialog::showEvent(QShowEvent * event)
{
    QDialog::showEvent(event);
    this->flm = new FaceLandmarker();
    this->hg = new HeadingGame();
    this->is = new ImageSave();

    tmrTimer = new QTimer(this);
    connect(tmrTimer, SIGNAL(timeout()), this, SLOT(getFrame()));
    tmrTimer->start(20);
}

void CaptureGameDialog::closeEvent(QCloseEvent *event)
{
    //QDialog::closeEvent(event);
    event->accept();
    delete flm;
    tmrTimer->stop();
    std::cout << "Game has finished." << std::endl;
}

void CaptureGameDialog::getFrame()
{
    //if (this->isVisible()){
    auto [cvCpuMatL, cvCpuMatR] = flm->findFaces(cm->cvGpuMatL, cm->cvGpuMatR);

    //hg->blendingTwoImages(cvCpuMatL);
    //hg->headingSuccessCecker(flm->faceL, cvCpuMatL);

    hg->setThreadingArgs(flm->faceL, cvCpuMatL);
    hg->start();

    if (!flm->faceL.empty()) {
        is->setArgs(dirPath, cvCpuMatL(flm->faceL));
        is->start();
    }

    //auto [cvCpuMatL, cvCpuMatR] = cm->getCpuMat();

    qImgL = QImage((uchar*)cvCpuMatL.data, cvCpuMatL.cols, cvCpuMatL.rows,
                    cvCpuMatL.step, QImage::Format_RGB888);

    ui->gameDisplayQlabel->setPixmap((QPixmap::fromImage(qImgL)));

    //ui->gameInfoTextEdit->appendPlainText();
    //}
}

void CaptureGameDialog::on_dirSetupButton_clicked()
{
    QString dirQPath = QFileDialog::getExistingDirectory(this,
                                                        tr("Open Directory"),
                                                        "/DATASETs/Face/Face_SJC/DEMO/",
                                                        QFileDialog::DontResolveSymlinks);
    ui->gameInfoTextEdit->appendPlainText(dirQPath);

    this->dirPath = dirQPath.toStdString();
    //std::cout << this->dirPath << std::endl;

    //this->is = new ImageSave(dirPath);
}
