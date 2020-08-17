#include "dialog.h"
#include "ui_dialog.h"

Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{
    ui->setupUi(this);

    cm = new CameraMng();
    //cgd = new CaptureGameDialog(qImgL, qImgR, this);

    auto [isOpened, errMsg] = cm->openCamera();

    ui->logInfoTextEdit->appendPlainText(QString::fromStdString(errMsg));

    // tmrTimer
    tmrTimer = new QTimer(this);
    connect(tmrTimer, SIGNAL(timeout()), this, SLOT(frameUpdate()));
    tmrTimer->start(20);

    cgd = new CaptureGameDialog(cm, this);
}

Dialog::~Dialog()
{   tmrTimer->stop();

    delete ui;
    delete cm;
    delete cgd;
}

void Dialog::frameUpdate() {

    auto isGetSucess = cm->getOneFrameFromZED();

    if (isGetSucess == true) {
        //std::cout << cm->cvCpuMatL.type() << std::endl;
        /*QImage qImgL((uchar*)cm->cvCpuMatL.data, cm->cvCpuMatL.cols, cm->cvCpuMatL.rows,
                            cm->cvCpuMatL.step, QImage::Format_RGB888);
        QImage qImgR((uchar*)cm->cvCpuMatR.data, cm->cvCpuMatR.cols, cm->cvCpuMatR.rows,
                            cm->cvCpuMatR.step, QImage::Format_RGB888);*/
        /*qImgL = QImage((uchar*)cm->cvCpuMatL.data, cm->cvCpuMatL.cols, cm->cvCpuMatL.rows,
                       cm->cvCpuMatL.step, QImage::Format_RGB888);
        qImgR = QImage((uchar*)cm->cvCpuMatR.data, cm->cvCpuMatR.cols, cm->cvCpuMatR.rows,
                       cm->cvCpuMatR.step, QImage::Format_RGB888);*/

        /*cv::Mat cvCpuMatL, cvCpuMatR;
        cm->cvGpuMatL.download(cvCpuMatL);
        cm->cvGpuMatR.download(cvCpuMatR);*/
        auto [cvCpuMatL, cvCpuMatR] = cm->getCpuMat();

        qImgL = QImage((uchar*)cvCpuMatL.data, cvCpuMatL.cols, cvCpuMatL.rows,
                       cvCpuMatL.step, QImage::Format_RGB888);
        qImgR = QImage((uchar*)cvCpuMatR.data, cvCpuMatR.cols, cvCpuMatR.rows,
                       cvCpuMatR.step, QImage::Format_RGB888);

        ui->displayL->setPixmap(QPixmap::fromImage(qImgL.scaled(cvCpuMatL.cols/2, cvCpuMatL.rows/2)));
        ui->displayR->setPixmap(QPixmap::fromImage(qImgR.scaled(cvCpuMatR.cols/2, cvCpuMatR.rows/2)));
    }
}

void Dialog::on_videoSartBtn_clicked()
{
    //cm = CameraMng();
    //cm->openCamera();

    //std::this_thread::sleep_for(std::chrono::milliseconds(2000));


    if(tmrTimer->isActive() == true) {
        tmrTimer->stop();
        ui->videoSartBtn->setText("Resume");
    } else {
        tmrTimer->start(20);
        ui->videoSartBtn->setText("Pause");
    }

    /*char key = 'a';

    while (key != 'q') {
        auto isGetSucess = cm->getOneFrameFromZED();

        if (isGetSucess == true) {
            cv::imshow("Image", cm->cvCpuMatL);
            key = cv::waitKey(10);
        }
    }*/
}

void Dialog::on_pushButton_clicked()
{
    //cgd = new CaptureGameDialog(cm, this);

    if (cgd->isVisible() == false) {
        //std::cout << "cgd open." << std::endl;
        cgd->show();
    }
    /*else {
        //std::cout << "cgd close." << std::endl;
        cgd->close();
        delete cgd;
    }*/
}
