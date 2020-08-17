#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>
#include <QTimer>
#include <opencv2/opencv.hpp>

#include "cameramng.h"
#include "capturegamedialog.h"

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

private:
    Ui::Dialog *ui;

    QImage qimgLeft;
    QImage qimgRight;
    QTimer* tmrTimer;

    CameraMng *cm;
    CaptureGameDialog *cgd;

    QImage qImgL, qImgR;

public:
    explicit Dialog(QWidget *parent = 0);
    ~Dialog();

public slots:
    void frameUpdate();

private slots:
    void on_videoSartBtn_clicked();
    void on_pushButton_clicked();
};

#endif // DIALOG_H
