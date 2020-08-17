/********************************************************************************
** Form generated from reading UI file 'dialog.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DIALOG_H
#define UI_DIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_Dialog
{
public:
    QLabel *displayL;
    QLabel *displayR;
    QPushButton *videoSartBtn;
    QPlainTextEdit *logInfoTextEdit;
    QPushButton *pushButton;

    void setupUi(QDialog *Dialog)
    {
        if (Dialog->objectName().isEmpty())
            Dialog->setObjectName(QStringLiteral("Dialog"));
        Dialog->resize(1310, 560);
        displayL = new QLabel(Dialog);
        displayL->setObjectName(QStringLiteral("displayL"));
        displayL->setGeometry(QRect(10, 10, 640, 360));
        displayL->setAutoFillBackground(true);
        displayR = new QLabel(Dialog);
        displayR->setObjectName(QStringLiteral("displayR"));
        displayR->setGeometry(QRect(660, 10, 640, 360));
        displayR->setAutoFillBackground(true);
        videoSartBtn = new QPushButton(Dialog);
        videoSartBtn->setObjectName(QStringLiteral("videoSartBtn"));
        videoSartBtn->setGeometry(QRect(720, 380, 100, 50));
        logInfoTextEdit = new QPlainTextEdit(Dialog);
        logInfoTextEdit->setObjectName(QStringLiteral("logInfoTextEdit"));
        logInfoTextEdit->setGeometry(QRect(10, 380, 700, 170));
        logInfoTextEdit->setReadOnly(true);
        pushButton = new QPushButton(Dialog);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(720, 440, 100, 50));

        retranslateUi(Dialog);

        QMetaObject::connectSlotsByName(Dialog);
    } // setupUi

    void retranslateUi(QDialog *Dialog)
    {
        Dialog->setWindowTitle(QApplication::translate("Dialog", "Dialog", Q_NULLPTR));
        displayL->setText(QString());
        displayR->setText(QString());
        videoSartBtn->setText(QApplication::translate("Dialog", "Pause", Q_NULLPTR));
        pushButton->setText(QApplication::translate("Dialog", "Game", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class Dialog: public Ui_Dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOG_H
