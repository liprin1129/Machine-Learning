/********************************************************************************
** Form generated from reading UI file 'capturegamedialog.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CAPTUREGAMEDIALOG_H
#define UI_CAPTUREGAMEDIALOG_H

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

class Ui_CaptureGameDialog
{
public:
    QLabel *gameDisplayQlabel;
    QPlainTextEdit *gameInfoTextEdit;
    QPushButton *dirSetupButton;

    void setupUi(QDialog *CaptureGameDialog)
    {
        if (CaptureGameDialog->objectName().isEmpty())
            CaptureGameDialog->setObjectName(QStringLiteral("CaptureGameDialog"));
        CaptureGameDialog->resize(1300, 850);
        gameDisplayQlabel = new QLabel(CaptureGameDialog);
        gameDisplayQlabel->setObjectName(QStringLiteral("gameDisplayQlabel"));
        gameDisplayQlabel->setGeometry(QRect(10, 10, 1280, 720));
        gameDisplayQlabel->setAutoFillBackground(true);
        gameInfoTextEdit = new QPlainTextEdit(CaptureGameDialog);
        gameInfoTextEdit->setObjectName(QStringLiteral("gameInfoTextEdit"));
        gameInfoTextEdit->setGeometry(QRect(10, 740, 681, 100));
        gameInfoTextEdit->setAutoFillBackground(false);
        gameInfoTextEdit->setReadOnly(true);
        dirSetupButton = new QPushButton(CaptureGameDialog);
        dirSetupButton->setObjectName(QStringLiteral("dirSetupButton"));
        dirSetupButton->setGeometry(QRect(700, 740, 91, 41));

        retranslateUi(CaptureGameDialog);

        QMetaObject::connectSlotsByName(CaptureGameDialog);
    } // setupUi

    void retranslateUi(QDialog *CaptureGameDialog)
    {
        CaptureGameDialog->setWindowTitle(QApplication::translate("CaptureGameDialog", "Dialog", Q_NULLPTR));
        gameDisplayQlabel->setText(QString());
        dirSetupButton->setText(QApplication::translate("CaptureGameDialog", "...", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class CaptureGameDialog: public Ui_CaptureGameDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CAPTUREGAMEDIALOG_H
