import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QLineEdit, QPushButton, QFileDialog, QCalendarWidget, QMessageBox, QVBoxLayout, QMainWindow, QHBoxLayout, QScrollArea, QDialog, QDialogButtonBox
from PyQt5.QtCore import QDate, QCoreApplication, Qt
from PyQt5.QtGui import QTextDocument, QTextCursor, QPixmap, QIcon, QImage
from PyQt5.QtWidgets import QTextEdit, QTableWidget, QTableWidgetItem
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.backends.backend_qt5agg as mpl_backend

from print_and_display_anomaly import print_number_of_high_anomalies, print_number_of_mid_anomalies, print_number_of_low_anomalies, display_high_anomalies, display_mid_anomalies, display_low_anomalies, display_all_anomalies, drop_rows_by_timestamp
from methods import zscore_method, lof_method, autoencoder_method, isolation_forest_method, one_class_svm_method, robust_covariance_method, pca_method, prophet_method, unify_dataframes, print_graph, drop_rows_by_timestamp, detect_anomalies_ARIMA, exponential_smoothing_method, detect_anomalies_MA

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DateSelectionWindow(QMainWindow):
    def __init__(self, parent=None):
        super(DateSelectionWindow, self).__init__(parent)
        self.setWindowTitle("Tarih Seç")
        self.calendar_widget = QCalendarWidget(self)
        self.calendar_widget.setGridVisible(True)
        self.calendar_widget.clicked[QDate].connect(self.selectDate)
        self.setCentralWidget(self.calendar_widget)

    def selectDate(self, date):
        selected_date = date.toString("yyyy-MM-dd")
        parent = self.parent()
        if parent:
            parent.setDate(selected_date)
        self.close()

class MyProgram(QWidget):
    def __init__(self):
        super().__init__()
        self.language = "TR"  # Başlangıçta Türkçe olarak ayarlanmıştır
        app = QCoreApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        self.initUI()

        # QVBoxLayout oluştur
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.file_label)
        self.layout.addWidget(self.file_button)
        # self.layout.addWidget(self.type_label)
        # self.layout.addWidget(self.type_combo)
        self.layout.addWidget(self.date_label)
        self.layout.addWidget(self.start_date_label)
        self.layout.addWidget(self.start_date_edit)
        self.layout.addWidget(self.start_date_button)
        self.layout.addWidget(self.add_range_button)
        self.layout.addWidget(self.clear_ranges_button)
        self.layout.addWidget(self.range_label)
        self.layout.addWidget(self.range_list_label)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.close_button)
        self.layout.addWidget(self.info_button)
        self.layout.addWidget(self.language_button)
        self.setLayout(self.layout)
        self.date_ranges = []

    def initUI(self):
        self.setWindowTitle("Anomali Tespiti Projesi - Uğur Çakıl/Fatih Altıncı")
        self.setGeometry(300, 300, 450, 400)

        self.file_label = QLabel("Kullanmak istediğiniz veriyi seçin:", self)
        self.file_label.move(20, 20)

        self.file_button = QPushButton("Dosya Seç", self)
        self.file_button.move(200, 20)
        self.file_button.clicked.connect(self.openFileDialog)

        # self.type_label = QLabel("Verinin tipini girin:", self)
        # self.type_label.move(20, 60)

        # self.type_combo = QComboBox(self)
        # self.type_combo.addItem("Univariate")
        # self.type_combo.addItem("Multivariate")
        # self.type_combo.move(200, 60)

        self.date_label = QLabel("Anomali olmadığını düşündüğünüz tarihleri seçin:", self)
        self.date_label.move(20, 100)

        self.start_date_label = QLabel("Tarih:", self)
        self.start_date_label.move(20, 130)

        self.start_date_edit = QLineEdit(self)
        self.start_date_edit.move(150, 130)
        self.start_date_edit.setReadOnly(True)

        self.start_date_button = QPushButton("Tarih Seç", self)
        self.start_date_button.move(300, 130)
        self.start_date_button.clicked.connect(self.openStartDateCalendar)

        self.add_range_button = QPushButton("Yeni Tarih Ekle", self)
        self.add_range_button.move(20, 200)
        self.add_range_button.clicked.connect(self.addDateRange)

        self.clear_ranges_button = QPushButton("Seçilenleri Temizle", self)
        self.clear_ranges_button.move(150, 200)
        self.clear_ranges_button.clicked.connect(self.clearDateRanges)

        self.range_label = QLabel("Seçilen Tarihler:", self)
        self.range_label.move(20, 240)

        self.range_list_label = QLabel("", self)
        self.range_list_label.move(20, 270)

        self.save_button = QPushButton("Devam", self)
        self.save_button.move(150, 350)
        self.save_button.clicked.connect(self.saveData)
        self.save_button.clicked.connect(self.displayAnomalies)

        self.close_button = QPushButton("Kapat", self)
        self.close_button.move(250, 350)
        self.close_button.clicked.connect(self.closeEvent)

        self.info_button = QPushButton("?", self)
        self.info_button.setGeometry(430, 30, 20, 20)
        self.info_button.clicked.connect(self.showInfoDialog)

        self.language_button = QPushButton("TR/EN", self)
        self.language_button.setGeometry(400, 5, 50, 20)
        self.language_button.clicked.connect(self.changeLanguage)

        self.start_date_calendar = None

        self.date_ranges = []

    def openFileDialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Veri Seç", "", "CSV Dosyaları (*.csv)", options=options)
        df = None  # df değişkenini None olarak başlatıyoruz
        if file_name:
            print("Seçilen dosya:", file_name)
            df = pd.read_csv(file_name)
        self.df = df

    def openStartDateCalendar(self):
        self.start_date_calendar = DateSelectionWindow(self)
        self.start_date_calendar.show()



    def setDate(self, selected_date):
        if self.start_date_calendar.isVisible():
            self.start_date_edit.setText(selected_date)
            self.start_date_calendar.close()


    def selectStartDate(self, date):
        selected_date = date.toString("yyyy-MM-dd")
        self.start_date_edit.setText(selected_date)
        self.start_date_calendar.close()



    def addDateRange(self):
        start_date = self.start_date_edit.text()
        self.date_ranges.append(start_date)
        self.updateDateRanges()

        self.start_date_edit.clear()


    def clearDateRanges(self):
        self.date_ranges = []
        self.updateDateRanges()

    def updateDateRanges(self):
        self.range_list_label.setText("")
        for i, (start_date) in enumerate(self.date_ranges, start=1):
            self.range_list_label.setText(self.range_list_label.text() + f"{i}. {start_date} \n")
    def saveData(self):
        # selected_type = self.type_combo.currentText()
        start_date = self.start_date_edit.text()


        # Tarihleri kaydetmek için self.date_ranges'i kullanabilirsiniz
        print("Kaydedilen tarihler:")
        for i, (start) in enumerate(self.date_ranges, start=1):
            print(f"{i}. {start} ")

        # Verileri saklamak veya başka işlemler yapmak için buraya ekleyebilirsiniz
        # print("Seçilen veri tipi:", selected_type)
        print("Tarih:", start_date)


        self.lof_df = lof_method(self.df)
        self.zscore_df = zscore_method(self.df)
        self.autorencoder_df = autoencoder_method(self.df)
        self.isoforest_df = isolation_forest_method(self.df)
        self.ocsm_df = one_class_svm_method(self.df)
        self.robust_df = robust_covariance_method(self.df)
        self.pca_df = pca_method(self.df)
        self.arima_df = detect_anomalies_ARIMA(self.df)
        self.movav_df = detect_anomalies_MA(self.df)
        self.expsmooth_df = exponential_smoothing_method(self.df)
        self.prophet_df = prophet_method(self.df)
        self.df_sorted = unify_dataframes(self.lof_df, self.zscore_df, self.autorencoder_df, self.isoforest_df, self.ocsm_df, self.robust_df, self.pca_df, self.prophet_df, self.movav_df, self.arima_df, self.expsmooth_df)
        self.timestamps = self.date_ranges
    def displayAnomalies(self):
        self.anomaly_window = displayAnomalies(self.df_sorted, self.df, self.lof_df, self.zscore_df, self.autorencoder_df, self.isoforest_df, self.ocsm_df, self.robust_df, self.pca_df, self.prophet_df, self.movav_df, self.arima_df, self.expsmooth_df, self.timestamps)
        self.anomaly_window.show()

    def showInfoDialog(self):
        if self.language == "TR":
            message = "Bu program ile verideki anomalileri tespit edebilirsiniz. Bitirme projesi için hazırlanmıştır.\n\nFatih Altıncı\nfatihaltinci@gmail.com\nUğur Çakıl\nugrckl@gmail.com"
            title = "Hakkında"
        else:
            message = "You can detect anomalies on data with this program. Prepared for graduation project.\n\nFatih Altıncı\nfatihaltinci@gmail.com\nUğur Çakıl\nugrckl@gmail.com"
            title = "About"

        msg_box = QMessageBox()
        msg_box.information(self, title, message)

    def changeLanguage(self):
        if self.language == "TR":
            self.language = "EN"
            self.setWindowTitle("Anomaly Detection Project - Uğur Çakıl/Fatih Altıncı")
            self.language_button.setText("TR/EN")
            self.updateLabels()
        else:
            self.language = "TR"
            self.setWindowTitle("Anomali Tespiti Projesi - Uğur Çakıl/Fatih Altıncı")
            self.language_button.setText("EN/TR")
            self.updateLabels()

    def updateLabels(self):
        if self.language == "TR":
            self.file_label.setText("Kullanmak istediğiniz veriyi seçin:")
            self.file_button.setText("Dosya Seç")
            # self.type_label.setText("Veri tipini girin:")
            self.date_label.setText("Anomali olmadığını düşündüğünüz tarihi seçin:")
            self.start_date_label.setText("Tarih:")
            self.start_date_button.setText("Tarih Seç")
            # self.end_date_label.setText("Bitiş Tarihi:")
            # self.end_date_button.setText("Tarih Seç")
            self.add_range_button.setText("Yeni Tarih Ekle")
            self.clear_ranges_button.setText("Seçilenleri Temizle")
            self.range_label.setText("Seçilen Tarihler:")
            self.save_button.setText("Devam")
            self.close_button.setText("Kapat")
            self.setWindowTitle("Anomali Tespiti Projesi - Uğur Çakıl/Fatih Altıncı")
        else:
            self.file_label.setText("Select the data you want to use:")
            self.file_button.setText("Select File")
            # self.type_label.setText("Enter the type of data:")
            # self.type_label.setFixedWidth(200)
            self.date_label.setText("Select the date you believe has no anomalies:")
            self.start_date_label.setText("Date:")
            self.start_date_button.setText("Select Date")
            # self.end_date_label.setText("End Date:")
            # self.end_date_button.setText("Select Date")
            self.add_range_button.setText("Add Date")
            self.clear_ranges_button.setText("Clear Selections")
            self.range_label.setText("Selected Dates:")
            self.save_button.setText("Continue")
            self.close_button.setText("Close")
            self.setWindowTitle("Anomaly Detection Project - Uğur Çakıl/Fatih Altıncı")

    def closeApplication(self):
        QApplication.instance().quit()

    def closeEvent(self, event):
        self.closeApplication()

class displayAnomalies(QWidget):
    def __init__(self, df_sorted, df, lof_df, zscore_df, autorencoder_df, isoforest_df, ocsm_df, robust_df, pca_df, prophet_df, arima_df, movav_df, expsmooth_df, timestamps):
        super().__init__()
        self.setWindowTitle("Anomaly Detection")
        self.setGeometry(300, 300, 450, 400)
        self.timestamps = timestamps
        self.df_sorted = df_sorted
        self.df = df
        self.lof_df = lof_df
        self.zscore_df = zscore_df
        self.autorencoder_df = autorencoder_df
        self.isoforest_df = isoforest_df
        self.ocsm_df = ocsm_df
        self.robust_df = robust_df
        self.pca_df = pca_df
        self.prophet_df = prophet_df
        self.arima_df = arima_df
        self.movav_df = movav_df
        self.expsmooth_df = expsmooth_df
        self.language = "TR"
        # DisplayAnomaliesWindow örneği
        self.display_high_anomalies_window = None
        self.display_low_anomalies_window = None
        self.display_mid_anomalies_window = None
        self.display_all_anomalies_window = None
        self.high_anomalies_window = None
        self.low_anomalies_window = None
        self.mid_anomalies_window = None
        self.all_anomalies_window = None
        self.initUI()

    def initUI(self):
        high_value = print_number_of_high_anomalies(self.df_sorted, self.timestamps)
        medium_value = print_number_of_mid_anomalies(self.df_sorted, self.timestamps)
        low_value = print_number_of_low_anomalies(self.df_sorted, self.timestamps)

        high_label = QLabel(f"Yüksek Anomali: - {high_value} - ", self)
        high_button = QPushButton("Görüntüle", self)
        high_layout = QHBoxLayout()
        high_layout.addWidget(high_label)
        high_layout.addWidget(high_button)

        medium_label = QLabel(f"Orta Anomali: - {medium_value} - ", self)
        medium_button = QPushButton("Görüntüle", self)
        medium_layout = QHBoxLayout()
        medium_layout.addWidget(medium_label)
        medium_layout.addWidget(medium_button)

        low_label = QLabel(f"Düşük Anomali: - {low_value} - ", self)
        low_button = QPushButton("Görüntüle", self)
        low_layout = QHBoxLayout()
        low_layout.addWidget(low_label)
        low_layout.addWidget(low_button)

        display_button = QPushButton("Tüm Anomalileri Görüntüle", self)
        save_anomaly_button = QPushButton("Anomalileri Kaydet", self)

        file_name_input = QLineEdit(self)

        method_label = QLabel("Methodlara Göre Listele", self)
        method_label.setAlignment(Qt.AlignCenter)

        self.method_combobox = QComboBox(self)
        self.method_combobox.addItem("Z-Score")
        self.method_combobox.addItem("LOF")
        self.method_combobox.addItem("Autoencoder")
        self.method_combobox.addItem("Isolation Forest")
        self.method_combobox.addItem("OCSVM")
        self.method_combobox.addItem("Robust Covariance")
        self.method_combobox.addItem("PCA")
        self.method_combobox.addItem("Prophet")
        self.method_combobox.addItem("ARIMA")
        self.method_combobox.addItem("Moving Average")
        self.method_combobox.addItem("Exponential Smoothing")

        method_button = QPushButton("Görüntüle", self)

        layout = QVBoxLayout()
        layout.addLayout(high_layout)
        layout.addLayout(medium_layout)
        layout.addLayout(low_layout)
        layout.addSpacing(20)
        layout.addWidget(display_button)
        layout.addSpacing(20)
        layout.addWidget(file_name_input)
        layout.addWidget(save_anomaly_button)
        layout.addWidget(method_label)
        layout.addWidget(self.method_combobox)
        layout.addWidget(method_button)
        self.setLayout(layout)

        save_anomaly_button.clicked.connect(lambda: self.save_anomalies(file_name_input.text()))

        high_button.clicked.connect(self.display_in_high_anomalies)
        low_button.clicked.connect(self.display_in_low_anomalies)
        medium_button.clicked.connect(self.display_in_mid_anomalies)
        display_button.clicked.connect(self.display_all_anomalies)
        method_button.clicked.connect(self.print_method_graph)

    def save_anomalies(self, file_name):
        # Dosya yolunu oluştur
        file_path = os.path.join(os.getcwd(), f"{file_name}.csv")

        # Dizeyi CSV dosyasına yazdır
        all_anomalies = display_all_anomalies(self.df_sorted, self.timestamps)
        with open(file_path, "w") as f:
            f.write(all_anomalies)

        print("Anomalileri Kaydet:", file_name)


    def display_in_high_anomalies(self):
        high_anomalies = display_high_anomalies(self.df_sorted, self.timestamps)
        self.display_high_anomalies_window = DisplayHighAnomaliesWindow(high_anomalies)
        self.display_high_anomalies_window.show()

    def display_in_mid_anomalies(self):
        mid_anomalies = display_mid_anomalies(self.df_sorted, self.timestamps)
        self.display_mid_anomalies_window = DisplayMidAnomaliesWindow(mid_anomalies)
        self.display_mid_anomalies_window.show()

    def display_in_low_anomalies(self):
        low_anomalies = display_low_anomalies(self.df_sorted, self.timestamps)
        self.display_low_anomalies_window = DisplayLowAnomaliesWindow(low_anomalies)
        self.display_low_anomalies_window.show()

    def display_all_anomalies(self):
        all_anomalies = display_all_anomalies(self.df_sorted, self.timestamps)
        self.display_all_anomalies_window = DisplayAllAnomaliesWindow(all_anomalies)
        self.display_all_anomalies_window.show()

    def print_method_graph(self):
        method = self.method_combobox.currentText()
        if method == "Z-Score":
            print("Z-Score")
            graphs = print_graph(self.zscore_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.zscore_df, self.timestamps)
        elif method == "LOF":
            print("LOF")
            graphs = print_graph(self.lof_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.lof_df, self.timestamps)
        elif method == "Autoencoder":
            print("Autoencoder")
            graphs = print_graph(self.autorencoder_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.autorencoder_df, self.timestamps)
        elif method == "Isolation Forest":
            print("Isolation Forest")
            graphs = print_graph(self.isoforest_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.isoforest_df, self.timestamps)
        elif method == "OCSVM":
            print("OCSVM")
            graphs = print_graph(self.ocsm_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.ocsm_df, self.timestamps)
        elif method == "Robust Covariance":
            print("Robust Covariance")
            graphs = print_graph(self.robust_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.robust_df, self.timestamps)
        elif method == "PCA":
            print("PCA")
            graphs = print_graph(self.pca_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.pca_df, self.timestamps)
        elif method == "Prophet":
            print("Prophet")
            graphs = print_graph(self.prophet_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.prophet_df, self.timestamps)
        elif method == "ARIMA":
            print("ARIMA")
            graphs = print_graph(self.arima_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.arima_df, self.timestamps)
        elif method == "Moving Average":
            print("Moving Average")
            graphs = print_graph(self.movav_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.movav_df, self.timestamps)
        elif method == "Exponential Smoothing":
            print("Exponential Smoothing")
            graphs = print_graph(self.expsmooth_df, self.df, self.timestamps)
            anomalies = display_all_anomalies(self.expsmooth_df, self.timestamps)

        self.print_graph_window = PrintGraphWindow(graphs, anomalies)
        self.print_graph_window.show()

class DisplayHighAnomaliesWindow(QDialog):
    def __init__(self, high_anomalies):
        super().__init__()

        self.setWindowTitle("Yüksek Anomaliler")
        self.setGeometry(100, 100, 450, 280)

        self.layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.scroll_content = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_content)

        self.inner_layout = QVBoxLayout(self.scroll_content)

        self.label = QLabel(self.scroll_content)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText(high_anomalies)

        self.inner_layout.addWidget(self.label)
        self.layout.addWidget(self.scroll_area)

class DisplayMidAnomaliesWindow(QDialog):
    def __init__(self, high_anomalies):
        super().__init__()

        self.setWindowTitle("Orta Anomaliler")
        self.setGeometry(100, 100, 450, 280)

        self.layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.scroll_content = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_content)

        self.inner_layout = QVBoxLayout(self.scroll_content)

        self.label = QLabel(self.scroll_content)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText(high_anomalies)

        self.inner_layout.addWidget(self.label)
        self.layout.addWidget(self.scroll_area)


class DisplayLowAnomaliesWindow(QDialog):
    def __init__(self, high_anomalies):
        super().__init__()

        self.setWindowTitle("Düşük Anomaliler")
        self.setGeometry(100, 100, 450, 280)

        self.layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.scroll_content = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_content)

        self.inner_layout = QVBoxLayout(self.scroll_content)

        self.label = QLabel(self.scroll_content)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText(high_anomalies)

        self.inner_layout.addWidget(self.label)
        self.layout.addWidget(self.scroll_area)

class DisplayAllAnomaliesWindow(QDialog):
    def __init__(self, high_anomalies):
        super().__init__()

        self.setWindowTitle("Bütün Anomaliler")
        self.setGeometry(100, 100, 450, 280)

        self.layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.scroll_content = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_content)

        self.inner_layout = QVBoxLayout(self.scroll_content)

        self.label = QLabel(self.scroll_content)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText(high_anomalies)

        self.inner_layout.addWidget(self.label)
        self.layout.addWidget(self.scroll_area)

class PrintGraphWindow(QDialog):
    def __init__(self, graphs, anomalies):
        super().__init__()
        self.setWindowTitle("Grafikler")
        self.setModal(True)

        layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        for graph in graphs:
            graph.setMinimumSize(graph.sizeHint())  # Grafiklerin orijinal boyutunu ayarla
            scroll_layout.addWidget(graph)

        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Anomalileri gösteren QLabel oluştur
        anomalies_label = QLabel()
        anomalies_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        anomalies_label.setWordWrap(True)
        anomalies_label.setAlignment(Qt.AlignCenter)
        anomalies_label.setText(anomalies)

        # Anomalileri gösteren QLabel'ı bir scroll alanına yerleştir
        anomalies_scroll_area = QScrollArea()
        anomalies_scroll_area.setWidgetResizable(True)
        anomalies_scroll_area.setWidget(anomalies_label)

        layout.addWidget(anomalies_scroll_area)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)

        layout.addWidget(button_box)
        self.setLayout(layout)
        self.setMinimumSize(1300, 750)  # İlk açılış boyutunu ayarla

if __name__ == "__main__":
    app = QApplication(sys.argv)
    program = MyProgram()
    program.show()
    app.exec_()
    del app