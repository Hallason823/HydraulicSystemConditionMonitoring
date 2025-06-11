import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io
import os

class PlotManager:
    def __init__(self, training_history_path='./plot/training_history_images', results_path='./plot/results_images', confusion_matrix_path ='./plot/confusion_matrix_images', format_image='png', display_time=25):
        self.training_history_path = training_history_path
        self.results_path = results_path
        self.confusion_matrix_path = confusion_matrix_path
        self.format_image = format_image
        self.display_time = display_time
        self.training_fig = None
        self.confusion_matrix_fig = None
        self.results_fig = None
        self.training_history_images = []
        self.confusion_matrix_images = []
        self.results_images = []
    
    def transformFiguretoImage(self):
        buf = io.BytesIO()
        plt.savefig(buf, format=self.format_image)
        buf.seek(0)
        return buf
    
    def showandCloseFigurebyDisplayTime(self, fig):
        plt.show(block=False)
        plt.pause(self.display_time)
        if plt.fignum_exists(fig.number):  
            plt.close(fig)
        
    def plotLosses(self, history, title=None, window_title='Loss'):
        self.history = history
        if self.training_fig is None:
            self.training_fig = plt.figure()
            self.training_fig.canvas.manager.window.title(window_title)
        plt.title(title)
        plt.plot(self.history["train_loss"], c="g", label="train")
        plt.plot(self.history["val_loss"], c="r", label="valid")
        plt.legend()
        self.training_history_images.append(self.transformFiguretoImage())
        self.showandCloseFigurebyDisplayTime(self.training_fig)
        self.training_fig = None
    
    def plotResults(self, categories_name, estimated_conditions, evaluated_targets, window_title='Result: '):
        self.categories_name = categories_name
        for idx, category_name in enumerate(self.categories_name):
            if self.results_fig is None:
                self.results_fig = plt.figure()
                self.results_fig.canvas.manager.window.title(window_title+category_name)
            plt.plot([estimated_condition[idx] for estimated_condition in estimated_conditions], marker='o', linestyle='None', label='Estimated', color='blue')
            plt.plot([evaluated_target[idx] for evaluated_target in evaluated_targets], marker='o', linestyle='None', label='Data', color='red')
            plt.title(category_name)
            plt.xlabel('Samples')
            plt.ylabel('Values')
            plt.legend()
            self.results_images.append(self.transformFiguretoImage())
            self.showandCloseFigurebyDisplayTime(self.results_fig)
            self.results_fig = None

    def plotConfusionMatrix(self, categories_name, values_by_category, estimated_conditions, evaluated_targets, window_title='Confusion Matrix: '):
        self.categories_name = categories_name
        for idx, category_name in enumerate(self.categories_name):
            if self.confusion_matrix_fig is None:
                 self.confusion_matrix_fig = plt.figure()
                 self.confusion_matrix_fig.canvas.manager.window.title(window_title+category_name)
            cm = confusion_matrix([evaluated_target[idx] for evaluated_target in evaluated_targets], [estimated_condition[idx] for estimated_condition in estimated_conditions])
            ConfusionMatrixDisplay(cm, display_labels=values_by_category[idx]).plot(ax=plt.gca(), xticks_rotation=35.0)
            plt.title(category_name)
            self.confusion_matrix_images.append(self.transformFiguretoImage())
            self.showandCloseFigurebyDisplayTime(self.confusion_matrix_fig)
            self.confusion_matrix_fig = None
          
    def eraseFolder(self, folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delect {file_path}: {e}")
    
    def saveFilesbyFolder(self, files_main_name, images, output_folder_path):
        for idx, image_buf in enumerate(images):
            image = Image.open(image_buf)
            image_path = os.path.join(output_folder_path, f"{files_main_name}_{idx+1}.{self.format_image}")
            image.save(image_path)
    
    def saveAllImages(self):
        self.eraseFolder(self.training_history_path)
        self.eraseFolder(self.results_path)
        if self.training_history_images:
            self.saveFilesbyFolder('training_history', self.training_history_images, self.training_history_path)
        self.saveFilesbyFolder('results', self.results_images, self.results_path)
        self.saveFilesbyFolder('confusion_matrix_images', self.confusion_matrix_images, self.confusion_matrix_path)