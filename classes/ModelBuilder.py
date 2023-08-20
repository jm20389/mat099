import numpy      as np
import seaborn    as sns
import tensorflow as tf
import keras
from matplotlib      import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages

class ModelBuilder:
    def __init__(self, epochs, batch_size, optimizer, metrics, loss, early_stopping, model = None):
        self.epochs =         epochs
        self.batch_size =     batch_size
        self.optimizer =      optimizer
        self.metrics =        metrics
        self.loss =           loss
        self.early_stopping = early_stopping
        self.model =          self.build_model() if model is None else model

    def build_model(self):
        model = Sequential()  # Sequential Model
        model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
        model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
        model.add(MaxPool2D(pool_size = (2, 2)))
        model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
        model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
        model.add(MaxPool2D(pool_size = (2, 2)))
        model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
        model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
        model.add(MaxPool2D(pool_size = (2, 2)))
        model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
        model.add(MaxPool2D(pool_size = (2, 2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation = 'sigmoid'))
        return model

    def modelCompile(self):
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)

    def modelTrain(self, X_train, Y_train, X_val, Y_val, as_dict = True):
        hist = self.model.fit(X_train
                              ,Y_train
                              ,batch_size = self.batch_size
                              ,epochs = self.epochs
                              ,validation_data = (X_val, Y_val)
                              ,callbacks = [self.early_stopping])
        self.history_dict = hist.history
        return hist.history if as_dict else hist

    def modelSave(self):
        self.model.save('h5')

    def plotHist(self, historyDict = None):
        history_dict = self.history_dict if historyDict is not None else historyDict

        fig, ax = plt.subplots(1,2,figsize=(15,5))
        #Figure 1
        ax[0].plot(history_dict['loss'], color='b', label = "Training loss")
        ax[0].plot(history_dict['val_loss'], color='r', label = "Validation loss",axes =ax[0])
        ax[0].set_xlabel('Epochs',fontsize=16)
        ax[0].set_ylabel('Loss',fontsize=16)
        legend = ax[0].legend(loc='best', shadow=True)
        #Figure 2
        ax[1].plot(history_dict['accuracy'], color='b', label = "Training accuracy")
        ax[1].plot(history_dict['val_accuracy'], color='r',label = "Validation accuracy")
        ax[1].set_xlabel('Epochs',fontsize=16)
        ax[1].set_ylabel('Accuracy',fontsize=16)
        legend = ax[1].legend(loc='best', shadow=True)

        fig.suptitle('Metrics',fontsize=20)
        return None

    def plot_confusion_matrix(cf_matrix):
        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()] #number of images in each classification block
        group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)] #percentage value of images in each block w.r.t total images

        axes_labels=['Forged', 'Authentic']
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cf_matrix, annot=labels, fmt='',cmap="flare" , xticklabels=axes_labels, yticklabels=axes_labels)

        plot_xlabel = plt.xlabel('Predicted labels', fontsize = 13)
        plot_ylabel = plt.ylabel('True labels', fontsize = 13)
        plot_title = plt.title('Confusion Matrix', fontsize= 10,fontweight='bold')

    def ROC(self):
        return roc_curve(self.Y_test, self.Y_prob) # fpr, tpr, thresholds = return...

    def AUC(self):
        return roc_auc_score(self.Y_test, self.Y_prob) # auc_score = return...

    def plotROC(fpr, tpr, auc_score):
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
