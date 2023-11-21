import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



if __name__ == '__main__': 
    res_dir =  '/home/weithan/Documents/projects/Spindle-Classification/result/Pt1_2_balanced_wavelet_data_padded_win1500_freq10_16_shift500/ckpt/'
    for fold in os.listdir(res_dir):
        if not os.path.isdir(os.path.join(res_dir, fold)):
            continue
        for file in os.listdir(os.path.join(res_dir, fold)):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(res_dir, fold, file))
            # Get AUC score
            auc_score = roc_auc_score(df['labels'], df['outputs'])

            # Generate ROC curve
            fpr, tpr, _ = roc_curve(df['labels'], df['outputs'])

            # Plot ROC curve
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.2f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(res_dir, fold, file.split('.')[0] + '_roc.png'))

            # Generate Confusion Matrix
            threshold = 0.5  # Adjust threshold based on your needs
            predictions = (df['outputs'] >= threshold).astype(int)
            conf_matrix = confusion_matrix(df['labels'], predictions)

            # Plot Confusion Matrix using Seaborn
            plt.figure(figsize=(6, 6))
            # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            #             xticklabels=['Predicted 0', 'Predicted 1'],
            #             yticklabels=['Actual 0', 'Actual 1'])
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
            disp.plot()
            plt.title('Confusion Matrix')
            # plt.xlabel('Predicted')
            # plt.ylabel('Actual')
            plt.savefig(os.path.join(res_dir, fold, file.split('.')[0] + '_confusion_matrix.png'))  
                    
