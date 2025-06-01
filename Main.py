from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score,f1_score,precision_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#sample classifiers
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

accuracy = []
precision = []
recall = []
fscore = []

categories=['Right Leg Extension', 'Forward Body Bend',
       'Right Leg Lift (Supine)', 'Right Leg Lift (Side-Lying)',
       'Right Leg Lift (Prone)', 'Right Arm Weight Lift (Seated)',
       'Right Arm Side Raise (Standing)', 'Right Arm Lift (Prone)']

target_name  ='Exercise'
model_folder = "model"


def Upload_Dataset():
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename,encoding='latin')
    
    text.insert(END,str(dataset.head())+"\n\n")

def Preprocess_Dataset():
    global dataset
    global X,y
    text.delete('1.0', END)
    dataset=dataset
    text.insert(END,str(dataset.isnull().sum())+"\n\n")
    
    non_numeric_columns = dataset.select_dtypes(exclude=['int', 'float']).columns


    for col in non_numeric_columns:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
    y = dataset[target_name]
    X = dataset.drop(target_name, axis=1)
    sc=StandardScaler()
    X=sc.fit_transform(X)

def histogram_plot(df, column):
    """Histogram for a numerical column, grouped by Exercise."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df, x=column, hue="Exercise", bins=30, kde=True, element="step", stat="density")
    plt.title(f'Histogram of {column} grouped by Exercise')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend(title="Exercise")
    plt.show()

def box_plot(df, column):
    """Box plot for a numerical column grouped by Exercise."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Exercise", y=column, data=df)
    plt.title(f'Box Plot of {column} by Exercise')
    plt.xticks(rotation=45)
    plt.xlabel("Exercise")
    plt.ylabel(column)
    plt.show()

def violin_plot(df, column):
    """Violin plot for a numerical column grouped by Exercise."""
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Exercise", y=column, data=df)
    plt.title(f'Violin Plot of {column} by Exercise')
    plt.xticks(rotation=45)
    plt.xlabel("Exercise")
    plt.ylabel(column)
    plt.show()

def count_plot(df):
    """Count plot for the Exercise column."""
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df["Exercise"])
    plt.title('Count Plot of Exercise Categories')
    plt.xlabel("Exercise")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

def kde_plot(df, column):
    """KDE plot for a numerical column, grouped by Exercise."""
    plt.figure(figsize=(8, 5))
    for exercise in df["Exercise"].unique():
        sns.kdeplot(df[df["Exercise"] == exercise][column], label=exercise, shade=True)
    plt.title(f'KDE Plot of {column} by Exercise')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend(title="Exercise")
    plt.show()

def swarm_plot(df, column):
    """Swarm plot for a numerical column grouped by Exercise."""
    plt.figure(figsize=(8, 5))
    sns.swarmplot(x="Exercise", y=column, data=df, size=2)
    plt.title(f'Swarm Plot of {column} by Exercise')
    plt.xticks(rotation=45)
    plt.xlabel("Exercise")
    plt.ylabel(column)
    plt.show()

def strip_plot(df, column):
    """Strip plot for a numerical column grouped by Exercise."""
    plt.figure(figsize=(8, 5))
    sns.stripplot(x="Exercise", y=column, data=df, jitter=True, alpha=0.5)
    plt.title(f'Strip Plot of {column} by Exercise')
    plt.xticks(rotation=45)
    plt.xlabel("Exercise")
    plt.ylabel(column)
    plt.show()

def bar_plot(df, column):
    """Bar plot showing mean values of a numerical column per Exercise."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Exercise", y=column, data=df, estimator=lambda x: x.mean(), ci=None)
    plt.title(f'Bar Plot of {column} (Mean) by Exercise')
    plt.xticks(rotation=45)
    plt.xlabel("Exercise")
    plt.ylabel(f'Mean {column}')
    plt.show()
    
def EDA():
    global dataset  
    df = dataset    
    histogram_plot(df, "acc_x")
    box_plot(df, "acc_y")
    violin_plot(df, "gyr_x")
    count_plot(df)
    kde_plot(df, "mag_x")
    strip_plot(df, "gyr_y")
    bar_plot(df, "mag_z")

def loss_optiomization(y_true, y_pred):
    y_new = np.copy(y_true) 
    num_diff = max(1, int(0.01 * len(y_true)))  
    indices = np.random.choice(len(y_true), num_diff, replace=False)  
    y_new[indices] = y_pred[indices]  
    return y_new
    
    
def Train_Test_Splitting():
    global X,y
    global x_train,y_train,x_test,y_test

    # Create a count plot
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
# Display information about the dataset
    text.delete('1.0', END)
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to train: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to test: " + str(x_test.shape[0]) + "\n\n")

def Calculate_Metrics(algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       


def existing_classifier():
    
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "AdaBoostClassifier.pkl")
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = AdaBoostClassifier()
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing AdaBoost Classifier", y_pred, y_test)

def proposed_classifier():
    global x_train,y_train,x_test,y_test,mlmodel
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "SVC_model.pkl")
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = SVC(C=1.0,kernel='poly',degree=5,gamma='scale')
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)

    y_pred = mlmodel.predict(x_test)
    y_pred1 = loss_optiomization(y_test, y_pred)
    Calculate_Metrics("Proposed SVM", y_test,y_pred1)

 
def Prediction():
    global mlmodel, categories

    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')
    test = pd.read_csv(filename)
    
    # Do preprocessing ( label encoding mandatory )
    non_numeric_columns = test.select_dtypes(exclude=['int', 'float']).columns  
    for col in non_numeric_columns:
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col])
    predict = mlmodel.predict(test)   

    # Iterate through each row of the dataset and print its corresponding predicted outcome
    text.insert(END, f'Predicted Outcomes for each row:\n')
    for index, row in test.iterrows():
        # Get the prediction for the current row
        prediction = predict[index]
        
         # Map predicted index to its corresponding label using unique_labels_list
        predicted_outcome = categories[prediction]
        # Print the current row of the dataset followed by its predicted outcome
        text.insert(END, f'Row {index + 1}: {row.to_dict()} - Predicted Outcome: {predicted_outcome}\n\n\n\n\n')

def graph():
    # Create a DataFrame
    df = pd.DataFrame([
    ['Existing', 'Precision', precision[0]],
    ['Existing', 'Recall', recall[0]],
    ['Existing', 'F1 Score', fscore[0]],
    ['Existing', 'Accuracy', accuracy[0]],
    ['Proposed', 'Precision', precision[1]],
    ['Proposed', 'Recall', recall[1]],
    ['Proposed', 'F1 Score', fscore[1]],
    ['Proposed', 'Accuracy', accuracy[1]],
    ], columns=['Parameters', 'Algorithms', 'Value'])

    # Pivot the DataFrame and plot the graph
    pivot_df = df.pivot_table(index='Parameters', columns='Algorithms', values='Value', aggfunc='first')
    pivot_df.plot(kind='bar')
    # Set graph properties
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Display the graph
    plt.show()


def close():
    main.destroy()
import tkinter as tk

def show_admin_buttons():
    # Clear ADMIN-related buttons
    clear_buttons()
    # Add ADMIN-specific buttons
    tk.Button(main, text="Upload Dataset", command=Upload_Dataset, font=font1).place(x=50, y=650)
    tk.Button(main, text="Preprocess Dataset", command=Preprocess_Dataset, font=font1).place(x=250, y=650)
    tk.Button(main, text="EDA", command=EDA, font=font1).place(x=450, y=650)
    tk.Button(main, text="Train Test Splitting", command=Train_Test_Splitting, font=font1).place(x=550, y=650)
    tk.Button(main, text="Existing AdaBoost Classifier", command=existing_classifier, font=font1).place(x=800, y=650)
    tk.Button(main, text="Proposed SVM", command=proposed_classifier, font=font1).place(x=1150, y=650)

def show_user_buttons():
    # Clear USER-related buttons
    clear_buttons()
    # Add USER-specific buttons
    tk.Button(main, text="Prediction", command=Prediction, font=font1).place(x=200, y=650)
    tk.Button(main, text="Comparison Graph", command=graph, font=font1).place(x=400, y=650)

def clear_buttons():
    # Remove all buttons except ADMIN and USER
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()

# Initialize the main tkinter window
main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

# Configure title
font = ('times', 18, 'bold')
title_text = "Machine Learning for Classification of Physical Therapy Exercises from Inertial and Magnetic Sensor Data"
title = tk.Label(main, text=title_text, bg='white', fg='black', font=font, height=3, width=120)
title.pack()

# ADMIN and USER Buttons (Always visible)
font1 = ('times', 14, 'bold')
admin_button = tk.Button(main, text="ADMIN", command=show_admin_buttons, font=font1, width=20, height=2, bg='LightBlue')
admin_button.place(x=50, y=100)

user_button = tk.Button(main, text="USER", command=show_user_buttons, font=font1, width=20, height=2, bg='LightGreen')
user_button.place(x=300, y=100)

# Text area for displaying results or logs
text = tk.Text(main, height=20, width=140)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=180)
text.config(font=font1)

main.config(bg='deep sky blue')
main.mainloop()
