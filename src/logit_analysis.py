import pandas as pd
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
 
df = pd.read_csv('./pictures/logits/e5_finetuned_model_predictions_with_logits.csv')
#df['logits'] = df['logits'].apply(ast.literal_eval)
#df['std'] = df['logits'].apply(lambda x: pd.Series(x).std())
 
def Distribution_of_Confidence_Levels_by_Certainty(x_label, df, class_name, save_path=None):
      certainty_level = df['certainty'].unique().tolist()
      for level in certainty_level:
        df_level = df[df['certainty'] == level]
        sns.set(style="whitegrid")
        plt.figure(figsize=(20, 4))
        sns.histplot(df_level, x= x_label, hue='certainty', kde=True, bins=10, multiple="stack", palette="coolwarm")
        plt.title(f'Distribution of Confidence Levels by Certainty: {level} for classname: {class_name}', fontsize=30)
        plt.xlabel('Confidence Levels', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)  # Adjust the rotation and fontsize
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0:  # Avoid annotating empty bars
                plt.gca().annotate(f'{int(height)}',
                                  (p.get_x() + p.get_width() / 2., height),
                                  ha='center', va='center',
                                  fontsize=10, color='black',
                                  xytext=(0, 8), textcoords='offset points')
        if save_path:
                    file_name = f"{class_name}_certainty_{level}.png"  # Customize the file name
                    full_path = f"{save_path}/{file_name}"
                    plt.savefig(full_path, bbox_inches='tight')  # Save the figure to the path
                    print(f"Plot saved to: {full_path}")
       
     
        plt.show()
 
def plot_all_in_one(x_label, df, class_name, status, save_path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.histplot(df, x=x_label, hue='certainty', kde=True, bins=10, multiple="stack", palette="coolwarm")
    plt.title(f'Distribution of Confidence Levels for classname: {class_name} and {status}', fontsize=30)
    plt.xlabel('Confidence Levels', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:  # Avoid annotating empty bars
            plt.gca().annotate(f'{int(height)}',
                              (p.get_x() + p.get_width() / 2., height),
                              ha='center', va='center',
                              fontsize=10, color='black',
                              xytext=(0, 8), textcoords='offset points')
    if save_path:
                    file_name = f"{class_name}_certainty_{status}.png"  # Customize the file name
                    full_path = f"{save_path}/{file_name}"
                    plt.savefig(full_path, bbox_inches='tight')  # Save the figure to the path
                    print(f"Plot saved to: {full_path}")
       
    plt.show()
 
def plot_second(x_label, df, class_name, save_path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(40, 30))
    sns.histplot(df, x=x_label, hue='second_highest_label_name', kde=True, bins=10, multiple="stack", palette="coolwarm")
    plt.title(f'MissClassification of classname: {class_name}', fontsize=30)
    plt.xlabel('MissClassification', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.xticks(rotation=45, ha="right", fontsize=30)  # Adjust the rotation and fontsize
   
    for p in plt.gca().patches:
        height = p.get_height()
       # if height > 0:  # Avoid annotating empty bars
        #    plt.gca().annotate(f'{int(height)}',
         #                     (p.get_x() + p.get_width() / 2., height),
          #                    ha='center', va='center',
           #                   fontsize=20, color='black',
            #                  xytext=(20, 30), textcoords='offset points')
    if save_path:
                    file_name = f"{class_name}_certainty_.png"  # Customize the file name
                    full_path = f"{save_path}/{file_name}"
                    plt.savefig(full_path, bbox_inches='tight')  # Save the figure to the path
                    print(f"Plot saved to: {full_path}")
       
    plt.show()
 
 
save_path='./pictures/logits/'
class_name = df['true_label_name'].unique().tolist()
for clas in class_name:
  df_class = df[df['true_label_name'] == clas]
  df_misclassified = df_class[df_class['predicted_label_name'] != clas]
 
  plot_second('predicted_label_name', df_misclassified, clas, save_path)  
 
  Distribution_of_Confidence_Levels_by_Certainty('predicted_label_name', df_class, clas, save_path)  
 
 
  df_correctly_classified = df_class[df_class['predicted_label_name'] == clas]
  plot_all_in_one('confidence', df_correctly_classified, clas, 'Correctly_classified', save_path)
 
  df_misclassified = df_class[df_class['predicted_label_name'] != clas]
  plot_all_in_one('confidence', df_misclassified, clas, 'Miss_classified', save_path)
 
  #df_misclassified_second = df_misclassified[df_misclassified['second_highest_label_name'] == clas]