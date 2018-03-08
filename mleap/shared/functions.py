# from src.static_variables import REFORMATTED_DATASETS_DIR
# import pandas as pd

# def normalize_df(self,dataset_name):
#     dataFrame = pd.read_csv(REFORMATTED_DATASETS_DIR + dataset_name + '.csv')
#     dataFrame = (dataFrame - dataFrame.mean()) / dataFrame.std()
#     dataFrame.to_csv(REFORMATTED_DATASETS_DIR+dataset_name + '.csv', sep=',',index=False)
    
# def to_cat_values(df, cat_vals_array):
#     #convert text to categorical values
#     #approch taken from: http://pbpython.com/categorical-encoding.html
#     for c in cat_vals_array:
#         df[c] = df[c].astype('category')
#         df[c] = df[c].cat.codes
#     return df

# def process_dataset(self, dataset, min_num_examples):
#     #process dataset to change class of examples if number below certain treashold 
#     max_class_num = dataset['clase'].max()
#     replacement_index = max_class_num + 1
#     split_per_class = dataset['clase'].value_counts()
    
#     list_classes_below_treashold = []
#     for i in range(len(split_per_class)):
#         if split_per_class.iloc[i] < min_num_examples:
#             list_classes_below_treashold.append(split_per_class.index[i])
    
#     if len(list_classes_below_treashold) == 1:
#         #delete class if there are fewer examples than the treashold value
#         dataset = dataset[dataset.clase != list_classes_below_treashold[0]]
#     if len(list_classes_below_treashold) > 1:
#         dataset['clase'].replace(list_classes_below_treashold, replacement_index, inplace=True )
#     return dataset
    

        