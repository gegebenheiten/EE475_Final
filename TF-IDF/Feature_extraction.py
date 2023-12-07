import os
import glob
from googletrans import Translator

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

# 创建翻译器实例
translator = Translator()

# 翻译函数
def translate_to_english(text):
    # 使用googletrans进行翻译
    translation = translator.translate(text, src='zh-cn', dest='en')
    return translation.text


# Assuming your base directory is 'dataset_folder', adjust the path as necessary
base_dir = "C:\\Users\\26387\\Desktop\\learning\\NW\\First Year\\EE475\\final\\smart_feat_health\\dataset_folder"
text_files = []

# Use glob to match the directory pattern and then iterate over each match
for folder in glob.glob(os.path.join(base_dir, 'health_report_*')):
    # Construct the path to the text_file.txt within the folder
    file_path = os.path.join(folder, 'text_file.txt')
    # Check if the file exists
    if os.path.isfile(file_path):
        text_files.append(file_path)

content = []
# If you need to read the contents, you can do so like this:
for file in text_files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        english_text = translate_to_english(text)
        content.append(english_text)

print(content[0])


stop_words = set(open('en_stopwords.txt', encoding='utf-8').read().splitlines())

def cut_text(text):
    words = jieba.cut(text)
    # 过滤停用词
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


# Segment each piece of content
segmented_content = [cut_text(text) for text in content]
# print(segmented_content)
# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.9, max_features=10000)

# Fit and transform the segmented content
tfidf_matrix = vectorizer.fit_transform(segmented_content)

# Retrieve and print the feature names to understand what features have been created
feature_names = vectorizer.get_feature_names_out()
print("Features:", feature_names)

# Print the tf-idf values for the first document to understand its vector representation
first_document_vector = tfidf_matrix[0]
print(first_document_vector.T.todense(), feature_names)

# If you want to print the tf-idf matrix for all documents, convert to dense and print
dense_tfidf_matrix = tfidf_matrix.todense()
print(dense_tfidf_matrix)

df = pd.DataFrame(dense_tfidf_matrix, columns=feature_names)
print(df)
print(f'df.shape: {df.shape}')

# Assuming 'df' is the DataFrame created from the TF-IDF matrix
df.to_csv('tfidf_matrix.csv', index=False)




