import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sklearn.svm
import sklearn.metrics as METRICS

def get_all_files_with_extension (folder_address, file_extension, process_sub_folders = True):
    all_files = []
    if process_sub_folders:
        for root, dirs, files in shutil.os.walk(folder_address):
            for file in files:
                if file.endswith("." + file_extension):
                    all_files.append(shutil.os.path.join(root, file))
        return (all_files)
    else:
        for file in shutil.os.listdir(folder_address):
            if file.endswith("." + file_extension): #".txt" ;
                all_files.append(folder_address + file)
        return (all_files)

def get_raw_data():
    data_input = {
     "train_pos": {"path" : "data/train/pos", "label" : 1},
     "train_neg": {"path" : "data/train/neg", "label" : 0},
     "test_pos" : {"path" : "data/test/pos" , "label" : 1},
     "test_neg" : {"path" : "data/test/neg" , "label" : 0},
    }

    train = []
    test  = []
    for key in data_input:
        all_files = get_all_files_with_extension(data_input[key]["path"] , "txt")
        label = data_input[key]["label"]
        for file_path in all_files:
            with open (file_path, "rt" , encoding="utf-8") as file_handle:
                file_content = file_handle.read()

            if "train" in key:
                train.append((file_content, label))
            else:
                test.append((file_content, label))
    return train, test

if __name__ == "__main__":
    train , test = get_raw_data()

    print("postivie train example:"  , train[0])
    print("negative train example:"  , train[-1])

    vectorizer = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(1,3))

    vectorizer.fit([x[0] for x in train])
    feature_names = vectorizer.get_feature_names()
    print ("all 1-grams: " , [x for x in vectorizer.get_feature_names() if x.count(" ") == 0])

    train_x = vectorizer.transform([x[0] for x in train])
    train_y = np.asarray([x[1] for x in train])

    test_x = vectorizer.transform([x[0] for x in test])
    test_y_true = np.asarray([x[1] for x in test])

    for C_value_range in range(-5, +5):
        print("Training   ... C_value_range:" , C_value_range, " C:" , 2**C_value_range)
        classifier = sklearn.svm.LinearSVC(C=2**C_value_range, verbose=0)
        classifier.fit(train_x, train_y)
        print("Predicting ...")
        test_y_pred = classifier.predict(test_x)
        print ("accuracy :" , METRICS.accuracy_score(test_y_true , test_y_pred))
        print ("-"*80)