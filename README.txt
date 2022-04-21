Dependencies:
  - python=3.9.7
  - csv
  - string
  - nltk
  - keras
  - numpy
  - pandas
  - gensim.models.keyedvectors
  - tensorflow
  - sklearn
  - datetime
  - time
  - gzip
  - shutil
  - statistics
  - spacy
 
Download:
  - Graders will need to navigate to https://github.com/commonsense/conceptnet-numberbatch and download numberbatch-en-19.08.txt.gz (English-only), from there graders will need to unzip the file and save it as "numberbatch-en.txt" in the data folder
  - All other data, feature creation, and model training/evaluation scripts can be downloaded from our team's GitHub at [INSERT GITHUB URL], note some of these files were not created by our team and have been included for grader convenience.
  - Our project team gathered the following files from external data sources and included them in our GitHub:
    - 500_Reddit_users_posts_labels.csv from https://github.com/jpsain/Suicide-Severity/tree/master/Data
    - labMT from https://rdrr.io/cran/qdapDictionaries/man/labMT.html
    - AFINN-en-165.txt from https://github.com/fnielsen/afinn/tree/master/afinn/data
  - Our team created the External_Features.csv file using the Create_External_Features.py python script

Functionality:
  - 3+1-Label_Classification.py - preprocessing/training/evaluation for 3+1-label suicide risk classification scheme
  - 4-Label_Classification.py - preprocessing/training/evaluation for 4-label suicide risk classification scheme
  - 5-Label_Classification.py - preprocessing/training/evaluation for 5-label suicide risk classification scheme
  - Create_External_Features.py - preprocessing/feature creation for External_Features.csv. These external features are used in all 3 classification scripts (3+1, 4, 5)
  - read_oe_w2v_file_test_vectorize_data.ipynb - converting conceptnet txt.gz file to txt (the txt file is provided for graders)

Instructions to Run Code:
IMPORTANT: Please ensure you have downloaded and unzipped the the conceptnet file (see Download section) before starting
  1. Download code scripts and data files from the GitHub page at [INSERT GITHUB URL]
  2. [Optional] Open and run Create_External_Features.py to create the supplemental external features dataset OR do not run the script and use the provided external features csv file
  3. Open and run the 3+1-Label_Classification.py script to preprocess, train, and evaluate CNN and SVM-Linear model performance on the 3+1-label classification scheme. This script trains the model using the 500-user dataset and supplemental external feature dataset (I2)
    a. [Optional] Run the script again with lines 330 and 337 commented out to test model performance without the supplemental external features (I1)
    b. For both 3 and 3.a above the scripts will print out cross-validated model performance metrics and save CNN model predictions in a tsv file
    c. NOTE: This script takes approximately 30 minutes to run
  4. Open and run the 4-Label_Classification.py script to preprocess, train, and evaluate CNN and SVM-Linear model performance on the 4-label classification scheme. This script trains the model using the 500-user dataset and supplemental external feature dataset (I2)
    a. [Optional] Run the script again with lines 312 and 318 commented out to test model performance without the supplemental external features (I1)
    b. For both 4 and 4.a above the scripts will print out cross-validated model performance metrics and save CNN model predictions in a tsv file
    c. NOTE: This script takes approximately 30 minutes to run
  5. Open and run the 5-Label_Classification.py script to preprocess, train, and evaluate CNN and SVM-Linear model performance on the 5-label classification scheme. This script trains the model using the 500-user dataset and supplemental external feature dataset (I2)
    a. [Optional] Run the script again with lines 330 and 337 commented out to test model performance without the supplemental external features (I1)
    b. For both 5 and 5.a above the scripts will print out cross-validated model performance metrics and save CNN model predictions in a tsv file
    c. NOTE: This script takes approximately 30 minutes to run   
