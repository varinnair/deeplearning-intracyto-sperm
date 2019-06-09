#import tensorflow as tf
#!pip install -q keras
from glob import glob
from sklearn.model_selection import train_test_split
cwd = os.getcwd()

normal_sperm = glob(cwd + '/train/normal_sperm/*.jpeg')
abnormal_sperm = glob(cwd + '/train/abnormal_sperm/*.jpeg')

normal_train, normal_test = train_test_split(normal_sperm, test_size=0.30)
abnormal_train, abnormal_test = train_test_split(abnormal_sperm, test_size=0.30)

TRAIN_DIR = 'train'
TEST_DIR = 'test'

!mkdir test

!mkdir test/Cat
files = ' '.join(normal_test)
!mv -t test/normal_sperm $files

!mkdir test/Dog
files = ' '.join(abnormal_test)
!mv -t test/abnormal_sperm $files