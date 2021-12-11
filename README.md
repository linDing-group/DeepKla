# DeepKla
an attention mechanism-based deep neural network for protein lysine lactylation site prediction in Rice (Oryza sativa)

DeepKla provides a deep-learning method for lactylation site prediction. It is implemented by deep learning library Keras and Tensorflow backend. At present, DeepKla only provides prediction of rice lactylation sites; however, it also provides customized model training that enables users to train other PTM prediction models by using their own training data sets.

Installation
Download DeepKla by
git clone https://github.com/linDing-group/DeepKla

Installation has been tested in Mac OS X with Python 2.7.

Since the package is written in python 2.7, python 2.7 with the pip tool must be installed first. DeepKla uses the following dependencies: numpy, h5py, keras version=2.0.6 

You can install these packages first, by the following commands:

pip install numpy

pip install h5py

pip install -v keras==2.0.6

pip install tensorflow


Predict on your own data :
cd to the predict folder which contains load_model.py, 
run:

python load_model.py <your_test_file>.fa <predict_result>.txt <pklFile>

Example:

python load_model.py ../data/fungiForTest.fa fungiForTest_predict_result.txt fungiForTest.pkl 

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For advanced users who want to perform training by using their own data:

For training:

python ../SEL_CNN_BiGRU_Attention.py

Note: You need to change fastafile, modelfile, pklfile, and model_save's name on your own.
