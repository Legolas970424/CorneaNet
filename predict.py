from model import *
from data import *
from keras import backend as K


test_path = './data/test/aug'
model_path = './saved_models/weights.264-0.1044.hdf5'

model= load_model(model_path) 
test_Gene = testGenerator(test_path = test_path, as_gray = True)
results = model.predict_generator(test_Gene, steps=200, verbose=1)    
saveResult(test_path, results)    