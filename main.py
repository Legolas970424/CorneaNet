from model import *
from data import *
from loss import *
from metric import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import ReduceLROnPlateau
#from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model
from keras import backend as K


# Training parameters
batch_size = 4
epochs = 300
steps_per_epoch = 60
validation_steps = 10
flag_multi_class = True
num_class = 4
data_augmentation = True


# define model
model = CorneaNet()

model.compile(optimizer = Adam(lr = 1e-4),
              loss = cce_dice_loss,             
              metrics=[iou_score])

print(model.summary())
plot_model(model, to_file='CorneaNet.png', show_shapes=True)

# define generator
train_Gene = trainGenerator(image_path = './data/train/image',
                            mask_path = './data/train/label',
                            batch_size = batch_size,
                            flag_multi_class = flag_multi_class,
                            num_class = num_class,
                            image_as_gray = True,
                            mask_as_gray = True)

valid_Gene = validGenerator(image_path = './data/valid/image',
                            mask_path = './data/valid/label',
                            batch_size = batch_size,
                            flag_multi_class = flag_multi_class,
                            num_class = num_class,
                            image_as_gray = True,
                            mask_as_gray = True)

# Prepare model model saving directory
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'weights.{epoch:02d}-{val_loss:.4f}.hdf5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               patience=5,                              
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, TensorBoard(log_dir='./logs')]

# Run training
if data_augmentation:
    print('Using real-time data augmentation.')
    hist = model.fit_generator(train_Gene,
                               steps_per_epoch = steps_per_epoch,
                               epochs = epochs,
                               verbose = 1,
                               validation_data = valid_Gene,
                               validation_steps = validation_steps,
                               callbacks=callbacks)
    
    with open('log.txt','w') as f:
        f.write(str(hist.history))   
        
else:
    print('Not using real-time data augmentation.')
    


