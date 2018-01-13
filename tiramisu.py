# Implementation of The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
# Author: Shobhit Lamba
# e-mail: slamba4@uic.edu

# Importing the libraries
from keras.layers import Dropout, Activation, Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2

# weight decay = 0.0001
decay = 1e-4

class TIRAMISU():
    '''A class to create the One Hundred Layers Tiramisu architecture.''' 
    
    def __init__(self):
        '''Initialization'''
        self.create_tiramisu()
        
    def dense_block(self, nb_layers, filters):
        ''' Function to define the Dense Block which consistes of:
            Batch Normalization
            ReLU activation
            3 x 3 Convolution layer
            Dropout with p = 0.2'''
            
        model = self.model
        
        for i in range(nb_layers):
            model.add(BatchNormalization(axis = -1, 
                                         gamma_regularizer = l2(decay),
                                         beta_regularizer = l2(decay)))
            model.add(Activation("relu"))
            model.add(Conv2D(filters, (3, 3), 
                       padding = "same", 
                       kernel_initializer = "he_uniform"))
            model.add(Dropout(0.2))
        
    def transition_down(self, filters):
        '''Function to define the Transition Down block which consists of:
            Batch Normalization
            ReLU activation
            1 x 1 Convolution layer
            Dropout with p = 0.2
            2 x 2 Max Pooling'''
            
        model = self.model
        
        model.add(BatchNormalization(axis = -1, 
                                     gamma_regularizer = l2(decay),
                                     beta_regularizer = l2(decay)))
        model.add(Activation("relu"))
        model.add(Conv2D(filters, (1, 1), 
                   padding = "same", 
                   kernel_initializer = "he_uniform"))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D((2, 2), strides = (2, 2)))
    
    def transition_up(self, filters, input_shape, output_shape):
        '''Function to create the transition Up block which consists of
           a 3 x 3 Transposed Convolution with stride = 2'''
        
        model = self.model
        
        model.add(Conv2DTranspose(filters, (3, 3), 
                            strides = (2, 2),
                            input_shape = input_shape,
                            output_shape = output_shape, 
                            kernel_initializer = "he_uniform"))
    
    def create_tiramisu(self):
        '''Function to generate the 103 layered Neural Network using 
           the previously defined Dense Block, Transition Down and Transition Up.'''
        
        model = self.model = Sequential()
        
        model.add(Conv2D(48, (3, 3), 
                         padding = "same", 
                         input_shape = (224, 224, 3), 
                         kernel_initializer = "he_uniform", 
                         kernel_regularizer = l2(decay)))
        
        self.dense_block(4, 112) # m = 4 * 16 + 48 = 112
        self.transition_down(112)     
        
        self.dense_block(5, 192) # m = 5 * 16 + 112 = 192
        self.transition_down(122) 
        
        self.dense_block(7, 304) # m = 7 * 16 + 192 = 304
        self.transition_down(304)     
        
        self.dense_block(10, 464) # m = 10 * 16 + 304 = 464
        self.transition_down(464)     
        
        self.dense_block(12, 656) # m = 12 * 16 + 464 = 656
        self.transition_down(656)
        
        self.dense_block(15, 896) # m = 15 * 16 + 656 = 896
        
        self.transition_up(1088, (1088, 7, 7), (None, 1088, 14, 14)) # m = 656 + 12 * 16 = 1088
        self.dense_block(12, 1088)
        
        self.transition_up(816, (816, 14, 14), (None, 816, 28, 28)) # m = 464 + 10 * 16 = 816
        self.dense_block(10, 816)
        
        self.transition_up(576, (576, 28, 28), (None, 576, 56, 56)) # m = 304 + 7 * 16 = 578
        self.dense_block(7, 578)        
        
        self.transition_up(384, (384, 56, 56), (None, 384, 112, 112)) # m = 192 + 5 * 16 = 384
        self.dense_block(5, 384)
        
        self.transition_up(256, (256, 112, 112), (None, 256, 224, 224)) # m = 112 + 4 * 16 = 256
        self.dense_block(4, 256)
        
        model.add(Conv2D(12, (1,1), 
                         padding = "same",
                         kernel_initializer = "he_uniform",
                         kernel_regularizer = l2(decay)))
        
        model.add(Activation("softmax"))
        model.summary()
        
TIRAMISU()        
