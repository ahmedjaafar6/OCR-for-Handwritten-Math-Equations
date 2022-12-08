import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # fork11 = Convolution2D(nb_filters_1, nb_conv_init, nb_conv_init,  activation="relu", border_mode='same')(init)
        # fork12 = Convolution2D(nb_filters_1, nb_conv_init, nb_conv_init, activation="relu", border_mode='same')(init)
        # merge1 = concatenate([fork11, fork12], axis=1, name='merge1')
        # # concat_feat = concatenate([concat_feat, x], mode='concat', axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))
        # maxpool1 = MaxPooling2D(strides=(2,2), border_mode='same')(merge1)

        # fork21 = Convolution2D(nb_filters_2, nb_conv_mid, nb_conv_mid, activation="relu", border_mode='same')(maxpool1)
        # fork22 = Convolution2D(nb_filters_2, nb_conv_mid, nb_conv_mid, activation="relu", border_mode='same')(maxpool1)
        # merge2 = concatenate([fork21, fork22, ], axis=1, name='merge2')
        # maxpool2 = MaxPooling2D(strides=(2,2), border_mode='same')(merge2)

        # fork31 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
        # fork32 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
        # fork33 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
        # fork34 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
        # fork35 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
        # fork36 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
        # merge3 = concatenate([fork31, fork32, fork33, fork34, fork35, fork36, ], axis=1, name='merge3')
        # maxpool3 = MaxPooling2D(strides=(2,2), border_mode='same')(merge3)



        # Covolutional layer
        self.conv1 = nn.Sequential(
                        #fork1_1
                        nn.Conv2d(in_channels=1,
                                  out_channels=64,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2
                                  ),
                        nn.ReLU(),
                        #fork1_2
                        nn.Conv2d(in_channels=1,
                                  out_channels=64,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2
                                  ),
                        nn.ReLU(),
                        #merge
                        torch.cat
                        

                        
                        
        )
        #Fully connected layer
        self.out = nn.Linear(1080, 10)

    def forward(self, x):
        x = self.conv1(x)
        # flatten the output of conv2 to (batch_size, 16*14*14)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
