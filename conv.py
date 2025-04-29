import numpy as np

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters

        # Filters is a 3d array 
        # Divide by 9 to reduce the variance and output valid values for effective training
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    

    def iterate_regions(self, image): 
        '''
        Generates all the possible regions for our filter during
        convulution.
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i: (i + 3), j: (j + 3)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs forward pass of the conv layer. Applys filters to input image
        then returns the convulved image.
        - input is a 2d numpy array
        '''
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            # Multiply values in filter and im_region and sums
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output
    
    def backprop(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''

        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        

        self.filters -= learn_rate * d_L_d_filters

        return None