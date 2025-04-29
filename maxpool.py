import numpy as np

class MaxPool:

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions for pooling.
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h, new_w = h // 2, w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2): (i * 2 + 2), (j * 2): (j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Pools over image and returns a 3d numpt array
        - image is a 2d numpy array
        '''
        self.last_input_shape = input.shape
        self.last_input = input
        h, w, num_filters = input.shape

        input_reshaped = input.reshape(h//2, 2, w//2, 2, num_filters)
        output = input_reshaped.max(axis=(1, 3))

        return output
    
    def backprop(self, d_L_d_out):
        '''
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        '''

        d_L_d_input = np.zeros(self.last_input_shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            for f2 in range(f):
                region = im_region[:, :, f2]
                i2, j2 = np.unravel_index(np.argmax(region), region.shape)
                d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input