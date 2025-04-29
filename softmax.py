import numpy as np


class Softmax:


    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def backprop(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            
            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients against totals
            d_out_d_t = (-t_exp[i] * t_exp) / (S ** 2) 
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # gradients of totals against w/b/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against w/b/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.last_input_shape)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        # cache
        self.last_input_shape = input.shape

        # flatten input
        input = input.flatten()

        # cache
        self.last_input = input

        input_len, nodes = self.weights.shape

        # totals for nodes with dot product
        totals = np.dot(input, self.weights) + self.biases

        # cache
        self.last_totals = totals

        # e^totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)
