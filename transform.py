import numpy as np
import matplotlib.pyplot as plt
import math


import pywt.data

class WaveletFilter:
    def __init__(self, array, offset=0):
        self.offset = offset
        self.array = array.copy()
    def forward_periodic(self, input_array):
        output = []
        if len(input_array)%2==0:
            multiple = 1
        else:
            multiple = 2
        for n in range(multiple*len(input_array)):
            value = 0
            for index in range(len(self.array)):
                value += self.array[(index)%len(self.array)]*input_array[(index+n+self.offset)%len(input_array)]
            if(n%2==0):
                output.append(2*value)
        return np.array(output)
    def backward_periodic(self, input_array):
        output = []
        for n in range(2*len(input_array)):
            value = 0
            for index in range(len(self.array)):
                if (n-index-self.offset)%2==0:
                    input_value = input_array[((n-index-self.offset)//2)%len(input_array)]
                else:
                    input_value = 0
                #print(self.array[index])
                #print(n)
                #print(input_value)
                value += self.array[index]*input_value
            output.append(2*value)
        return np.array(output)

class Wavelet:
    def __init__(self, arrays, offsets = None):
        self.filters = []
        if offsets is None:
            for index in range(len(arrays)):
                self.filters.append(WaveletFilter(np.array(arrays[index])))
        else:
            for index in range(len(arrays)):
                self.filters.append(WaveletFilter(np.array(arrays[index]), offsets[index]))
    def forward_periodic(self, input_array):
        output = []
        for f in self.filters:
            output.append(f.forward_periodic(input_array))
        return output
    def backward_periodic(self, input_arrays):
        output = None
        for index in range(len(input_arrays)):
            if output is None:
                output = self.filters[index].backward_periodic(input_arrays[index])
            else:
                output += self.filters[index].backward_periodic(input_arrays[index])
        return output/2

def psnr(source, approximation):
    max_value = 255
    mse = np.mean((source-approximation)**2)
    return 20*math.log10(max_value)-10*math.log10(mse)

def add_noise(original, std_dev):
    noise = np.random.normal(scale=std_dev, size=original.shape)
    return original+noise

def threshold(values, epsilon, typeStr = "soft"):
    output = []
    if isinstance(typeStr, str):
        if typeStr.lower() == "soft":
            for item in values:
                output.append(soft_threshold(item, epsilon))
        elif typeStr.lower() == "hard":
            for item in values:
                output.append(hard_threshold(values, epsilon))
    return output

def hard_threshold(values, epsilon):
    thresholded = values.copy()
    thresholded[abs(thresholded)<=epsilon] = 0
    return thresholded

def soft_threshold(values, epsilon):
    return np.sign(values) * np.maximum(np.abs(values)-epsilon, 0)

def calc_threshold(data, std_dev):
    dev_orig_approx = max(np.var(data)-std_dev**2, 0)
    if dev_orig_approx != 0:
        return std_dev**2/math.sqrt(dev_orig_approx)
    else:
        return abs(data).max()

def randomNull(data, percentRemain=1):
    out_data = data.copy()
    random_indecies = np.random.permutation(len(data))
    for i in range(int(len(out_data)*percentRemain)):
        out_data[random_indecies[i]] = 0
    return out_data

def main():
    data_length = 1024
    std_dev = 20
    # Define wavelets
    haar = Wavelet(([1/2, 1/2], [-1/2, 1/2]), [0,0])
    bi1 = Wavelet(([-1/8, 1/4, 3/4, 1/4, -1/8], [-1/4, 1/2, -1/4]), [-2, 0])
    bi2 = Wavelet(([1/4, 1/2, 1/4], [-1/8, -1/4, 3/4, -1/4, -1/8]), [-1, -1])
    data = pywt.data.demo_signal('piece-polynomial', data_length)
    original_data = data.copy()
    plt.figure()
    plt.subplot(321)
    plt.plot(original_data)
    data = add_noise(data, std_dev)
    plt.subplot(322)
    plt.plot(data)
    output = haar.forward_periodic(data)
    output2 = haar.forward_periodic(output[0])
    thresholded2= threshold(output2, calc_threshold(original_data, std_dev))
    thresholded = threshold(output, calc_threshold(original_data, std_dev))
    out = haar.backward_periodic(thresholded)
    thresholded[0] = haar.backward_periodic(thresholded2)
    out2 = haar.backward_periodic(thresholded)
    plt.subplot(323)
    plt.plot(out)
    plt.subplot(324)
    plt.plot(out2)
    print(psnr(original_data, out))
    print(psnr(original_data, out2))
    output = bi1.forward_periodic(data)
    thresholded = threshold(output, calc_threshold(original_data, std_dev))
    out = bi2.backward_periodic(thresholded)
    plt.subplot(325)
    plt.plot(out)
    output = bi2.forward_periodic(data)
    thresholded = threshold(output, calc_threshold(original_data, std_dev))
    out = bi1.backward_periodic(output)
    plt.subplot(326)
    plt.plot(out)
    plt.show()
    # Load image
    #dev = 1
    #original = pywt.data.camera()
    # original = 
    #noisy = add_noise(original, dev)
    #LL, (LH, HL, HH)=pywt.dwt2(noisy, 'haar')
    #epsilon = calc_threshold(noisy, dev)
    #soft_LL = soft_threshold(LL, epsilon)
    #soft_LH = soft_threshold(LH, epsilon)
    #soft_HL = soft_threshold(HL, epsilon)
    #soft_HH = soft_threshold(HH, epsilon)
    #approx = pywt.idwt2((soft_LL,(soft_LH, soft_HL, soft_HH)), 'haar')
    #plt.imshow(original, plt.cm.gray)
    #print( psnr(original, approx))
    #plt.show()
    #plt.imshow(approx, plt.cm.gray)
    #plt.show()

main()
