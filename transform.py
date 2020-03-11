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
    def backward_periodic(self, input_array, odd):
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
            if not odd or (odd and n%2==0):
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
    def backward_periodic(self, input_arrays, odd):
        output = None
        for index in range(len(input_arrays)):
            if output is None:
                output = self.filters[index].backward_periodic(input_arrays[index], odd)
            else:
                output += self.filters[index].backward_periodic(input_arrays[index], odd)
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
                output.append(hard_threshold(item, epsilon))
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
    # Data length, deviation of gaussian noise, datasources, thresholding type, number of trials for each deviation
    data_length = 1024
    std_devs = np.arange(0.125, 10, 0.125)
    datasources = ["Blocks", "Piece-Polynomial", "Piece-regular", "Ramp", "Doppler"]
    threshold_type = ["soft", "hard"]
    average_psnr = {"haar":{}, "haar2":{}, "bi1":{}, "bi1_2":{}, "bi2":{}, "bi2_2":{}}
    numRandom = 10
    # Define wavelets
    haar = Wavelet(([1/2, 1/2], [-1/2, 1/2]), [0,0])
    bi1 = Wavelet(([-1/8, 1/4, 3/4, 1/4, -1/8], [-1/4, 1/2, -1/4]), [-2, 0])
    bi2 = Wavelet(([1/4, 1/2, 1/4], [-1/8, -1/4, 3/4, -1/4, -1/8]), [-1, -1])
    # Iterate over the datasources
    for source in datasources:
        for key in average_psnr.keys():
            average_psnr[key][source] = {}
        data = pywt.data.demo_signal(source, data_length)
        original_data = data.copy()
        plt.plot(original_data)
        plt.savefig("SourceData-"+source+".png")
        plt.close()
        for std_dev in std_devs:
            for key in average_psnr.keys():
                average_psnr[key] = {}
            # Calculate a decent spread of thresholds
            thresholds = np.linspace(0, 5*calc_threshold(original_data, std_dev))
            for t in threshold_type:
                for key in average_psnr.keys():
                    average_psnr[key][t] = np.zeros(thresholds.shape)
            for i in range(numRandom):
                added_noise=add_noise(data, std_dev)
                plt.plot(added_noise)
                plt.savefig("Noisy-"+source+"-"+str(std_dev)+"-"+str(i)+".png")
                plt.close()
                # Calculate forward transform for all wavelets, and 2 deep combinations (currently of same wavelet)
                haar1 = haar.forward_periodic(added_noise)
                haar2 = haar.forward_periodic(haar1[0])
                bi1_1 = bi1.forward_periodic(added_noise)
                bi1_2 = bi1.forward_periodic(bi1_1[0])
                bi2_1 = bi2.forward_periodic(added_noise)
                bi2_2 = bi2.forward_periodic(bi2_1[0])
                # Threshold
                for thresh_index in range(len(thresholds)):
                    for t in threshold_type:
                        
                        # Technically, the transformed values is \sqrt(2)* the value received from forward. This doesn't change much as a constant multiple can continue though so long as it is kept track of, and indeed my backward takes the \sqrt(2) into account. Threshold values need to be adjusted though
                        haar1_t = threshold(haar1, thresholds[thresh_index]/math.sqrt(2), t)
                        haar2_t = threshold(haar2, thresholds[thresh_index]/2, t)
                        bi1_1_t = threshold(bi1_1, thresholds[thresh_index]/math.sqrt(2), t)
                        bi1_2_t = threshold(bi1_2, thresholds[thresh_index]/2, t)
                        bi2_1_t = threshold(bi2_1, thresholds[thresh_index]/math.sqrt(2), t)
                        bi2_2_t = threshold(bi2_2, thresholds[thresh_index]/2, t)
                        # Reconstruct from the thresholded values
                        haar4 = haar.backward_periodic(haar2_t, len(haar1[0])%2==1)
                        bi1_4 = bi2.backward_periodic(bi1_2_t, len(bi1_1[0])%2==1)
                        bi2_4 = bi1.backward_periodic(bi2_2_t, len(bi2_1[0])%2==1)
                        # Reverse depth 1
                        haar3 = haar.backward_periodic(haar1_t, data_length%2==1)
                        bi1_3 = bi2.backward_periodic(bi1_1_t, data_length%2==1)
                        bi2_3 = bi1.backward_periodic(bi2_1_t, data_length%2==1)
                        # Set first to denoised level 2 value
                        haar5 = haar1.copy()
                        bi1_5 = bi1_1.copy()
                        bi2_5 = bi2_1.copy()
                        haar5[0] = haar4
                        bi1_5[0] = bi1_4
                        bi2_5[0] = bi2_4
                        # Reverse depth 2
                        haar_2 = haar.backward_periodic(haar5, data_length%2==1)
                        bi1_22 = bi2.backward_periodic(bi1_5, data_length%2==1)
                        bi2_22 = bi1.backward_periodic(bi2_5, data_length%2==1)
                        if i==numRandom//2 and thresh_index%10==0:
                            plt.plot(haar3)
                            plt.savefig("Haar-1-Denoised-"+source+"-"+str(std_dev)+"-"+str(thresholds[thresh_index])+"-"+t+"-"+str(i)+".png")
                            plt.close()
                            plt.plot(haar_2)
                            plt.savefig("Haar-2-Denoised-"+source+"-"+str(std_dev)+"-"+str(thresholds[thresh_index])+"-"+t+"-"+str(i)+".png")
                            plt.close()
                            plt.plot(bi1_3)
                            plt.savefig("Bi1-1-Denoised-"+source+"-"+str(std_dev)+"-"+str(thresholds[thresh_index])+"-"+t+"-"+str(i)+".png")
                            plt.close()
                            plt.plot(bi2_3)
                            plt.savefig("Bi2-1-Denoised-"+source+"-"+str(std_dev)+"-"+str(thresholds[thresh_index])+"-"+t+"-"+str(i)+".png")
                            plt.close()
                            plt.plot(bi1_22)
                            plt.savefig("Bi1-2-Denoised-"+source+"-"+str(std_dev)+"-"+str(thresholds[thresh_index])+"-"+t+"-"+str(i)+".png")
                            plt.close()
                            plt.plot(bi2_22)
                            plt.savefig("Bi2-2-Denoised-"+source+"-"+str(std_dev)+"-"+str(thresholds[thresh_index])+"-"+t+"-"+str(i)+".png")
                            plt.close()
                        average_psnr["haar"][t][thresh_index] = i*average_psnr["haar"][t][thresh_index]/(i+1) + psnr(original_data, haar3)/(i+1)
                        average_psnr["haar2"][t][thresh_index] = i*average_psnr["haar2"][t][thresh_index]/(i+1) + psnr(original_data, haar_2)/(i+1)
                        average_psnr["bi1"][t][thresh_index] = i*average_psnr["bi1"][t][thresh_index]/(i+1) + psnr(original_data, bi1_3)/(i+1)
                        average_psnr["bi1_2"][t][thresh_index] = i*average_psnr["bi1_2"][t][thresh_index]/(i+1) + psnr(original_data, bi1_22)/(i+1)
                        average_psnr["bi2"][t][thresh_index] = i*average_psnr["bi2"][t][thresh_index]/(i+1) + psnr(original_data, bi2_3)/(i+1)
                        average_psnr["bi2_2"][t][thresh_index] = i*average_psnr["bi2_2"][t][thresh_index]/(i+1) + psnr(original_data, bi2_22)/(i+1)
            for key in average_psnr.keys():
                for t in threshold_type:
                    plt.plot(thresholds, average_psnr[key][t], label=t)
                    np.save("average_psnr-"+key+"-"+str(std_dev)+"-"+source+"-"+t, average_psnr[key][t])
                    plt.xlabel("Threshold")
                    plt.ylabel("PSNR")
                plt.legend(loc="upper right")
                plt.savefig(key+"-"+"-"+str(std_dev)+"-"+source+"-psnr-threshold.png")
                plt.close()

                        
##        output2 = haar.forward_periodic(output[0])
##        thresholded2= threshold(output2, calc_threshold(original_data, std_dev))
##        thresholded = threshold(output, calc_threshold(original_data, std_dev))
##        out = haar.backward_periodic(thresholded, False)
##        thresholded[0] = haar.backward_periodic(thresholded2, False)
##        out2 = haar.backward_periodic(thresholded, False)
##        plt.subplot(323)
##        plt.plot(out)
##        plt.subplot(324)
##        plt.plot(out2)
##        print(psnr(original_data, out))
##        print(psnr(original_data, out2))
##        output = bi1.forward_periodic(data)
##        thresholded = threshold(output, calc_threshold(original_data, std_dev))
##        out = bi2.backward_periodic(thresholded, False)
##        plt.subplot(325)
##        plt.plot(out)
##        output = bi2.forward_periodic(data)
##        thresholded = threshold(output, calc_threshold(original_data, std_dev))
##        out = bi1.backward_periodic(output, False)
##        plt.subplot(326)
##        plt.plot(out)
##        plt.show()
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
