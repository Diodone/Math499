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
    def forward_2d(self, multi_array):
        output = []
        for f in self.filters:
            array = None
            for row in multi_array:
                if array is None:
                    array = np.expand_dims(f.forward_periodic(row), axis=0)
                else:
                    array = np.append(array, np.expand_dims(f.forward_periodic(row), axis=0), axis=0)
            for altF in self.filters:
                newArray = None
                for col in array.T:
                    if newArray is None:
                        newArray = np.expand_dims(altF.forward_periodic(col), axis=1)
                    else:
                        newArray = np.append(newArray, np.expand_dims(altF.forward_periodic(col), axis=1), axis=1)
                output.append(newArray)
        return output
    
    def backward_periodic(self, input_arrays, odd):
        output = None
        for index in range(len(input_arrays)):
            if output is None:
                output = self.filters[index].backward_periodic(input_arrays[index], odd)
            else:
                output += self.filters[index].backward_periodic(input_arrays[index], odd)
        return output/2
    def backward_2d(self, input_arrays, odd_rows, odd_col):
        # Has a swapping issue with odd rows/cols. When both even, is correct, and appears to have the correct odd thing as a subset, though that might be coincidence. Use powers of 2 to avoid for now
        output = None
        for index in range(len(input_arrays)):
            array = None
            for col in input_arrays[index].T:
                if array is None:
                    array = np.expand_dims(self.filters[index%2].backward_periodic(col, odd_col), axis=1)
                else:
                    array = np.append(array, np.expand_dims(self.filters[index%2].backward_periodic(col, odd_col), axis=1), axis=1)
            newArray = None
            for row in array:
                if newArray is None:
                    newArray = np.expand_dims(self.filters[index//2].backward_periodic(row, odd_rows), axis=0)
                else:
                    newArray = np.append(newArray, np.expand_dims(self.filters[index//2].backward_periodic(row, odd_rows), axis=0), axis=0)
            # print(newArray)
            if output is None:
                output = newArray
            else:
                output += newArray
        return output/4
def psnr(source, approximation):
    max_value = 255
    mse = np.mean((source-approximation)**2)
    return 20*math.log10(max_value)-10*math.log10(mse)

def add_noise(original, std_dev):
    noise = np.random.normal(scale=std_dev, size=original.shape)
    return original+noise

def threshold(values, std, depth, data_mod=1, epsilon_mod=1, typeStr = "soft"):
    output = []
    if isinstance(typeStr, str):
        if typeStr.lower() == "soft":
            for item in values:
                output.append(soft_threshold(item, std, depth, data_mod, epsilon_mod))
        elif typeStr.lower() == "hard":
            for item in values:
                output.append(hard_threshold(item, std, depth, data_mod, epsilon_mod))
    return output

def hard_threshold(values, std_dev, depth, data_mod, epsilon_mod):
    epsilon = epsilon_mod*calc_threshold(values, std_dev, depth)/math.pow(2, depth/2)
    thresholded = values.copy()
    thresholded[abs(thresholded)<=epsilon] = 0
    return thresholded

def soft_threshold(values, std_dev, depth, data_mod, epsilon_mod):
    epsilon = epsilon_mod*calc_threshold(values, std_dev, depth, data_mod)/math.pow(2, depth/2)
    return np.sign(values) * np.maximum(np.abs(values)-epsilon, 0)

def calc_threshold(data, std_dev, depth, data_mod):
    dev_orig_approx = max(np.var(math.pow(2, data_mod*depth/2)*data)-std_dev**2, 0)
    if dev_orig_approx != 0:
        return std_dev**2/math.sqrt(dev_orig_approx)
    else:
        return abs(data).max()*(math.pow(2, depth/2))

def randomNull(data, percentRemain=1):
    out_data = data.copy()
    random_indecies = np.random.permutation(len(data))
    for i in range(int(len(out_data)*(1-percentRemain))):
        out_data[random_indecies[i]] = 0
    return out_data

def main():
    # Data length, deviation of gaussian noise, datasources, thresholding type, number of trials for each deviation
    data_length = 1024
    std_devs = [1] #np.arange(0.125, 10, 0.125)
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
            for i in range(numRandom):
                added_noise=add_noise(data, std_dev)
                plt.plot(added_noise)
                plt.savefig("Noisy-"+source+"-"+str(std_dev)+"-"+str(i)+".png")
                plt.close()
                # Calculate forward transform for all wavelets
                haar1 = haar.forward_periodic(added_noise)
                haar2 = haar.forward_periodic(haar1[0])
                haar3 = haar.forward_periodic(haar2[0])
                haar4 = haar.forward_periodic(haar3[0])
                haar5 = haar.forward_periodic(haar4[0])
                bi1_1 = bi1.forward_periodic(added_noise)
                bi1_2 = bi1.forward_periodic(bi1_1[0])
                bi1_3 = bi1.forward_periodic(bi1_2[0])
                bi1_4 = bi1.forward_periodic(bi1_3[0])
                bi1_5 = bi1.forward_periodic(bi1_4[0])
                bi2_1 = bi2.forward_periodic(added_noise)
                bi2_2 = bi2.forward_periodic(bi2_1[0])
                bi2_3 = bi2.forward_periodic(bi2_2[0])
                bi2_4 = bi2.forward_periodic(bi2_3[0])
                bi2_5 = bi2.forward_periodic(bi2_4[0])
                # Threshold
                for t in threshold_type:
                    
                    # Technically, the transformed values is \sqrt(2)* the value received from forward. This doesn't change much as a constant multiple can continue though so long as it is kept track of, and indeed my backward takes the \sqrt(2) into account. Threshold values need to be adjusted though, hence the depth parameter
                    haar1_t = threshold(haar1[1:], std_dev, -1, t)
                    haar2_t = threshold(haar2[1:], std_dev, -2, t)
                    bi1_1_t = threshold(bi1_1[1:], thresholds[thresh_index]/math.sqrt(2), t)
                    bi1_2_t = threshold(bi1_2[1:], thresholds[thresh_index]/2, t)
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

def two_d():
    p = pywt.data.camera()
    std = 10
    noisy=add_noise(p, std)
    threshold_type = ["soft", "hard"]
    data_mods = [1, 2, 0.5, 4, 0.25, -1, -2, -0.5, -4, -0.25, 8, 1/8, -8, -1/8]
    epsilon_mods = [1, 1.5, 2, 2/3, 1/2]
    bi1 = Wavelet(([-1/8, 1/4, 3/4, 1/4, -1/8], [-1/4, 1/2, -1/4]), [-2, 0])
    bi2 = Wavelet(([1/4, 1/2, 1/4], [-1/8, -1/4, 3/4, -1/4, -1/8]), [-1, -1])
    t = bi1.forward_2d(noisy)
    t2 = bi1.forward_2d(t[0])
    t3 = bi1.forward_2d(t2[0])
    t4 = bi1.forward_2d(t3[0])
    t5 = bi1.forward_2d(t4[0])
    t6 = bi1.forward_2d(t5[0])
    t7 = bi1.forward_2d(t6[0])
    for ty in threshold_type:
        psnrs = []
        for data_mod in data_mods:
            psnr_inter = []
            for epsilon_mod in epsilon_mods:
                f= t.copy()
                f2 = t2.copy()
                f3 = t3.copy()
                f4 = t4.copy()
                f5 = t5.copy()
                f6 = t6.copy()
                f7 = t7.copy()
                # Threshold
                f[1:] = threshold(f[1:], std, 2, data_mod, epsilon_mod, ty)
                f2[1:] = threshold(f2[1:], std, 4, data_mod, epsilon_mod, ty)
                f3[1:] = threshold(f3[1:], std, 6, data_mod, epsilon_mod, ty)
                f4[1:] = threshold(f4[1:], std, 8, data_mod, epsilon_mod, ty)
                f5[1:] = threshold(f5[1:], std, 10, data_mod, epsilon_mod, ty)
                f6[1:] = threshold(f6[1:], std, 12, data_mod, epsilon_mod, ty)
                f7[1:] = threshold(f7[1:], std, 14, data_mod, epsilon_mod, ty)
                # Reconstruct
                f6[0] = bi2.backward_2d(f7,False, False)
                f5[0] = bi2.backward_2d(f6,False, False)
                f4[0] = bi2.backward_2d(f5,False, False)
                f3[0] = bi2.backward_2d(f4,False, False)
                f2[0] = bi2.backward_2d(f3,False, False)
                f[0] = bi2.backward_2d(f2,False, False)
                g = bi2.backward_2d(f,False, False)
                # Save psnr
                psnr_inter.append(psnr(p, g))
                plt.subplot(121)
                plt.axis('off')
                plt.imshow(p, plt.cm.gray)
                #plt.subplot(122)
                #plt.axis('off')
                #plt.imshow(noisy, plt.cm.gray)
                plt.subplot(122)
                plt.axis('off')
                plt.imshow(g, plt.cm.gray)
                plt.savefig('.png')
                plt.close()
            psnrs.append(psnr_inter)
        array = numpy.array(psnrs)
        for i in range(len(array)):
            plt.plot(epsilon_mods, array[i])
            plt.xlabel("Threshold modifier")
            plt.ylabel("PSNR")
            plt.savefig("data_"+str(datamods[i])+"_camera_"+ty+".png")
        array = array.T
        for i in range(len(array)):
            plt.plot(data_mods, array[i])
            plt.xlabel("Data modifier (2^modifier)")
            plt.ylabel("PSNR")
            plt.savefig("epsilon_"+str(datamods[i])+"_camera_"+ty+".png")
    

#main()
two_d()
