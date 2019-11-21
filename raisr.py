import os
import numba as nb
import numpy as np
import pickle
import tqdm
import multiprocessing
from scipy.ndimage.filters import gaussian_filter

from numba import cuda, float32
#import logging
#logging.getLogger("numba").setLevel(logging.WARNING)

def gradSizes(ar):
    dx, dy, dz = np.gradient(ar)
    g = ((dx**2.0) + (dy**2.0) + (dz ** 2.0)) ** 0.5
    return g

def transformation(x):
    return gradSizes(x)

class RAISR(object):

    def __init__(self,
                 #filter_width=11,
                 filter_width=5,
                 sigma=1.0,
                 num_orientation=5,
                 order=1,
                 num_coherence=5, min_coherence=0.0, max_coherence=1.0,
                 num_strength=5, min_strength=0.0, max_strength=0.07, num_orientation2 = 5,
                 lamda=1e-10, train_flip=False):

        assert filter_width % 2 == 1

        self.filter_width = filter_width
        self.sigma = sigma
        self.order = order
        self.num_orientation = num_orientation
        self.num_orientation2 = num_orientation2
        self.num_coherence = num_coherence
        self.num_strength = num_strength
        self.min_coherence = min_coherence
        self.min_strength = min_strength
        self.max_coherence = max_coherence
        self.max_strength = max_strength
        self.num_filters = num_orientation * num_orientation2 * num_coherence * num_strength #* 3 * 3
        self.lamda = lamda
        self.filters = None
        self.train_flip = train_flip

    def collect_normal_equation(self, image_pair):
        input_image, output_image = image_pair
        print ("A0")
        indices1 = self.calculate_indices(input_image)
        print ("A1_")

        #input_image = transformation(input_image)
        #output_image = transformation(output_image)

        #'''
        num_filters = self.num_filters
        filter_width = int(self.filter_width)
        print (filter_width)
        order = self.order
        filter_size = filter_width**3
        feature_size = 1 + filter_size * order
        #AHA = list(np.zeros((num_filters, feature_size, feature_size), dtype=input_image.dtype))
        #AHy = list(np.zeros((num_filters, feature_size), dtype=input_image.dtype))
        #a_i = list(np.zeros(feature_size, dtype=input_image.dtype))
        AHA = np.zeros((num_filters, feature_size, feature_size), dtype=np.float32)
        #AHA = np.zeros((feature_size, feature_size), dtype=np.float32)
        AHy = np.zeros((num_filters, feature_size), dtype=np.float32)#)
        AHA = np.ascontiguousarray(AHA)
        AHy = np.ascontiguousarray(AHy)
        a_i = np.zeros(feature_size, dtype=np.float32)#)
        a_i = np.ascontiguousarray(a_i)
        indices = np.zeros((320, 250, 250))
        indices = indices1
        indices  = np.ascontiguousarray(indices)
        output_image = np.array(output_image).astype(np.float32)
        #print (AHA[0][0][0])
        output_image = np.ascontiguousarray(output_image) #output_image = np.ascontiguousarray(output_image[100:120, 100:120, 100:120])

        indices = np.ascontiguousarray(indices) #kindices = np.ascontiguousarray(indices[100:120, 100:120, 100:120])

        import time
        #print (time.time())
        #_collect_normal_equation(input_image, output_image, indices,
        #                                num_filters, self.filter_width, self.order, AHA, AHy, a_i)
        #print (time.time())
        #AHA2, AHy2 = _collect_normal_equation(input_image, output_image, indices, num_filters, self.filter_width, self.order)
        #print (AHA2[0].shape)
        #quit()
        #'''
        #print (time.time())
        #print (AHA[0, 0, 0])
        #print (AHA[0, 100, 100])
        #rint (AHy[0, 100])
        #print (AHy[0, 0])
        #print (AHA[1, 100, 100])
        #print (AHy[1, 100])
        #quit()

        #for a in range(0, np.max(indices)+1):
        #    args = np.argwhere(indices==a)
        #    print (a)
        #    print (args.shape)
        #quit()

        time1 = time.time()
        #print (time1)
        for a in range(0, np.max(indices)+1):
            if True:#a == 1:
                args = np.argwhere(indices==a)
                #args = args[:10] #TODO: REMOVE
                args = np.ascontiguousarray(args)
                print (a)
                print (args.shape)
                AHA1 = AHA[a]
                AHy1 = AHy[a]
                #_collect_normal_equation(input_image, output_image, indices,
                #                            num_filters, filter_width, self.order, AHy, AHA[a], a_i, args)
                #bpg = (1,1)
                #tpb = (1, ((filter_width ** 3) + 1)**2)
                numIters = ((((filter_width ** 3) * order) + 1)**2)
                blockDim = (numIters//500 + 1)
                bpg = (1, blockDim)
                tpb = (1, 500)
                stream = cuda.stream()
                with stream.auto_synchronize():
                    #_collect_normal_equation_AHA[bpg, tpb, stream](input_image, indices,
                    #                            num_filters, filter_width, AHA1, order, a_i, args)
                    _collect_normal_equation_AHA[bpg, tpb, stream](input_image,
                                                num_filters, filter_width, AHA1, order, a_i, args)

                AHA[a] = AHA1
                #print ("A")
                #print (filter_width)
                #print (AHA.shape)
                #print (np.mean(AHA[0]))
                #print (np.mean(np.abs(AHA[0] - AHA2[0])))
                #print (AHA[0, 1, 0])
                #print (AHA2[0, 1, 0])
                #print (AHA[0, 10, 10])
                #print (AHA2[0, 10, 10])

                #if a == 0:
                    #print ("A")
                    #print (AHA[0, 100, 100])
                    #quit()
                numIters = ((filter_width ** 3) * order) + 1
                blockDim = (numIters//500 + 1)
                bpg = (1, blockDim)
                tpb = (1, 500)
                stream = cuda.stream()
                with stream.auto_synchronize():
                    #_collect_normal_equation_AHA[bpg, tpb, stream](input_image, indices,
                    #                            num_filters, filter_width, AHA1, order, a_i, args)
                    #_collect_normal_equation_AHA[bpg, tpb, stream](input_image,
                    #                            num_filters, filter_width, AHA1, order, a_i, args)
                    _collect_normal_equation_AHy[bpg, tpb, stream](input_image, output_image,
                                                num_filters, filter_width, AHy1, order, a_i, args)
                AHy[a] = AHy1
                #print (list(AHy[0]))
                #print (list(AHy2[0]))
                #print ("B")
                #print (np.mean(AHy[0]))
                #print (np.mean(np.abs(AHy[0] - AHy2[0])))
                #print (AHy[0, 0])
                #print (AHy2[0, 0])
                #print (AHy[0, -1])
                #print (AHy2[0, -1])
                #print (AHy[0, 10])
                #print (AHy2[0, 10])
                #quit()
                #print (np.mean(AHy[1]))
                #print (np.mean(np.abs(AHy[1] - AHy2[1])))

                #print (np.mean(AHA[0]))
                #print (np.mean(np.abs(AHA[0] - AHA2[0])))
                #print (np.mean(AHA[1]))
                #print (np.mean(np.abs(AHA[1] - AHA2[1])))
                #print (AHy[0].shape)
                #quit()
                #AHA[a] = AHA1
        #'''
        #_collect_normal_equation(np.ascontiguousarray(input_image), np.ascontiguousarray(output_image), np.ascontiguousarray(indices), np.ascontiguousarray(self.num_filters), np.ascontiguousarray(self.filter_width), np.ascontiguousarray(self.order), np.ascontiguousarray(AHA), np.ascontiguousarray(AHy), np.ascontiguousarray(a_i))
        #'''
        return AHA, AHy

    def train(self, input_images, output_images):
        if self.train_flip:
            input_images = (input_images
                            + [np.rot90(i) for i in input_images]
                            + [np.rot90(i, 2) for i in input_images]
                            + [np.rot90(i, 3) for i in input_images]
                            + [np.fliplr(i) for i in input_images]
                            + [np.rot90(np.fliplr(i)) for i in input_images]
                            + [np.rot90(np.fliplr(i), 2) for i in input_images]
                            + [np.rot90(np.fliplr(i), 3) for i in input_images])

            output_images = (output_images
                             + [np.rot90(i) for i in output_images]
                             + [np.rot90(i, 2) for i in output_images]
                             + [np.rot90(i, 3) for i in output_images]
                             + [np.fliplr(i) for i in output_images]
                             + [np.rot90(np.fliplr(i)) for i in output_images]
                             + [np.rot90(np.fliplr(i), 2) for i in output_images]
                             + [np.rot90(np.fliplr(i), 3) for i in output_images])

        AHA = 0
        AHy = 0
        #print (args.filter)
        print ("B")
        #self.calculate_structure_tensor(input_images[0])
        #self.calculate_indices(input_images[0])
        #print (input_images[0].shape)
        #self.collect_normal_equation([input_images[0], output_images[0]])
        #print ("New Version")
        #quit()
        #quit()
        #zip(input_images, output_images)
        for i in range(0, len(input_images)):
            #'''
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("NUMBER --------------------------------------------------------------")
            print (i)
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            print ("")
            #'''
            #print (input_images[i].shape)
            #quit()
            AHA1, AHy1 = self.collect_normal_equation([input_images[i], output_images[i]])

            AHA += AHA1
            AHy += AHy1

            #AHA += self.lamda * np.eye(AHA.shape[-1])
            self.filters = np.linalg.solve(AHA + (self.lamda * np.eye(AHA.shape[-1])), AHy)
            self.save('filter.raisr_Nov14_w5_2')
            #quit()
        #print ("E")
        #AHA += self.lamda * np.eye(AHA.shape[-1])
        #self.filters = np.linalg.solve(AHA, AHy)


    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        #filename = './' + filename + '.npy'
        #filters = np.array(filename, self.filters)
        #np.save(filters)

    def load(self, filename):
        filename = './' + filename + '.npy'
        filters = np.load(filename)
        self.filters = filters
        return self


    def filter(self, input_image):
        #if self.filters is None:
        #    raise ValueError("No filters! Please train on images first!")

        indices = self.calculate_indices(input_image)

        '''
        indices = np.ndarray.flatten(indices)
        ar, counts = np.unique(indices, return_counts=True)
        print ("HI")
        import matplotlib.pyplot as plt
        plt.plot(ar, counts)#np.sort(counts))
        plt.show()
        quit()
        '''
        #input_image = transformation(input_image)
        return _filter(input_image, indices, self.filters, self.filter_width, self.order)

    def batch_filter(self, input_images):
        output_images = []
        with tqdm.tqdm(total=len(input_images)) as pbar:
            with multiprocessing.Pool(min(os.cpu_count(), len(input_images))) as p:
                for i, res in tqdm.tqdm(enumerate(p.imap(self.filter, input_images))):
                    output_images.append(res)
                    pbar.update()


        return output_images

    def quantize(self, orientation1, orientation2, strength, coherence, angle3, strength2):
        orientation1 = _quantize(orientation1, 0, 2.0*np.pi, self.num_orientation)
        orientation2 = _quantize(orientation2, 0, np.pi, self.num_orientation2)
        strength = _quantize(strength, self.min_strength, self.max_strength, self.num_strength)
        coherence = _quantize(coherence, self.min_coherence, self.max_coherence, self.num_coherence)
        #angle3 = _quantize(angle3, 0, np.pi, 3)
        #strength2 = _quantize(strength2, 0, 0.025, 3)

        return orientation1, orientation2, strength, coherence, angle3, strength2

    def calculate_quantized_structure_tensor(self, input_image):
        orientation1, orientation2, angle3, strength1, coherence, strength2 = self.calculate_structure_tensor(input_image)

        #import matplotlib.pyplot as plt
        #angle3 = np.ndarray.flatten(strength2)
        #strength1 = ((strength1 / self.max_strength) ** 0.5) * self.max_strength

        #strength2[strength2>0.025] = 0.025


        #strength2 = ((strength2 / 0.025)**0.6) * 0.025


        #plt.hist(np.ndarray.flatten(angle3), bins=100)
        #plt.show()
        #quit()
        #print (np.min(angle1))
        #print (np.max(angle1))
        #print (np.min(angle2))
        #print (np.max(angle2))
        #quit()
        #print (np.min(strength1))
        #print (np.max(strength1))
        #print (np.argwhere(np.ndarray.flatten(strength1) > 0.07).shape)
        orientation1, orientation2, strength1, coherence, angle3, strength2 = self.quantize(orientation1, orientation2, strength1, coherence, angle3, strength2)
        return orientation1, orientation2, strength1, coherence, angle3, strength2

    def stefan_calculate_indices(self, input_image):
        angle1, angle2, angle3, strength1, coherence, strength2 = self.calculate_quantized_structure_tensor(input_image)
        #I am throwing away "angle3" and "strength2" for now to prevent their being too many keys
        #indices = orientation + self.num_orientation * strength + self.num_strength * coherence
        indices = angle1, angle2, angle3, strength1, coherence, strength2

        return indices

    def calculate_indices(self, input_image):
        orientation1, orientation2, strength, coherence, angle3, strength2 = self.calculate_quantized_structure_tensor(input_image)
        indices = orientation1 + self.num_orientation * (orientation2 + self.num_orientation2 * (strength + self.num_strength * coherence))

        #import matplotlib.pyplot as plt
        '''
        stats = [orientation1, orientation2, strength, coherence]
        for a in range(0, 4):
            stats[a] = stats[a][strength==2]
            stats[a] = np.ndarray.flatten(stats[a])
            unique, count = np.unique(stats[a], return_counts=True)
            plt.plot(unique, count)
        plt.show()
        '''
        #plt.plot(ar, counts)#np.sort(counts))
        #plt.show()
        #quit()

        #indices = orientation1 + self.num_orientation * (orientation2 + self.num_orientation2 * (strength + self.num_strength * (coherence + 3 * (angle3 + 3 * (strength2)) )))
        return indices

    def calculate_structure_tensor(self, input_image):
        dx, dy, dz = np.gradient(input_image)
        #dx, _, _ = np.gradient(dx)
        #_, dy, _ = np.gradient(dy)
        #_, _, dz = np.gradient(dz)
        #print ("2nd dir")

        dv = [dx, dy, dz]
        matrix = [[[], [], []], [[], [], []], [[], [], []]]
        for a in range(0, 3):
            for b in range(0, a+1):
                matrix[a][b] = gaussian_filter(dv[a] * dv[b], self.sigma)

                #TODO CHANGE TO NORMAL!
                #img1 = gaussian_filter(dv[a] * dv[b], 0.5)
                #if scale == 0.5:
                #img1 = img1 - (dv[a] * dv[b] * (0.48664 - 0.06349))
                #img1 = img1 * (1.0/(1.0 - (0.48664 - 0.06349)))
                #matrix[a][b] = img1

                matrix[b][a] = matrix[a][b]
        matrix = np.array(matrix)
        matrix = np.swapaxes(matrix, 0, 3)
        matrix = np.swapaxes(matrix, 1, 4)
        #dxx = gaussian_filter(dx**2, self.sigma)
        #dxy = gaussian_filter(dx * dy, self.sigma)
        #dyy = gaussian_filter(dy**2, self.sigma)
        #trace = dxx + dyy
        #det = dxx * dyy - dxy**2

        eigVal, eigVec = np.linalg.eigh(matrix)
        eigVal = eigVal ** 0.5
        vec1 = eigVec[:, :, :, :, 0]
        vec2 = eigVec[:, :, :, :, 1]
        angle1 = np.arctan2(vec1[:, :, :, 0], vec1[:, :, :, 1]) + np.pi
        xyTotalVec1 = ((vec1[:, :, :, 0] ** 2.0) + (vec1[:, :, :, 1] ** 2.0)) ** 0.5
        angle2 = np.arctan2(xyTotalVec1, vec1[:, :, :, 2])

        #print (np.min(angle1))
        #print (np.max(angle1))
        #quit()

        '''
        shape1 = angle1.shape
        angle1 = np.ndarray.flatten(angle1)
        angle2 = np.ndarray.flatten(angle2)
        args = np.argwhere(angle1 > np.pi)
        args = np.squeeze(args)

        angle1[args] = angle1[args] % np.pi
        angle2[args] = np.pi - angle2[args]
        '''

        #angle1 = angle1 * np.sin(angle2)

        #import matplotlib.pyplot as plt
        #plt.scatter(angle1[0::10000], angle2[0::10000])
        #plt.show()
        #quit()

        #angle1 = angle1.reshape(shape1)
        #angle2 = angle2.reshape(shape1)





        xProj = np.zeros(vec1.shape)
        yProj = np.zeros(vec1.shape)
        xProj[:, :, :, 0] = 1.0
        yProj[:, :, :, 1] = 1.0
        for a in range(0, 3): #This loop is a bit dumb but it's the easiest way to program it I thought of.
            xProj[:, :, :, a] = xProj[:, :, :, a] - (vec1[:, :, :, 0] * vec1[:, :, :, a])
            yProj[:, :, :, a] = yProj[:, :, :, a] - (vec1[:, :, :, 1] * vec1[:, :, :, a])
        vec2x = np.sum(xProj * vec2, axis=3)
        vec2y = np.sum(yProj * vec2, axis=3)
        angle3 = np.arctan2(vec2x, vec2y)
        angle3 = angle3 % np.pi

        strength1 = eigVal[:, :, :, 2]
        strength2 = eigVal[:, :, :, 1]
        strength3 = eigVal[:, :, :, 0]
        coherence = (strength1 -  (0.5 * (strength2 + strength3))) / (strength1 +  (0.5 * (strength2 + strength3)))
        #print (angle1[100, 100, 100], angle2[100, 100, 100], angle3[100, 100, 100])
        #coherence2 = (strength2 - strength3) / (strength2 + strength3)
        #quit()
        outputs = [angle1, angle2, angle3, strength1, coherence, strength2]

        for a in range(0, len(outputs)):
            outputs[a] = np.swapaxes(outputs[a], 1, 2)
            outputs[a] = np.swapaxes(outputs[a], 0, 1)

        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]

    def reshape_filters(self):
        return self.filters.reshape([self.num_coherence, self.num_strength, self.num_orientation,
                                     self.filter_width, self.filter_width])


def _quantize(x, min_x, max_x, num_x):
    return np.round(np.clip((x - min_x) / (max_x - min_x) * num_x, 0.5,
                            num_x - 0.5) - 0.5).astype(np.int)

#'''
@nb.jit(nopython=True, cache=True)
def _collect_normal_equation(input_image, output_image, indices, num_filters, filter_width, order):
    filter_size = filter_width**3
    feature_size = 1 + filter_size * order
    #print (np.outer(np.ones(10), np.ones(10)))
    AHA = np.zeros((num_filters, feature_size, feature_size))
    AHy = np.zeros((num_filters, feature_size))
    #AHA = np.empty((num_filters, feature_size, feature_size))
    #AHy = np.empty((num_filters, feature_size))

    for x in range(input_image.shape[0]):
        for y in range(input_image.shape[1]):
            for z in range(0, input_image.shape[2]): #Stefan

                if (indices[x, y, z] <= 1):# and (indices[x, y, z] >= 40):
                    # Get patch, or equaivalantly, one row of A
                    a_i = np.empty(feature_size, dtype=np.float64)
                    #a_i = []
                    #for a in range(0, feature_size):
                    #   a_i.append(0.0)
                    c = 0

                    # Zeroth term
                    a_i[c] = 1
                    c += 1

                    # Higher order term
                    for d in range(1, order + 1):
                        for p in range(x - filter_width // 2, x + filter_width // 2 + 1):
                            for q in range(y - filter_width // 2, y + filter_width // 2 + 1):
                                for r in range(z - filter_width // 2, z + filter_width // 2 + 1):
                                    a_i[c] = input_image[p % input_image.shape[0], q % input_image.shape[1], r % input_image.shape[2]]**d
                                    c += 1

                    # Get y_j
                    y_i = output_image[x, y, z]

                    # Get index
                    idx = indices[x, y, z]

                    # Accumulate AHA, and AHy
                    a_i_conj = np.conj(a_i)
                    AHA[idx] += np.outer(a_i, a_i_conj)

                    AHy[idx] += a_i_conj * y_i
    return AHA, AHy
#'''

'''
@cuda.jit()
#@nb.jit(nopython=True, cache=True)
def _collect_normal_equation(input_image, output_image, indices, num_filters, filter_width, order, AHA, AHy, a_i):
    filter_size = filter_width**3
    feature_size = 1 + filter_size * order
    #print (np.outer(np.ones(10), np.ones(10)))
    #AHA = np.zeros((num_filters, feature_size, feature_size))
    #AHy = np.zeros((num_filters, feature_size))
    #AHA = np.empty((num_filters, feature_size, feature_size))
    #AHy = np.empty((num_filters, feature_size))

    for x in range(input_image.shape[0]): #TODO CHANGE THIS ONCE MEMORY ISSUE IS SORTED OUT!
        for y in range(input_image.shape[1]):
            for z in range(0, input_image.shape[2]): #Stefan
                # Get patch, or equaivalantly, one row of A
                #a_i = np.empty(feature_size, dtype=np.float64)
                #a_i = []
                #for a in range(0, feature_size):
                #   a_i.append(0.0)
                c = 0

                # Zeroth term
                a_i[c] = 1
                c += 1

                # Higher order term
                for d in range(1, order + 1):
                    for p in range(x - filter_width // 2, x + filter_width // 2 + 1):
                        for q in range(y - filter_width // 2, y + filter_width // 2 + 1):
                            for r in range(z - filter_width // 2, z + filter_width // 2 + 1):
                                a_i[c] = input_image[p % input_image.shape[0], q % input_image.shape[1], r % input_image.shape[2]]**d
                                c += 1

                # Get y_j
                y_i = output_image[x, y, z]

                # Get index
                idx = indices[x, y, z]

                # Accumulate AHA, and AHy
                #a_i_conj = np.conj(a_i)
                #AHA[idx] += np.outer(a_i, a_i_conj)
                #AHy[idx] += a_i_conj * y_i
                for a in range(0, len(AHA[idx])):
                    for b in range(0, len(AHA[idx][0])):
                        AHA[idx] +=  a_i[a] * a_i[b]
                    AHy[idx][a] += a_i[a] * y_i

    #return 0.0#AHA, AHy
#'''


'''
@cuda.jit
def _collect_normal_equation(input_image, output_image, indices, num_filters, filter_width, order, AHy, AHA, a_i, args):
    arg = cuda.grid(1)
    #if x < an_array.shape[0] and y < an_array.shape[1]:
    #   an_array[x, y] += 1
    #if x < input_image.shape[0] and y < input_image.shape[1] and z < input_image.shape[2]:
    if arg < args.shape[0]:
        [x, y, z] = args[arg]
        c = 0
        # Zeroth term
        a_i[c] = 1
        c += 1

        for d in range(1, order + 1):
            for p in range(x - filter_width // 2, x + filter_width // 2 + 1):
                for q in range(y - filter_width // 2, y + filter_width // 2 + 1):
                    for r in range(z - filter_width // 2, z + filter_width // 2 + 1):
                        a_i[c] = input_image[p % input_image.shape[0], q % input_image.shape[1], r % input_image.shape[2]]**d
                        c += 1

        # Get y_j
        y_i = output_image[x, y, z]

        # Get index
        idx = indices[x, y, z]

        # Accumulate AHA, and AHy
        #a_i_conj = np.conj(a_i)
        #AHA[idx] += np.outer(a_i, a_i_conj)
        #AHy[idx] += a_i_conj * y_i
        for a in range(0, len(AHy[idx])):
            for b in range(0, len(AHy[idx])):
                val = a_i[a] * a_i[b]
                if val > 0.001:
                    AHA[a][b] += val
                #AHA[a][b] += (a_i[a] * a_i[b])
                #fake1 = 1
                #AHA =  a_i[a] * a_i[b]
            #if a_i[a] * y_i > 0.1:
            #    AHy[idx][a] += a_i[a] * y_i
        if arg < 10:#88695 - 5:
            AHy[:, :] = 1000

'''

'''
@cuda.jit
def _collect_normal_equation(input_image, output_image, indices, num_filters, filter_width, order, AHy, AHA, a_i, args):
    #p0, q0, r0 = cuda.grid(3)

    ty = cuda.threadIdx.y
    bh = cuda.blockDim.y
    #num_1 = (((filter_width**3)+1))
    #num_2 = (((filter_width**3)+1))

    AHA2 = cuda.shared.array(((5**3)+1, (5**3)+1), float32)
    ##
    #AHA2 = cuda.shared.array(((filter_width**3)+1, (filter_width**3)+1), float32)


    #if x < an_array.shape[0] and y < an_array.shape[1]:
    #   an_array[x, y] += 1
    num_filters2 = filter_width

    filter_width * 2

    filter_width ** 2

    num_filters3 = (num_filters2 ** 2) + 1

    p0  = ty % num_filters3
    ty = ty // num_filters3
    q0 = ty % num_filters3
    r0 = ty // num_filters3

    if p0 < num_filters3 and q0 < num_filters3 and r0 < num_filters3:
        #ar1 = cuda.shared.array((1), float32)
        #AHA2 = cuda.shared.array(((filter_width**3)+1, (filter_width**3)+1), float32)
        p1 = (p0 % num_filters2)
        p2 = (p0 // num_filters2)
        q1 = (q0 % num_filters2)
        q2 = (q0 // num_filters2)
        r1 = (r0 % num_filters2)
        r2 = (r0 // num_filters2)
        p_1 = p1 - (num_filters2 // 2)
        p_2 = p1 - (num_filters2 // 2)
        q_1 = q1 - (num_filters2 // 2)
        q_2 = q2 - (num_filters2 // 2)
        r_1 = r1 - (num_filters2 // 2)
        r_2 = r2 - (num_filters2 // 2)
        num1 = p1 + num_filters2 * ( q1 + (num_filters2 * r1))
        num2 = p2 + num_filters2 * ( q2 + (num_filters2 * r2))
        if p0 == 1 and q0 ==0 and r0 == 0:#(num1 == 1) and (num2 == 0):
            AHA2[:, :] = 100
        val0 = 0
        for a in range(0, args.shape[0]):
            [x, y, z] = args[a]
            val1 = input_image[(x + p_1) % input_image.shape[0], (y + q_1) % input_image.shape[1], (z + r_1) % input_image.shape[2]]
            val2 = input_image[(x + p_2) % input_image.shape[0], (y + q_2) % input_image.shape[1], (z + r_2) % input_image.shape[2]]
            #AHA[num1, num2] = AHA[num1, num2] + (val1 * val2)
        #AHA[num1, num2] = val0

    cuda.syncthreads()
    #AHA = AHA2
'''

'''
@cuda.jit
def _collect_normal_equation(input_image, output_image, indices, num_filters, filter_width, order, AHy, AHA, a_i, args):
    #p0, q0, r0 = cuda.grid(3)

    ty = cuda.threadIdx.y
    bh = cuda.blockDim.y

    #size1 = (filter_width**3)+1
    #n1 = ty % size1
    #n2 = ty // size1
    #num_1 = (((filter_width**3)+1))
    #num_2 = (((filter_width**3)+1))
    AHA2 = cuda.shared.array(((5**3)+1, (5**3)+1), float32)
    ##
    #AHA2 = cuda.shared.array(((filter_width**3)+1, (filter_width**3)+1), float32)


    #if x < an_array.shape[0] and y < an_array.shape[1]:
    #   an_array[x, y] += 1
    num_filters2 = filter_width

    filter_width * 2

    filter_width ** 2

    num_filters3 = (num_filters2 ** 2) + 1

    p0  = ty % num_filters3
    ty = ty // num_filters3
    q0 = ty % num_filters3
    r0 = ty // num_filters3

    if p0 < num_filters3 and q0 < num_filters3 and r0 < num_filters3:
        #ar1 = cuda.shared.array((1), float32)
        #AHA2 = cuda.shared.array(((filter_width**3)+1, (filter_width**3)+1), float32)
        p1 = (p0 % num_filters2)
        p2 = (p0 // num_filters2)
        q1 = (q0 % num_filters2)
        q2 = (q0 // num_filters2)
        r1 = (r0 % num_filters2)
        r2 = (r0 // num_filters2)
        p_1 = p1 - (num_filters2 // 2)
        p_2 = p1 - (num_filters2 // 2)
        q_1 = q1 - (num_filters2 // 2)
        q_2 = q2 - (num_filters2 // 2)
        r_1 = r1 - (num_filters2 // 2)
        r_2 = r2 - (num_filters2 // 2)
        num1 = p1 + num_filters2 * ( q1 + (num_filters2 * r1))
        num2 = p2 + num_filters2 * ( q2 + (num_filters2 * r2))
        if p0 == 1 and q0 ==0 and r0 == 0:#(num1 == 1) and (num2 == 0):
            AHA2[:, :] = 100
        val0 = 0
        for a in range(0, args.shape[0]):
            [x, y, z] = args[a]
            val1 = input_image[(x + p_1) % input_image.shape[0], (y + q_1) % input_image.shape[1], (z + r_1) % input_image.shape[2]]
            val2 = input_image[(x + p_2) % input_image.shape[0], (y + q_2) % input_image.shape[1], (z + r_2) % input_image.shape[2]]
            #AHA[num1, num2] = AHA[num1, num2] + (val1 * val2)
        #AHA[num1, num2] = val0

    cuda.syncthreads()
'''

@cuda.jit
def _collect_normal_equation_AHA(input_image, num_filters, filter_width2, AHA, order, a_i, args):
    #p0, q0, r0 = cuda.grid(3)

    ty = cuda.threadIdx.y
    count2 = cuda.blockIdx.y
    filter_width = filter_width2#5
    bh = cuda.blockDim.y
    count = ty + (count2 * bh)

    size1 = ((filter_width**3)*order)+1
    if count < (size1**2):
        #AHA2 = cuda.shared.array((126, 63), float32)
        #size1 = (filter_width**3)+1
        n1 = count % size1
        n2 = count // size1

        n_1 = n1 - 1
        n_2 = n2 - 1

        r1 = n_1 % filter_width
        n_1 = n_1 // filter_width
        q1 = n_1 % filter_width
        n_1 = n_1 // filter_width
        p1 = n_1 % filter_width
        o1 = (n_1 // filter_width) + 1

        r2 = n_2 % filter_width
        n_2 = n_2 // filter_width
        q2 = n_2 % filter_width
        n_2 = n_2 // filter_width
        p2 = n_2 % filter_width
        o2 = (n_2 // filter_width) + 1

        p_1 = p1 - (filter_width // 2)
        p_2 = p2 - (filter_width // 2)
        q_1 = q1 - (filter_width // 2)
        q_2 = q2 - (filter_width // 2)
        r_1 = r1 - (filter_width // 2)
        r_2 = r2 - (filter_width // 2)
        #if count == 0:
        #AHA[n1, n2] = 1.0
        val3 = 0.0
        for a in range(0, args.shape[0]):#[1, 2, 3, 4]:#
            [x, y, z] = args[a]

            val1 = 1.0
            val2 = 1.0
            if n1 != 0:
                val1 = input_image[(x + p_1) % input_image.shape[0], (y + q_1) % input_image.shape[1], (z + r_1) % input_image.shape[2]]
                val1 = val1 ** o1
            if n2 != 0:
                val2 = input_image[(x + p_2) % input_image.shape[0], (y + q_2) % input_image.shape[1], (z + r_2) % input_image.shape[2]]
                val2 = val2 ** o2
            val3 = val3 + (val1 * val2)
            #if n1 == 100 and n2 == 100:
            #    print (val3)
            #    print (val1)
            #    print (val2)
            #    print ((x + p_1) % input_image.shape[0], (y + q_1) % input_image.shape[1], (z + r_1) % input_image.shape[2])
            #    print ((x + p_2) % input_image.shape[0], (y + q_2) % input_image.shape[1], (z + r_2) % input_image.shape[2])

        #cuda.syncthreads()
        AHA[n1, n2] = val3

@cuda.jit
def _collect_normal_equation_AHy(input_image, output_image, num_filters, filter_width2, AHy, order, a_i, args):
    #p0, q0, r0 = cuda.grid(3)

    ty = cuda.threadIdx.y
    count2 = cuda.blockIdx.y
    filter_width = filter_width2#5
    bh = cuda.blockDim.y
    count = ty + (count2 * bh)

    size1 = ((filter_width**3)*order)+1
    if count < size1:
        #AHA2 = cuda.shared.array((126, 63), float32)
        size1 = (filter_width**3)+1
        n1 = count

        n_1 = n1 - 1

        r1 = n_1 % filter_width
        n_1 = n_1 // filter_width
        q1 = n_1 % filter_width
        n_1 = n_1 // filter_width
        p1 = n_1 % filter_width
        o1 = (n_1 // filter_width) + 1

        #r1 = n_1 % filter_width
        #n_1 = n_1 // filter_width
        #q1 = n_1 % filter_width
        #p1 = n_1 // filter_width

        p_1 = p1 - (filter_width // 2)
        q_1 = q1 - (filter_width // 2)
        r_1 = r1 - (filter_width // 2)

        val2 = 0.0
        for a in range(0, args.shape[0]):#[1, 2, 3, 4]:#
            [x, y, z] = args[a]
            val1 = 1.0
            if n1 != 0:
                val1 = input_image[(x + p_1) % input_image.shape[0], (y + q_1) % input_image.shape[1], (z + r_1) % input_image.shape[2]]
            val1 = val1 ** o1
            val2 = val2 + (val1 * output_image[x, y, z])

        #cuda.syncthreads()
        AHy[n1] = val2

'''
@cuda.jit
def _collect_normal_equation(input_image, output_image, indices, num_filters, filter_width, order, AHA, AHy, a_i):
    x, y1, z1 = cuda.grid(3)
    #if x < an_array.shape[0] and y < an_array.shape[1]:
    #   an_array[x, y] += 1
    if x < input_image.shape[0] and y1 < (input_image.shape[1] * filter_width) and z1 < (input_image.shape[2] * filter_width):
        z = z1 % input_image.shape[2]
        p1 = z1 // filter_width
        y = y1 % input_image.shape[1]
        q1 = y1 // filter_width
        c = 0
        # Zeroth term
        a_i[c] = 1
        c += 1
        for d in range(1, order + 1):
            #for p in range(x - filter_width // 2, x + filter_width // 2 + 1):
            p = x - (filter_width // 2) + p1
            q = y - (filter_width // 2) + q1
            #for q in range(y - filter_width // 2, y + filter_width // 2 + 1):
            for r in range(z - filter_width // 2, z + filter_width // 2 + 1):
                a_i[c] = input_image[p % input_image.shape[0], q % input_image.shape[1], r % input_image.shape[2]]**d
                c += 1

        # Get y_j
        y_i = output_image[x, y, z]

        # Get index
        idx = indices[x, y, z]

        # Accumulate AHA, and AHy
        #a_i_conj = np.conj(a_i)
        #AHA[idx] += np.outer(a_i, a_i_conj)
        #AHy[idx] += a_i_conj * y_i
        for a in range(0, len(AHA[idx])):
            for b in range(0, len(AHA[idx][0])):
                AHA[idx] =  a_i[a] * a_i[b]
            AHy[idx][a] = a_i[a] * y_i

'''

@nb.jit(nopython=True, cache=True)
def _filter(input_image, indices, filters, filter_width, order):
    filter_size = filter_width**3
    feature_size = 1 + filter_size * order

    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)
    for x in range(input_image.shape[0]):
        for y in range(input_image.shape[1]):
            for z in range(input_image.shape[2]):

                # Get patch, or equaivalantly, one row of A
                a_i = np.empty(feature_size, dtype=input_image.dtype)
                c = 0

                # Zeroth term
                a_i[c] = 1
                c += 1

                # Higher order term
                for d in range(1, order + 1):
                    for p in range(x - filter_width // 2, x + filter_width // 2 + 1):
                        for q in range(y - filter_width // 2, y + filter_width // 2 + 1):
                            for r in range(z - filter_width // 2, z + filter_width // 2 + 1):
                                a_i[c] = input_image[p % input_image.shape[0], q % input_image.shape[1], r % input_image.shape[2]]**d
                                c += 1

                # Get filter
                filt = filters[indices[x, y, z]]

                # Dot product with filter
                output_image[x, y, z] = np.vdot(a_i, filt)

    return output_image
