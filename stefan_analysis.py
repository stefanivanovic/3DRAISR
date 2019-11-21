#stefan_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import sigpy as sp
import pickle

def calculate_structure_tensor(input_image):
    dx, dy, dz = np.gradient(input_image)
    #dx, _, _ = np.gradient(dx)
    #_, dy, _ = np.gradient(dy)
    #_, _, dz = np.gradient(dz)

    dv = [dx, dy, dz]
    matrix = [[[], [], []], [[], [], []], [[], [], []]]
    for a in range(0, 3):
        for b in range(0, a+1):
            matrix[a][b] = gaussian_filter(dv[a] * dv[b], 2.0)
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
    vec2 = eigVec[:, :, :, :, 0]
    angle1 = np.arctan2(vec1[:, :, :, 0], vec1[:, :, :, 1]) + np.pi
    xyTotalVec1 = ((vec1[:, :, :, 0] ** 2.0) + (vec1[:, :, :, 1] ** 2.0)) ** 0.5
    angle2 = np.arctan2(xyTotalVec1, vec1[:, :, :, 2])

    xProj = np.zeros(vec1.shape)
    yProj = np.zeros(vec1.shape)
    xProj[:, :, :, 0] = 1.0
    yProj[:, :, :, 0] = 1.0
    for a in range(0, 3): #This loop is a bit dumb but it's the easiest way to program it I thought of.
        xProj[:, :, :, a] = xProj[:, :, :, a] - (vec1[:, :, :, 0] * vec1[:, :, :, a])
        yProj[:, :, :, a] = yProj[:, :, :, a] - (vec1[:, :, :, 1] * vec1[:, :, :, a])
    vec2x = np.sum(xProj * vec2, axis=3)
    vec2y = np.sum(yProj * vec2, axis=3)
    angle3 = np.arctan2(vec2x, vec2y)

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

def calculate_structure_tensor_2D(input_image):
    dx, dy = np.gradient(input_image)
    sigma = 1.0
    dxx = gaussian_filter(dx**2, sigma)
    dxy = gaussian_filter(dx * dy, sigma)
    dyy = gaussian_filter(dy**2, sigma)
    trace = dxx + dyy
    det = dxx * dyy - dxy**2

    lamda_1 = trace / 2 + (trace**2 / 4 - det)**0.5
    lamda_2 = trace / 2 - (trace**2 / 4 - det)**0.5

    w1 = dxy
    w2 = lamda_1 - dxx
    orientation = np.arctan2(w2, w1)
    strength = lamda_1**0.5
    coherence = (lamda_1**0.5 - lamda_2**0.5) / (lamda_1**0.5 + lamda_2**0.5)

    return orientation, strength, coherence

def stefan2D(input_image):
    dx, dy = np.gradient(input_image)
    sigma = 1.0
    dxx = gaussian_filter(dx**2, sigma)
    dxy = gaussian_filter(dx * dy, sigma)
    dyy = gaussian_filter(dy**2, sigma)
    #trace = dxx + dyy
    #det = dxx * dyy - dxy**2

    #lamda_1 = trace / 2 + (trace**2 / 4 - det)**0.5
    #lamda_2 = trace / 2 - (trace**2 / 4 - det)**0.5

    w1 = dxy
    w2 = lamda_1 - dxx
    orientation = np.arctan2(w2, w1)
    strength = lamda_1**0.5
    coherence = (lamda_1**0.5 - lamda_2**0.5) / (lamda_1**0.5 + lamda_2**0.5)

    return orientation, strength, coherence

def plotField(strength, orientation):
    X = []
    Y = []
    U = []
    V = []
    for a in range(0, strength.shape[0]):
        #a = strength.shape[0] - 1 - a1
        for b in range(0, strength.shape[1]):
            #b = strength.shape[1] - 1 - b1
            X.append(a)
            Y.append(b)
            u = -np.cos(orientation[a][b]) * strength[a][b]
            v = np.sin(orientation[a][b]) * strength[a][b]
            U.append(u)
            V.append(v)
    plt.quiver(Y, X, V, U)
    #plt.colorbar()
    plt.show()

def find2ndGrad(img):
    def findSingleGrad(img, dir):
        args = np.argwhere(np.abs(img) > -1.0)
        args1 = np.copy(args)
        args2 = np.copy(args)
        args1[:, dir] = (args1[:, dir] - 1) % img.shape[dir]
        args2[:, dir] = (args2[:, dir] + 1) % img.shape[dir]

        img2 = 0.5 * (img[args1[:, 0], args1[:, 1], args1[:, 2]] + img[args2[:, 0], args2[:, 1], args2[:, 2]]).reshape(img.shape)  - img
        img2 = img2 ** 2.0
        return img2

    img2 = (findSingleGrad(img, 0) + findSingleGrad(img, 1) + findSingleGrad(img, 2)) ** 0.5
    return img2

def find2ndGrad2D(img):
    def findGradAvg2D(img, dir):
        args = np.argwhere(np.abs(img) > -1.0)
        args1 = np.copy(args)
        args2 = np.copy(args)
        args1[:, dir] = (args1[:, dir] - 1) % img.shape[dir]
        args2[:, dir] = (args2[:, dir] + 1) % img.shape[dir]
        img2 =  ((img[args1[:, 0], args1[:, 1]] + img[args2[:, 0], args2[:, 1]]).reshape(img.shape)  + img) / 3.0
        return img

    def findSingleGrad2D(img, dir):
        args = np.argwhere(np.abs(img) > -1.0)
        args1 = np.copy(args)
        args2 = np.copy(args)
        args1[:, dir] = (args1[:, dir] - 1) % img.shape[dir]
        args2[:, dir] = (args2[:, dir] + 1) % img.shape[dir]

        img2 = 0.5 * (img[args1[:, 0], args1[:, 1]] + img[args2[:, 0], args2[:, 1]]).reshape(img.shape)  - img
        #img2 = np.abs(img2)
        return img2


    img2 = ((findGradAvg2D(findSingleGrad2D(img, 0), 1) ** 2.0) + (findGradAvg2D(findSingleGrad2D(img, 1), 0) ** 2.0)) ** 0.5
    return img2

def doubleShow(imgs, axis, slice=-500):
    if axis == 1:
        if slice == -500:
            slice = 200
    else:
        if slice == -500:
            slice = 100
    for a in range(0, len(imgs)):
        imgs[a] = np.swapaxes(imgs[a], 1, 2)

    list2 = [0, 1, 2]
    list2.remove(axis)
    funnyImg = np.zeros((imgs[0].shape[list2[0]], imgs[0].shape[list2[1]]*len(imgs)))
    for a in range(0, len(imgs)):
        if axis == 0:
            img1 = imgs[a][slice, :, :]
        elif axis == 1:
            img1 = imgs[a][:, slice, :]
        else:
            img1 = imgs[a][:, :, slice]
        funnyImg[:, imgs[0].shape[list2[1]]*a:imgs[0].shape[list2[1]]*(a+1)] = img1
    funnyImg[funnyImg<0.0] = 0.0
    funnyImg[funnyImg>1.0] = 1.0
    plt.imshow(funnyImg, cmap='gray')
    plt.show()

    '''
    if axis == 1:
        if slice == -500:
            slice = 100
        funnyImg = np.zeros((img.shape[0], img.shape[2]*2))
        funnyImg[:, :img.shape[2]]  = img[:, slice, :]
        funnyImg[:, img.shape[2]:]  = img2[:, slice, :]
        funnyImg[funnyImg<0.0] = 0.0
        funnyImg[funnyImg>1.0] = 1.0
        plt.imshow(funnyImg, cmap='gray')
        plt.show()
    elif axis == 0:
        if slice == -500:
            slice = 200
        funnyImg = np.zeros((img.shape[0], img.shape[1]*2))
        funnyImg[:, :img.shape[1]]  = img[:, :, slice]
        funnyImg[:, img.shape[1]:]  = img2[:, :, slice]
        funnyImg[funnyImg<0.0] = 0.0
        funnyImg[funnyImg>1.0] = 1.0
        plt.imshow(funnyImg, cmap='gray')
        plt.show()
    elif axis == 2:
        if slice == -500:
            slice = 100
        img = np.swapaxes(img, 1, 2)
        img2 = np.swapaxes(img2, 1, 2)
        #img3 = np.swapaxes(img3, 1, 2)
        funnyImg = np.zeros((img.shape[1], img.shape[2]*2))
        funnyImg[:, :img.shape[2]]  = img[slice, :, :]
        funnyImg[:, img.shape[2]:]  = img2[slice, :, :]
        #funnyImg[:, (2*img.shape[2]):]  = img3[slice, :, :]
        funnyImg[funnyImg<0.0] = 0.0
        funnyImg[funnyImg>1.0] = 1.0
        plt.imshow(funnyImg, cmap='gray')
        plt.show()
    '''



def multiGradFinder(img_):
    img = np.copy(img_)
    def gradSizes(ar):
        dx, dy = np.gradient(ar)
        #dx = gaussian_filter(dx, 3)
        #dy = gaussian_filter(dy, 3)
        g = ((dx**2.0) + (dy**2.0)) ** 0.5
        return g

    for a in range(0, img.shape[0]):
        img[a] = gradSizes(img[a])
    return img

#imgGrad = multiGradFinder(img)
#imgGrado = np.copy(imgGrad)
#imgo = np.copy(img)
'''
diff = (output_grad-imgGrad)
diff[diff<0.0] = 0.0
plt.imshow(diff[:, 100, :], cmap='gray')
plt.show()
quit()
funnyImg = np.zeros((img.shape[0]*2, img.shape[2]))
funnyImg[:img.shape[0]]  = imgGrad[:, 100, :]
funnyImg[img.shape[0]:]  = output_grad[:, 100, :]/(imgGrad[:, 100, :] + (np.mean(imgGrad[:, 100, :])/10.0))
plt.imshow(funnyImg, cmap='gray')
plt.show()
quit()
'''
'''
min1 = np.mean(imgGrad)/5.0
imgGrad[imgGrad<min1] = min1
#imgGrad = np.max(imgGrad, np.mean(imgGrad)/20.0)
c_initial = 2.0
sharp = (img * (1+c_initial)) - (gaussian_filter(img, 2) * c_initial)
sharpGrad =  multiGradFinder(sharp)

c = (output_grad - imgGrad) / (sharpGrad - imgGrad)
c[c>3.0] = 3.0
c[c<0.0] = 0.0
c = gaussian_filter(c, 1.0)

#Try Mellow
#c = c / 6.0
c = c / 2.0


img = (c * sharp) + ((1.0-c) * img)
imgGrad = multiGradFinder(img)
'''

for a in range(0, 4):
    img = np.load("./stefan_data/output3D_Nov17_" + "train" + "_" + str(a) + ".npy")
    img[img>1.0] = 1.0
    img[img<0.0] = 0.0
    plt.imshow(img[:, 100, :], cmap='gray')
    plt.show()
quit()

type1 = 'test'
for a in range(1, 4):
    img_lr = np.load("./stefan_data/input3D_Nov17_test" + str(a) + ".npy")
    #img = np.load("./stefan_data/output3D_train4_Nov9_validNoCheat2_" + str(a) + ".npy")

    img = np.load("./stefan_data/output3D_Nov17_" + type1 + "_" + str(a) + ".npy")
    #img = np.load("./stefan_data/outputImage2D_valid_nov11_" + str(a) + ".npy")

    imgo = np.copy(img)
    #diff = img-img_lr
    #diff[diff>0.1] = 0.0
    #diff[diff<-0.1] = 0.0
    #img = img_lr + diff
    #doubleShow([img], 0)
    #quit()
    #'''
    #output_grad = np.load("./stefan_data/output3D_train4_Nov9_validNoCheat2_grad_" + str(a) + ".npy")

    output_grad = np.load("./stefan_data/output3D_Nov17_" + type1 + "_grad_" + str(a) + ".npy")
    #output_grad = np.load("./stefan_data/outputImage2D_valid_nov11_grad_" + str(a) + ".npy")

    imgGrad =  multiGradFinder(img)

    min1 = np.mean(imgGrad)/5.0
    imgGrad[imgGrad<min1] = min1

    c_initial = 2.0
    sharp = (img * (1+c_initial)) - (gaussian_filter(img, 2) * c_initial)
    sharpGrad =  multiGradFinder(sharp)
    c = (output_grad - imgGrad) / (sharpGrad - imgGrad)
    c[c>3.0] = 3.0
    c[c<0.0] = 0.0
    c = gaussian_filter(c, 1.0)

    #Try Mellow
    #c = c / 6.0
    c = c / 2.0
    img = (c * sharp) + ((1.0-c) * img)
    imgGrad = multiGradFinder(img)


    #img2 = np.load("./stefan_data/output3D_train4_Nov9_validNoCheat2_" + str(a) + ".npy")
    #doubleShow([c], 0)
    img_hd = np.load("./stefan_data/hd3D_Nov17_" + type1 + "_" + str(a) + ".npy")
    #img_2D = np.load("./stefan_data/outputImage2D_valid_nov11_" + str(a) + ".npy")
    #imgo5 = np.load("./stefan_data/output3D_Nov15_valid_" + str(a) + ".npy")
    #plt.imshow((img-img_lr)[:, 100, :], cmap='gray')
    #plt.show()
    #quit()

    cutOff = 0.4
    img[np.abs(img-img_lr)>cutOff] = img_lr[np.abs(img-img_lr)>cutOff]
    for b in range(0, 3):
        doubleShow([img_lr, img, img_hd], b)
    #'''
quit()

#img = np.load("./stefan_data/hr3D_train4_Nov9_validNoCheat1.npy")
img[img>1.0] = 1.0
img[img<0.0] = 0.0
plt.imshow(img[20, :, :], cmap='gray')
plt.show()
quit()

input_img = np.load("./stefan_data/inputImage3D_train3_noCheat.npy")
#img[img>1.0] = 1.0
#img[img<0.0] = 0.0
#plt.imshow(img[:, 100, :][100:125, 85:110], cmap='gray')
#plt.show()
#quit()
import raisr
R = raisr.RAISR()
angle1, angle2, angle3, strength1, coherence, strength2= R.calculate_structure_tensor(input_img)

plotField(strength1[:, 100, :][100:125, 85:110], angle2[:, 100, :][100:125, 85:110])
quit()




img = np.load("./stefan_data/outputImage3D_train3_Nov5_replicate2_2.npy")
img[img>1.0] = 1.0
img[img<0.0] = 0.0
plt.imshow(img[:, 100, :], cmap='gray')
plt.show()
quit()

#img = np.load("./stefan_data/outputImage3D_train3_DumbIndex_Recopy.npy")
#img = np.load("./stefan_data/outputImage3D_train5.npy")
img = np.load("./stefan_data/outputImage3D_train3_Index5.npy")
img = np.load("./stefan_data/outputImage3D_train3_noCheat2.npy")
img2 = np.load("./stefan_data/outputImage3D_train3_Nov5_replicate2_2.npy")
#img2 = np.load("./stefan_data/outputImage3D_train3_Nov5_replicate2_2.npy")
#doubleShow(img, img2, "banana", 1)
#quit()
#img = np.load("./stefan_data/outputImage3D_train3_Nov7_cheat1.npy")
'''
img[img>1.0] = 1.0
img[img<0.0] = 0.0
plt.imshow(img[:, :, 200], cmap='gray')
plt.show()
quit()

img2 = np.load("./stefan_data/outputImage3D_train3_Nov5_replicate2_2.npy")
'''

doubleShow(img, img2, "banana", 2)
quit()

#img = np.load("./stefan_data/outputImage3D_train3_Test_Old.npy")
img[img>1.0] = 1.0
img[img<0.0] = 0.0
plt.imshow(img[:, 100, :], cmap='gray')
plt.show()
quit()
#img = np.load("./stefan_data/outputImage3D_train3_Nov3_NoStrengthRoot_NoSinAngle.npy")
#plt.imshow(img[:, 100, :], cmap='gray')
#plt.show()
#quit()

'''
R = pickle.load(open('filter.raisr_3D_width7_DumbIndex', "rb"))
print (R.filter_width)
print (R.sigma)
print (R.order)
print (R.num_orientation)
print (R.num_orientation2)
print (R.num_coherence)
print (R.num_strength)
print (R.min_coherence)
print (R.min_strength)
print (R.max_coherence)
print (R.max_strength)
print (R.num_filters)
print (R.filters.shape)
quit()
'''

#img = np.load("./stefan_data/outputImage3D_train3_NoCheat_run2.npy")[:, 100, :]
#img = np.load("./stefan_data/outputImage3D_train3_NoCheat_5ori_run2.npy")
#img = np.load("./stefan_data/outputImage3D_train3_DumbIndex_Recopy.npy")#[:, 100, :]
img2 = np.load("./stefan_data/outputImage3D_train3_Cheat_Nov2_3.npy")

img = np.load("./stefan_data/outputImage3D_train3_Nov3_NoStrengthRoot_NoSinAngle.npy")

#img = np.load("./stefan_data/outputImage3D_train3_Cheat_Nov2.npy")
#img2 = np.load("./stefan_data/outputImage3D_train3_Cheat_Nov2_2.npy")
img[img<0.0] = 0.0
img[img>1.0] = 1.0
img2[img2<0.0] = 0.0
img2[img2>1.0] = 1.0
doubleShow(img, img2, "banana", 1)
#plt.imshow(img[:, 100, :], cmap='gray')
#plt.show()
quit()


#212
import pickle
#Rname = 'filter.raisr_noCheat'
Rname = 'filter.raisr_5orient_NoCheat'
#Rname = 'filter.raisr_sigma05'
R = pickle.load(open(Rname, "rb"))


img1 = np.load("./stefan_data/inputImage3D_train3_noCheat.npy")
img2 = np.load("./stefan_data/inputImage2D__train4_img2.npy")
imgs = [img1, img2]
for a in range(0, 2):
    indices = R.calculate_indices(imgs[a])
    indices = np.ndarray.flatten(indices)
    unique, counts = np.unique(indices, return_counts=True)
    plt.plot(unique, counts)
plt.show()
quit()
#indices2 = R.calculate_indices(img2)
#indices2 = np.ndarray.flatten(indices2)


R1 = pickle.load(open('filter.raisr_sigma05', "rb"))

'''
img = np.load("./stefan_data/inputImage3D_train3_noCheat.npy")
indices = R.calculate_indices(img)
indices = np.ndarray.flatten(indices)
unique, counts = np.unique(indices, return_counts=True)

#plt.plot(unique, counts)
#plt.show()
counts[unique<100] = 0.0
counts[173] = 0.0
counts[115] = 0.0
counts[94] = 0.0
counts[99] = 0.0
print (np.argmax(counts))
print (unique[np.argmax(counts)])
quit()
#'''

#FilterNum = 137
FilterNum = 212
#FilterNum = 217
#FilterNum = 142
#FilterNum = 112
#FilterNum = 117
filters = R.filters
filters = filters[:filters.shape[0]//9]
#print (filters.shape)
array = []
for a in range(0, filters.shape[0]):
    array.append(np.mean(filters[a]))
#print (array[-13])
#quit()
filter = filters[FilterNum]
size1 = filter.shape[0]
Width = ((size1 - 1) / 2) ** (1.0/3.0)
Width = int(np.round(Width))
filt1 = filter[1:(Width**3)+1].reshape(Width, Width, Width)
filt2 = filter[(Width**3)+1:(2*(Width**3))+1].reshape(Width, Width, Width)
mid = Width//2
cent1 = filt1[:, :, mid]
cent2 = filt2[:, :, mid]

plt.imshow(filt1[:, :, 3], cmap='gray')
plt.show()
quit()
print (np.sum(np.abs(filt1)))
filt1[3] = 0.0
filt1[:, 3] = 0.0
filt1[:, :, 3] = 0.0
print (np.sum(np.abs(filt1)))
filt1[2:5, 2:5, 2:5] = 0.0
print (np.sum(np.abs(filt1)))
filt1[1:6, 1:6, 1:6] = 0.0
print (np.sum(np.abs(filt1)))

quit()


cent1[mid+1:] = 0.0
#print (cent1)
#cent1[mid, mid] = 0
#filt1[3, 3] = 0
ar = np.zeros((Width+1, Width*3))
ar[:Width, :Width] = filt1[mid]
ar[:Width, Width:2*Width] = filt1[:, mid]
ar[:Width, 2*Width:] = filt1[:, :, mid]
plt.imshow(ar, cmap='gray')
plt.show()
#print (R.filters[100][1:(5**3)+1].reshape((5, 5, 5))[0].shape)
quit()

#img = np.load("./stefan_data/inputImage3D_train3_noCheat.npy")
img = np.load("./stefan_data/inputImage2D__train4_img2.npy")
#plt.imshow(img[:, 100, :], cmap='gray')
#plt.show()
#quit()
import raisr
R = raisr.RAISR()
R.filter(img)
quit()

#img = np.load("./stefan_data/outputImage3D_train2_2.npy")
#img = np.load("./stefan_data/outputImage3D_train2_5.npy")
#img = np.load("./stefan_data/outputImage3D_train2_6.npy")
#img1 = np.load("./stefan_data/inputImage3D_train3_3.npy")
#img = np.load("./stefan_data/outputImage3D_train3_9.npy")
#img = np.abs(img1 - img2)
#img = np.load("./stefan_data/outputImage3D_train3_DumbIndex2.npy")
#img = np.load("./stefan_data/outputImage3D_train3_Index4.npy")
#img = np.load("./stefan_data/outputImage3D_valid1_Index1.npy")

#img = np.load("./stefan_data/outputImage3D_train3_Index8.npy")[:, 100, :]

#img = np.load("./stefan_data/outputImage3D_train3_sigma05.npy")[:, 100, :]
img = np.load("./stefan_data/outputImage3D_train3_noCheat.npy")#[:, :, 200]
#img2 = np.load("./stefan_data/outputImage3D_train3_noCheat2.npy")#[:, :, 200]
img2 = np.load("./stefan_data/outputImage3D_train3_5orient_NoCheat.npy")
dataTrue = np.load('./stefan_data/train_hr_1.npy')
#img[img>1.0] = 1.0
#img[img<0.0] = 0.0

#img = find2ndGrad2D(img)

doubleShow(img, img2, dataTrue, 1, slice=12)#, slice=200)
quit()


img = np.load("./stefan_data/outputImage3D_train3_Index5.npy")

img[img>1.0] = 1.0
img[img<0.0] = 0.0

img = find2ndGrad(img)
plt.imshow(img[:, 100, :], cmap='gray')
plt.show()
quit()

ksp = sp.fft(img)#, axes=[0, 1, 2])
#img = np.sum(np.abs(sp.ifft(ksp, axes=[-1, -2, -3]))**2, axis=0)**0.5


#img = np.fft.ifft(img)
#img = np.abs(img)
#print (np.sum(img))
#print (img.shape)

#img[img>1.0] = 1.0
#img[img<0.0] = 0.0
plt.imshow(np.log(np.abs(ksp)[:, :, 160]), cmap='gray')
#plt.imshow(np.log(np.abs(ksp)[:, 128, :]), cmap='gray')
plt.show()
quit()

img_lr_bf = np.load("./stefan_data/img.npy")
plt.imshow(img_lr_bf[:, 100, :], cmap = 'gray')
plt.show()
#print (img_lr_bf.shape)
quit()

ksp = np.load('./stefan_data/ksp.npy')
ksp_lr = np.load('./stefan_data/ksp_lr.npy')
ksp_lr_bf = np.load('./stefan_data/ksp_lr_bf.npy')
print (ksp_lr_bf.shape)
#quit()
print (ksp.shape)
print (ksp_lr.shape)
print (ksp_lr_bf.shape)
#quit()
import sigpy as sp
img = np.sum(np.abs(sp.ifft(ksp_lr, axes=[-1, -2, -3]))**2, axis=0)**0.5
#img_lr_bf = np.sum(np.abs(sp.ifft(ksp_lr_bf, axes=[-1, -2, -3]))**2, axis=0)**0.5

plt.imshow(img[:, 100, :], cmap = 'gray')
plt.show()

#plt.imshow(np.real(ksp_lr[0, :, 100, :]))
#plt.show()
#plt.imshow(np.real(lr[0, :, 100, :]))
#plt.imshow(np.real(img_lr_bf[:, 100, :]), cmap='gray')
#plt.show()
quit()
'''
#R = np.load('filter.raisr')
#R = np.load(open(r'./filter.raisr', 'rb'), allow_pickle=True)
#quit()
dataTrue = np.load('./stefan_data/train_hr_1.npy')[:, 100, :]
data = np.load('./stefan_data/inputImage2D_train_str3.npy')[:, 100, :]
#orientation, strength, coherence = calculate_structure_tensor_2D(data)
incices = R.calculate_indices(data)
filters = R.filters
print (filters.shape)
quit()
'''


data = np.load("./stefan_data/outputImage3D_train2_1.npy")[:, 100, :]
data[data<0.0] = 0.0
data[data>1.0] = 1.0
plt.imshow(data, cmap="gray")
plt.show()
quit()
'''
dataTrue = np.load('./stefan_data/train_hr_1.npy')[:, 100, :]
#data = np.load("./stefan_data/outputImage3D_train5.npy")

data = np.load('./stefan_data/inputImage2D_train_str3.npy')[:, 100, :]
#print (np.mean((dataTrue-data)**2.0) ** 0.5)
data2 = np.load("./stefan_data/outputImage2D_train1.npy")[:, 100, :]
#print (np.mean((dataTrue-data)**2.0) ** 0.5)
#print (np.mean((data1-data2)**2.0) ** 0.5)


plt.imshow(data2[100:130, 80:120], cmap='gray')
plt.show()
quit()
plt.imshow(data, cmap='gray')
plt.show()
'''

#plt.quiver([0, 1], [0, 1], [1, 1], [1, 1])
#plt.show()
#quit()
R = np.load('filter.raisr', allow_pickle=True)


dataTrue = np.load('./stefan_data/train_hr_1.npy')[:, 100, :]
data = np.load('./stefan_data/inputImage2D_train_str3.npy')[:, 100, :]
#orientation, strength, coherence = calculate_structure_tensor_2D(data)
incices = R.calculate_indices(data)
filters = R.filters
quit()
plt.imshow(data, cmap='gray')
plt.scatter([100], [111])
#plt.imshow(data[100:130, 80:120], cmap='gray')
#plt.scatter([20], [11])
plt.show()
#plotField(strength[100:130, 80:120], orientation[100:130, 80:120])
quit()


data = np.load('./stefan_data/outputImage2D_train_str3.npy')[:, 100, :]

strength[strength<0.02] = 0.0
strength[strength>0.01] = 1.0
#print (np.mean((dataTrue-data)**2.0) ** 0.5)
#quit()
#img = np.sum((dataTrue-data)**2.0, axis=0)
#img = ((dataTrue-data)**2.0)[:, 50, :]
#img = dataTrue[:, 50, :]
#img = dataTrue[:, 100, :]
#print (img.shape)


plt.imshow((strength*orientation)[100:130, 80:120], cmap='gray')
plt.show()
#plt.imshow(dataTrue[90:140, 100, 75:125], cmap='gray')
#plt.imshow((dataTrue-data)[:, 100, :], cmap='gray')
#plt.show()
quit()



dataTrue = np.load('./stefan_data/train_hr_1.npy')
data = np.load("./stefan_data/outputImage3D_train.npy")
print (np.mean((dataTrue-data)**2.0) ** 0.5)
data = np.load("./stefan_data/outputImage3D_train5.npy")
print (np.mean((dataTrue-data)**2.0) ** 0.5)
quit()

#data = np.load("./stefan_data/inputImage3D_train5.npy")
data = np.load('./stefan_data/train_hr_1.npy')
#plt.imshow(data[100][120:150,222:225], cmap='gray')
plt.imshow(data[:, 100, :], cmap='gray')
#plt.imshow(data[:, :,223], cmap='gray')
plt.show()
quit()
data = np.load("./stefan_data/outputImage3D_train5.npy")
data[data<0.0] = 0.0
data[data>1.0] = 1.0
plt.imshow(data[:, 100, :], cmap='gray')
plt.show()

quit()



data = np.load('./stefan_data/train_hr_1.npy')
#plt.imshow(data[100], cmap='gray')
#plt.show()
'''
angle1, angle2, angle3, strength1, coherence, strength2 = calculate_structure_tensor(data)
np.save('./stefan_data/train_hr_1_strength1_4.npy', strength1)
np.save('./stefan_data/train_hr_1_angle1_4.npy', angle1)
np.save('./stefan_data/train_hr_1_angle2_4.npy', angle2)
np.save('./stefan_data/train_hr_1_angle3_4.npy', angle3) #Looks like noise
np.save('./stefan_data/train_hr_1_coherence_4.npy', coherence)
np.save('./stefan_data/train_hr_1_strength2_4.npy', strength2)
quit()
#'''


name2 = 'strength1'
data2 = np.load('./stefan_data/train_hr_1_' + name2 + '_3.npy')
data2[data2<0.02] = 0.0
data2[data2>0.01] = 1.0

name1 = 'angle1'
data1 = np.load('./stefan_data/train_hr_1_' + name1 + '_3.npy')
data1 = data1 * data2
plt.imshow(data1[:, 100, :], cmap='gray')
plt.show()
quit()

name1 = 'angle1'
data1 = np.load('./stefan_data/train_hr_1_' + name1 + '_1.npy')
plt.imshow(data1[:, 100, :], cmap='gray')
plt.show()

quit()


'''
dataTrue = np.load('./stefan_data/train_hr_1.npy')
data = np.load("./stefan_data/outputImage2D_train1.npy")
print (np.mean((data-dataTrue)**2.0)**0.5)
data = np.load("./stefan_data/outputImage3D_train3.npy")
print (np.mean((data-dataTrue)**2.0)**0.5)
quit()
'''

data = np.load("./stefan_data/outputImage3D_train3.npy")
#plt.imshow(data[100][120:150,222:225], cmap='gray')
plt.imshow(data[100], cmap='gray')
#plt.imshow(data[:, :,223], cmap='gray')
plt.show()

data = np.load('./stefan_data/train_hr_1.npy')
plt.imshow(data[100], cmap='gray')
plt.show()

quit()


data = np.load("./stefan_data/inputImage3D_train2.npy")
#plt.imshow(data[100][120:150,222:225], cmap='gray')
plt.imshow(data[200], cmap='gray')
#plt.imshow(data[:, :,223], cmap='gray')
plt.show()
#'''
data = np.load("./stefan_data/outputImage3D_train3.npy")
#plt.imshow(data[100][120:150,222:225], cmap='gray')
plt.imshow(data[200], cmap='gray')
#plt.imshow(data[:, :,223], cmap='gray')
plt.show()

data = np.load("./stefan_data/outputImage2D_train1.npy")
#plt.imshow(data[100][120:150,222:225], cmap='gray')
plt.imshow(data[200], cmap='gray')
#plt.imshow(data[:, :,223], cmap='gray')
plt.show()
quit()

data = np.load('./stefan_data/train_hr_1.npy')
plt.imshow(data[:, :,223], cmap='gray')
plt.show()
quit()

image = []
for a in range(0, 320):
    numPart = str(a)
    if len(numPart) == 1:
        numPart = "00" + numPart
    if len(numPart) == 2:
        numPart = "0" + numPart
    name = 'data/knees/train/img/' + '2588bfa8-0c97-478c-aa5a-487cc88a590d'+ "_" + numPart + ".npy"
    image.append(np.load(name))
image = np.array(image)
np.save('./stefan_data/train_hr_1.npy', image)

quit()
'''
name = 'data/knees/train/img/' + '2588bfa8-0c97-478c-aa5a-487cc88a590d'+ "_" + "200" + ".npy"
dataTrue = np.load(name)
data = np.load("./stefan_data/inputImage3D_train2.npy")[200]
print (np.mean((data-dataTrue)**2.0)**0.5)
data = np.load("./stefan_data/outputImage3D_train2.npy")[200]
print (np.mean((data-dataTrue)**2.0)**0.5)
quit()

'''
data = np.load("./stefan_data/inputImage3D_train2.npy")
print (np.min(data))
print (np.max(data))
plt.imshow(data[101], cmap='gray')
plt.show()



name = 'data/knees/train/img/' + '2588bfa8-0c97-478c-aa5a-487cc88a590d'+ "_" + "101" + ".npy"
data = np.load(name)
print (np.min(data))
print (np.max(data))
plt.imshow(data, cmap='gray')
plt.show()

data = np.load("./stefan_data/outputImage3D_train2.npy")
data[data<0.0] = 0.0
data[data>1.0] = 1.0
print (np.min(data))
print (np.max(data))
plt.imshow(data[101], cmap='gray')
plt.show()
#'''
'''
data = np.load("./stefan_data/inputImage3D1.npy")
print (np.min(data))
print (np.max(data))
plt.imshow(data[100], cmap='gray')
plt.show()
data = np.load("./stefan_data/outputImage3D1.npy")
data[data<0.0] = 0.0
data[data>1.0] = 1.0
print (np.min(data))
print (np.max(data))
plt.imshow(data[100], cmap='gray')
plt.show()

'''


quit()

data2 = data - data2
sizeMax = 0.02
data2[data2<-sizeMax] = -sizeMax
data2[data2>sizeMax] = sizeMax
print (np.mean(data2))
print (np.min(data2))
print (np.max(data2))
data = data + (data2 * 10.0)
plt.imshow(data[100], cmap='gray')
plt.show()
#data2 =




























#stefan_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def calculate_structure_tensor(input_image):
    dx, dy, dz = np.gradient(input_image)
    #dx, _, _ = np.gradient(dx)
    #_, dy, _ = np.gradient(dy)
    #_, _, dz = np.gradient(dz)

    dv = [dx, dy, dz]
    matrix = [[[], [], []], [[], [], []], [[], [], []]]
    for a in range(0, 3):
        for b in range(0, a+1):
            matrix[a][b] = gaussian_filter(dv[a] * dv[b], 2.0)
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
    vec2 = eigVec[:, :, :, :, 0]
    angle1 = np.arctan2(vec1[:, :, :, 0], vec1[:, :, :, 1]) + np.pi
    xyTotalVec1 = ((vec1[:, :, :, 0] ** 2.0) + (vec1[:, :, :, 1] ** 2.0)) ** 0.5
    angle2 = np.arctan2(xyTotalVec1, vec1[:, :, :, 2])

    xProj = np.zeros(vec1.shape)
    yProj = np.zeros(vec1.shape)
    xProj[:, :, :, 0] = 1.0
    yProj[:, :, :, 0] = 1.0
    for a in range(0, 3): #This loop is a bit dumb but it's the easiest way to program it I thought of.
        xProj[:, :, :, a] = xProj[:, :, :, a] - (vec1[:, :, :, 0] * vec1[:, :, :, a])
        yProj[:, :, :, a] = yProj[:, :, :, a] - (vec1[:, :, :, 1] * vec1[:, :, :, a])
    vec2x = np.sum(xProj * vec2, axis=3)
    vec2y = np.sum(yProj * vec2, axis=3)
    angle3 = np.arctan2(vec2x, vec2y)

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

def calculate_structure_tensor_2D(input_image):
    dx, dy = np.gradient(input_image)
    sigma = 1.0
    dxx = gaussian_filter(dx**2, sigma)
    dxy = gaussian_filter(dx * dy, sigma)
    dyy = gaussian_filter(dy**2, sigma)
    trace = dxx + dyy
    det = dxx * dyy - dxy**2

    lamda_1 = trace / 2 + (trace**2 / 4 - det)**0.5
    lamda_2 = trace / 2 - (trace**2 / 4 - det)**0.5

    w1 = dxy
    w2 = lamda_1 - dxx
    orientation = np.arctan2(w2, w1)
    strength = lamda_1**0.5
    coherence = (lamda_1**0.5 - lamda_2**0.5) / (lamda_1**0.5 + lamda_2**0.5)

    return orientation, strength, coherence

def stefan2D(input_image):
    dx, dy = np.gradient(input_image)
    sigma = 1.0
    dxx = gaussian_filter(dx**2, sigma)
    dxy = gaussian_filter(dx * dy, sigma)
    dyy = gaussian_filter(dy**2, sigma)
    #trace = dxx + dyy
    #det = dxx * dyy - dxy**2

    #lamda_1 = trace / 2 + (trace**2 / 4 - det)**0.5
    #lamda_2 = trace / 2 - (trace**2 / 4 - det)**0.5

    w1 = dxy
    w2 = lamda_1 - dxx
    orientation = np.arctan2(w2, w1)
    strength = lamda_1**0.5
    coherence = (lamda_1**0.5 - lamda_2**0.5) / (lamda_1**0.5 + lamda_2**0.5)

    return orientation, strength, coherence

def plotField(strength, orientation):
    X = []
    Y = []
    U = []
    V = []
    for a in range(0, strength.shape[0]):
        #a = strength.shape[0] - 1 - a1
        for b in range(0, strength.shape[1]):
            #b = strength.shape[1] - 1 - b1
            X.append(a)
            Y.append(b)
            u = -np.cos(orientation[a][b]) * strength[a][b]
            v = np.sin(orientation[a][b]) * strength[a][b]
            U.append(u)
            V.append(v)
    plt.quiver(Y, X, V, U)
    #plt.colorbar()
    plt.show()


#data1 = np.load('./data/knees/train/img_lr/1b197efe-9865-43be-ac24-f237c380513e_000.npy')
#data2 = np.load('./data/knees/train/img_lr2/1b197efe-9865-43be-ac24-f237c380513e_000.npy')
#img1 = np.load("./stefan_data/outputImage2D_train3_1.npy")
#img2 = np.load('./stefan_data/inputImage2D_train6.npy')
#print (data1[0, 0])
#print (data2[0, 0])
#quit()
#img = np.load("./stefan_data/2D_Temp/Output" + "002" + ".npy")
#img = np.load("./stefan_data/outputImage2D_train3_6.npy")
#img = np.load("./stefan_data/outputImage2D_train3_7.npy")
img = np.load("./stefan_data/outputImage2D_train3_41.npy")
print (np.min(img))
print (np.max(img))
#img = np.load('stefan_data/inputImage3D_train.npy')
#img = np.load('stefan_data/train_hr_1.npy')
img[img>1.0] = 1.0
img[img<0.0] = 0.0
plt.imshow(img[:, 100, :], cmap='gray')
#img = np.ndarray.flatten(img)
#plt.hist(img)
plt.show()
quit()

'''
import pickle
R = pickle.load(open('filter.raisr_9_lr2', "rb"))
filters = R.filters
print (filters.shape)
print (np.sum(np.abs(filters[:, 1:50])))
print (np.sum(np.abs(filters[:, 50:])))
quit()
'''

img = np.load("./stefan_data/inputImage2D_train3_2.npy")
img[img>1.0] = 1.0
img[img<0.0] = 0.0
plt.imshow(img[50], cmap='gray')
plt.show()
#plt.imshow(img[:, 100, :], cmap='gray')
#plt.show()
quit()












































#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filter', type=str)
parser.add_argument('input_dir', type=str)
parser.add_argument('output_dir', type=str)

args = parser.parse_args()
import glob
import os
import numpy as np
import multiprocessing
import raisr
import logging
import time
import pickle

def naiveIncreaseDefinition(image):
    image2 = np.zeros((image.shape[0]*2, image.shape[1]*2))
    image2[0::2, 0::2] = image
    image2[1::2, 0::2] = image
    image2[0::2, 1::2] = image
    image2[1::2, 1::2] = image
    return image2

scale1 = 10.0
scale2 = 10.0
def transf1(x):
    val = np.tanh((x-0.5)*scale1) / np.tanh(scale1 * 0.5)
    val = (val + 1.0) / 2.0
    return val

def transf2(x):
    val = ((x * 2.0) - 1.0) * np.tanh(scale2 * 0.5)
    val = (np.arctanh(val) / scale2) + 0.5
    return val

def raisr_filter(input_file):
    logging.info(input_file)

    basename = os.path.basename(input_file)
    output_file = os.path.join(args.output_dir, basename)
    input_image = np.load(input_file)
    start = time.time()
    output_image = R.filter(input_image)
    end = time.time()
    np.save(output_file, output_image)

    return end - start

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    #R = R.load(args.filter)
    #with open(args.filter, "rb") as f:
    #    #pickle.dump(self, f)
    #    R = pickle.load(f)
    R = pickle.load(open(args.filter, "rb"))

    #R = np.load("./" + args.filter + ".npz", allow_pickle=True)
    #print (R[0])
    '''
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    with multiprocessing.Pool() as p:
        times = p.map(raisr_filter, input_files)

    logging.info('Average inference time: {} s'.format(np.mean(times)))
    '''
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    inputNameStarts = []
    for file in input_files:
        inputNameStarts.append(file[24:-8])
    inputNameStarts = np.unique(np.array(inputNameStarts))
    input_images = []
    output_images = []
    for a in range(0, 1):
        input_image = []
        output_image = []
        for b in range(0, 320):
            numPart = str(b)
            if len(numPart) == 1:
                numPart = "00" + numPart
            if len(numPart) == 2:
                numPart = "0" + numPart
            print ("A")
            print (b)
            #name = 'data/knees/valid/img_lr/' + inputNameStarts[a] + "_" + numPart + ".npy"
            name = 'data/knees/train/img_lr/' + '2588bfa8-0c97-478c-aa5a-487cc88a590d'+ "_" + numPart + ".npy"
            #name = 'data/knees/train/img_lr2/' + '2588bfa8-0c97-478c-aa5a-487cc88a590d'+ "_" + numPart + ".npy"
            #name = 'data/knees/train/img_lr2_bf/' + '2588bfa8-0c97-478c-aa5a-487cc88a590d'+ "_" + numPart + ".npy"
            input_image1 = np.load(name)
            #input_image1 = naiveIncreaseDefinition(input_image1)
            #np.save("./stefan_data/2D_Temp/Input" + numPart + ".npy", input_image1)
            #print (np.min(input_image1))
            #print (np.max(input_image1))
            input_image1[input_image1>1.0] = 1.0
            #print (np.min(input_image1))
            #print (np.max(input_image1))
            input_image1 = transf2(input_image1)
            #print (np.min(input_image2))
            #print (np.max(input_image2))
            output_image1 = R.filter(input_image1)
            #print (np.min(output_image1))
            #print (np.max(output_image1))

            output_image1 = transf1(output_image1)

            #print (np.min(output_image1))
            #print (np.max(output_image1))
            #np.save("./stefan_data/2D_Temp/Output" + numPart + ".npy", output_image1)
            input_image.append(input_image1)
            output_image.append(output_image1)
        input_image = np.array(input_image)
        output_image = np.array(output_image)
        #input_images.append(input_image)

    #np.save("./stefan_data/inputImage2D_train3_38.npy", input_image)
    np.save("./stefan_data/outputImage2D_train3_41.npy", output_image)

    #print (inputNameStarts)
    #with multiprocessing.Pool() as p:
    #    times = p.map(raisr_filter, input_files)


def diffMaker(ar):
    ar2 = np.zeros(ar.shape)
    ar2[:, 1:, :] = ar[:, 1:, :] - ar[:, :-1, :]
    ar2[:, 0, :] = ar[:, 0, :] - ar[:, -1, :]

    ar3 = np.zeros(ar.shape)
    ar3[:, :, 1:] = ar[:, :, 1:] - ar[:, :, :-1]
    ar3[:, :, 0] = ar[:, :, 0] - ar[:, :, -1]

    ar4 = np.zeros(ar.shape)
    ar4[:, :-1, :] = ar[:, 1:, :] - ar[:, :-1, :]
    ar4[:, -1, :] = ar[:, 0, :] - ar[:, -1, :]

    ar5 = np.zeros(ar.shape)
    ar5[:, :, :-1] = ar[:, :, 1:] - ar[:, :, :-1]
    ar5[:, :, -1] = ar[:, :, 0] - ar[:, :, -1]
    #ar6 = np.abs(ar2) + np.abs(ar3) + np.abs(ar4) + np.abs(ar5)
    return ar6
