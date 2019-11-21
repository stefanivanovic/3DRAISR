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
    import matplotlib.pyplot as plt
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #print (args.filter)
    with open(args.filter, "rb") as f:
        print (f)
        R = pickle.load(f)
    #R = np.load(args.filter, allow_pickle=True)
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    #print (input_files[:10])
    #quit()
    #print (input_files[0::320])
    type1 = input_files[0]
    type1 = type1[11:15]
    inputNameStarts = []
    for file in input_files:
        if type1 == "test":
            inputNameStarts.append(file[23:-8])
        else:
            inputNameStarts.append(file[24:-8])
    #print (input_files[0])
    inputNameStarts = np.unique(np.array(inputNameStarts))
    input_images = []
    for a in range(0, len(inputNameStarts)):
        print (a)
        input_image = []
        for b in range(0, 320):
            numPart = str(b)
            if len(numPart) == 1:
                numPart = "00" + numPart
            if len(numPart) == 2:
                numPart = "0" + numPart
            name = 'data/knees/train/img_lr/' + inputNameStarts[a] + "_" + numPart + ".npy"
            #name = 'data/knees/train/img_lr/' + '2588bfa8-0c97-478c-aa5a-487cc88a590d'+ "_" + numPart + ".npy"
            #name = 'data/knees/valid/img_lr/' + 'b7d435a1-2421-48d2-946c-d1b3372f7c60'+ "_" + numPart + ".npy"
            #name = 'data/knees/valid/img_lr/' + 'b65b9167-a0d0-4295-9dd5-74641b3dd7e6'+ "_" + numPart + ".npy"
            input_image.append(np.load(name))
        input_image = np.array(input_image)
        #plt.imshow(input_image[:, 100, :], cmap='gray')
        #plt.show()
        #np.save("./stefan_data/hd3D_Nov17_test_" + str(a) + ".npy", input_image)

        np.save("./stefan_data/input3D_Nov17_train_" + str(a) + ".npy", input_image)
        #quit()
        #np.save("./stefan_data/input3D_train4_Nov9_validNoCheat1_" + str(a) + ".npy", input_image)
        output_image = R.filter(input_image)
        #output_image = np.load("./stefan_data/output3D_Nov17_test_" + str(a) + ".npy")
        np.save("./stefan_data/output3D_Nov17_train_" + str(a) + ".npy", output_image)

        #input_images.append(input_image)
    quit()

    #import matplotlib.pyplot as plt
    #plt.imshow(input_image[:, 100, :], cmap='gray')
    #plt.show()
    #quit()
    #np.save("./stefan_data/hr3D_train4_Nov9_validNoCheat2.npy", input_image)
    #quit()
    #np.save("./stefan_data/inputImage2D__train4_img2.npy", input_image)
    print ("A")
    print (R.filters.shape)
    print (R.num_orientation),
    print (R.num_coherence)
    print (R.num_strength)
    print (R.num_orientation2)
    output_image = R.filter(input_images[0])
    print ("B")
    #np.save("./stefan_data/outputImage3D_train3_Test_Old.npy", output_image)
    np.save("./stefan_data/output3D_train4_Nov9_validNoCheat2.npy", output_image)
    print ("C")
    #print (inputNameStarts)
    #with multiprocessing.Pool() as p:
    #    times = p.map(raisr_filter, input_files)
    quit()
    logging.info('Average inference time: {} s'.format(np.mean(times)))
