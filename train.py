#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filter_width', default=5, type=int)
#parser.add_argument('--filter_width', default=9, type=int)
parser.add_argument('--order', default=2, type=int) #For now !
parser.add_argument('input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('filter', type=str)

args = parser.parse_args()
import os
import numpy as np
import raisr
import glob
import logging

logging.basicConfig(level=logging.DEBUG)
#input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
#output_files = sorted(glob.glob(os.path.join(args.output_dir, '*.npy')))

logging.info('Reading images')
#input_images = [np.load(f) for f in input_files]
#output_images = [np.load(f) for f in output_files]
#input_images = [np.load('./stefan_data/img1.npy')]
#output_images = [np.load('./stefan_data/img_lr1.npy')]

input_images = []
output_images = []
fileEnds = ['2588bfa8-0c97-478c-aa5a-487cc88a590d', '530a812a-4870-4d01-9db4-772c853d693c', '280cf3f9-3b7e-4738-84e0-f72b21aa5266', '226e710b-725b-4bec-840e-bf47be2b8a44', '7a9f028c-8667-48aa-8e08-0acf3320c8d4', '38b9a8e8-2779-4979-8602-5e8e5f902863', '1b197efe-9865-43be-ac24-f237c380513e', '9a740e7b-8fc3-46f9-9f70-1b7bedec37e4', '52c2fd53-d233-4444-8bfd-7c454240d314', '54c077b2-7d68-4e77-b729-16afbccae9ac', '8ad53ab7-07f9-4864-98d0-dc43145ff588', '8eff1229-8074-41fa-8b5e-441b501f10e3']
for a in range(0, len(fileEnds)):
    fileEnd = fileEnds[a]
    input_image = []
    output_image = []
    for a in range(0, 320):
        nameStart = './data/knees/train/img'
        name_middle = '/' + fileEnd + '_'
        name_end = '.npy'
        numPart = str(a)
        if len(numPart) == 1:
            numPart = "00" + numPart
        if len(numPart) == 2:
            numPart = "0" + numPart
        name_end = name_middle + numPart + name_end
        name1 = nameStart + "_lr" + name_end
        name2 = nameStart + name_end
        input_image.append(np.load(name1))
        output_image.append(np.load(name2))
    input_images.append(np.array(input_image)) #= [np.array(input_image)]
    output_images.append(np.array(output_image)) #= [np.array(output_image)]
del input_image
del output_image



#input_images = input_images[1:]
#output_images = output_images[1:]

args2 = np.arange(len(input_images))
args2 = args2[args2 != 2]
args2 = args2[args2 != 6]
args2 = args2[args2 != 8]

input_images = np.array(input_images)[args2]
output_images = np.array(output_images)[args2]


'''
img = np.array(input_images[0])
img[img>1.0] = 1.0
img[img<0.0] = 0.0

import matplotlib.pyplot as plt
#img = find2ndGrad2D(img)
plt.imshow(img[:, 100, :], cmap='gray')
plt.show()
quit()
'''

logging.info('Training RAISR filter on {} image pairs'.format(len(input_images)))
R = raisr.RAISR(filter_width=args.filter_width, order=args.order)
R.train(input_images, output_images)

logging.info('Saving RAISR filter')

R.save(args.filter)
