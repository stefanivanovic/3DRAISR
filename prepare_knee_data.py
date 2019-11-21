import logging
import mridata
import ismrmrd
import itertools
import multiprocessing
import numpy as np
import sigpy as sp

from pathlib import Path

import sys #Stefan

data_dir = Path.cwd() / 'data' / 'knees'
train_dir = data_dir / 'train'
valid_dir = data_dir / 'valid'
test_dir = data_dir / 'test'

train_ismrmrd_dir = train_dir / 'ismrmrd'
valid_ismrmrd_dir = valid_dir / 'ismrmrd'
test_ismrmrd_dir = test_dir / 'ismrmrd'

train_img_dir = train_dir / 'img'
valid_img_dir = valid_dir / 'img'
test_img_dir = test_dir / 'img'

train_img_lr_dir = train_dir / 'img_lr'
valid_img_lr_dir = valid_dir / 'img_lr'
test_img_lr_dir = test_dir / 'img_lr'


def prepare_knee_data(ismrmrd_path):
    """Convert ISMRMRD file to slices along readout.

    Args:
        ismrmrd_path (pathlib.Path): file path to ISMRMRD file.

    """

    logging.info('Processing {}'.format(ismrmrd_path.stem))
    dset = ismrmrd.Dataset(str(ismrmrd_path))
    hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())

    matrix_size_x = hdr.encoding[0].encodedSpace.matrixSize.x
    matrix_size_y = hdr.encoding[0].encodedSpace.matrixSize.y
    number_of_slices = hdr.encoding[0].encodingLimits.slice.maximum + 1
    number_of_channels = hdr.acquisitionSystemInformation.receiverChannels

    ksp = np.zeros([number_of_channels, number_of_slices, matrix_size_y, matrix_size_x],
                   dtype=np.complex64)
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        y = acq.idx.kspace_encode_step_1
        ksp[:, acq.idx.slice, y, :] = acq.data

    ksp = np.fft.fft(np.fft.ifftshift(ksp, axes=-3), axis=-3)
    ksp_lr_bf = sp.resize(ksp, [number_of_channels,
                                       number_of_slices // 2,
                                       matrix_size_y // 2,
                                       matrix_size_x // 2])
    ksp_lr = sp.resize(ksp_lr_bf, ksp.shape)
    del ksp

    #ksp_lr = sp.resize(sp.resize(ksp, [number_of_channels,
    #                                   number_of_slices // 2,
    #                                   matrix_size_y // 2,
    #                                   matrix_size_x]),
    #                   ksp.shape)

    #img = np.sum(np.abs(sp.ifft(ksp, axes=[-1, -2, -3]))**2, axis=0)**0.5
    img_lr_bf = np.sum(np.abs(sp.ifft(ksp_lr_bf, axes=[-1, -2, -3]))**2, axis=0)**0.5
    #np.save("./stefan_data/img_lr_bf.npy", img_lr_bf)
    #quit()
    img_lr = np.sum(np.abs(sp.ifft(ksp_lr, axes=[-1, -2, -3]))**2, axis=0)**0.5
    smallMatrixX = matrix_size_x // 2
    scale = 1 / img_lr.max()
    scale2 = 1 / img_lr_bf.max()
    for i in range(matrix_size_x):
        logging.info('Processing {}_{:03d}'.format(ismrmrd_path.stem, i))
        #img_i_path = ismrmrd_path.parents[1] / 'img' / '{}_{:03d}'.format(ismrmrd_path.stem, i)
        #img_lr_i_path = ismrmrd_path.parents[1] / 'img_lr' / '{}_{:03d}'.format(ismrmrd_path.stem, i)
        img_lr_i_path = ismrmrd_path.parents[1] / 'img_lr2' / '{}_{:03d}'.format(ismrmrd_path.stem, i)
        if i < smallMatrixX:
            img_lr_bf_i_path = ismrmrd_path.parents[1] / 'img_lr2_bf' / '{}_{:03d}'.format(ismrmrd_path.stem, i)

        #img_i = img[..., i]
        img_lr_i = img_lr[..., i]
        if i < smallMatrixX:
            img_lr_bf_i = img_lr_bf[..., i]
        #np.save(str(img_i_path), img_i * scale)
        np.save(str(img_lr_i_path), img_lr_i * scale)
        if i < smallMatrixX:
            np.save(str(img_lr_bf_i_path), img_lr_bf_i * scale2)


def stefan_check_knee_data(ismrmrd_path):
    logging.info('Processing {}'.format(ismrmrd_path.stem))
    dset = ismrmrd.Dataset(str(ismrmrd_path))
    hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())

    matrix_size_x = hdr.encoding[0].encodedSpace.matrixSize.x
    matrix_size_y = hdr.encoding[0].encodedSpace.matrixSize.y
    number_of_slices = hdr.encoding[0].encodingLimits.slice.maximum + 1
    number_of_channels = hdr.acquisitionSystemInformation.receiverChannels
    print (number_of_slices)
    print (matrix_size_x)
    print (matrix_size_y)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_ismrmrd_dir.mkdir(parents=True, exist_ok=True)
    valid_ismrmrd_dir.mkdir(parents=True, exist_ok=True)
    test_ismrmrd_dir.mkdir(parents=True, exist_ok=True)

    train_img_dir.mkdir(parents=True, exist_ok=True)
    valid_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)

    train_img_lr_dir.mkdir(parents=True, exist_ok=True)
    valid_img_lr_dir.mkdir(parents=True, exist_ok=True)
    test_img_lr_dir.mkdir(parents=True, exist_ok=True)

    #mridata.batch_download(train_dir / 'uuids.txt', folder=train_ismrmrd_dir)
    #mridata.batch_download(valid_dir / 'uuids.txt', folder=valid_ismrmrd_dir)
    #mridata.batch_download(test_dir / 'uuids.txt', folder=test_ismrmrd_dir)

    fileNames = list(train_ismrmrd_dir.glob('*.h5')) + list(valid_ismrmrd_dir.glob('*.h5')) + list(test_ismrmrd_dir.glob('*.h5'))
    #print (fileNames)
    #prepare_knee_data(fileNames)
    #quit()
    for a in range(16, len(fileNames)):
        filename = str(fileNames[a])
        print (a)
        if (filename != "/Users/stefanivanovic/Desktop/raisr_mri-master/data/knees/valid/ismrmrd/.h5") and (filename != "/Users/stefanivanovic/Desktop/raisr_mri_stefan/data/knees/valid/ismrmrd/.h5"):
            prepare_knee_data(fileNames[a])

    #with multiprocessing.Pool() as p:
    #    p.map(prepare_knee_data, itertools.chain(train_ismrmrd_dir.glob('*.h5'),
    #                                             valid_ismrmrd_dir.glob('*.h5'),
    #                                             test_ismrmrd_dir.glob('*.h5')))
