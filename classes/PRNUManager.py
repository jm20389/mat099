# PRNU
import numpy as np
import pywt, configparser

from numpy.fft             import fft2, ifft2
from scipy.ndimage         import filters
from multiprocessing       import Pool, cpu_count
from tqdm                  import tqdm
from skimage.restoration   import denoise_tv_chambolle
from sklearn.metrics       import roc_curve, auc
from sklearn.decomposition import PCA

from classes.DatasetProcessor import DatasetProcessor
from classes.PRNUProcessor    import PRNUProcessor

# Test methods
import cv2, requests, re, os
from glob import glob
from PIL  import Image
from io   import BytesIO


class PRNUManager(PRNUProcessor):

    @staticmethod
    def runWorkload(wl):
        sequence = wl['sequence']
        if sequence.lower() == 'a':
            stats_cc, stats_pce, k = PRNUManager.testPRNU(image_manipulation = wl['manipulation'], deviceList = wl['deviceList'])

        return {'auc_cc': stats_cc['auc'], 'auc_pce': stats_pce['auc']}

    """
    Test sequences
    """
    @staticmethod
    def testPRNU(
                 ff_dir              = None
                 ,nat_dir            = None
                 ,image_manipulation = None
                 ,crop               = (512, 512, 3)
                 ,vision_dataset     = False
                 ,deviceList         = None
                 ):
        """
        Main example script. Load a subset of flatfield and natural images from Dresden.
        For each device compute the fingerprint from all the flatfield images.
        For each natural image compute the noise residual.
        Check the detection performance obtained with cross-correlation and PCE
        :return:
        """

        if ff_dir is None or nat_dir is None:
            config_file_path = os.path.join(os.path.dirname(__file__), '/config.ini')
            config = configparser.ConfigParser()
            config.read(config_file_path)

            ff_dir = config['vision_dataset']['ff_dir']
            nat_dir = config['vision_dataset']['nat_dir']

        if vision_dataset:
            print('Loading remote Vision Dataset..')
            ff_dirlist, ff_device, nat_dirlist, nat_device = DatasetProcessor.remote_vision_dataset()
        else:
            print('Loading local Vision Dataset..')
            ff_dirlist, ff_device, nat_dirlist, nat_device = DatasetProcessor.local_vision_dataset(ff_dir, nat_dir, deviceList)

        fingerprint_device = sorted(np.unique(ff_device))
        k = PRNUProcessor.compute_fingerprints(ff_dirlist, ff_device, crop)
        w = PRNUProcessor.compute_residuals(nat_dirlist, crop, image_manipulation)

        # Computing Ground Truth
        gt = PRNUProcessor.gt(fingerprint_device, nat_device)

        print('Computing cross correlation')
        cc_aligned_rot = PRNUProcessor.aligned_cc(k, w)['cc']

        print('Computing statistics cross correlation')
        stats_cc = PRNUProcessor.stats(cc_aligned_rot, gt)

        print('Computing PCE')
        pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

        for fingerprint_idx, fingerprint_k in enumerate(k):
            for natural_idx, natural_w in enumerate(w):
                cc2d = PRNUProcessor.crosscorr_2d(fingerprint_k, natural_w)
                pce_rot[fingerprint_idx, natural_idx] = PRNUProcessor.pce(cc2d)['pce']

        print('Computing statistics on PCE')
        stats_pce = PRNUProcessor.stats(pce_rot, gt)

        print('AUC on CC {:.2f}, expected {:.2f}'.format(stats_cc['auc'], 0.98))
        print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))

        return stats_cc, stats_pce, k