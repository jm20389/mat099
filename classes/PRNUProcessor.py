# PRNU
import numpy as np
import pywt

from numpy.fft             import fft2, ifft2
from scipy.ndimage         import filters
from multiprocessing       import Pool, cpu_count
from tqdm                  import tqdm
from skimage.restoration   import denoise_tv_chambolle
from sklearn.metrics       import roc_curve, auc
from sklearn.decomposition import PCA

from classes.ImageProcessor   import ImageProcessor
from classes.PickleHandler    import PickleHandler
from classes.DatasetProcessor import DatasetProcessor

# Test methods
import cv2, requests, re, os
from glob import glob
from PIL  import Image
from io   import BytesIO


class PRNUProcessor(ImageProcessor, PickleHandler, DatasetProcessor):
    """
    Extraction functions
    """
    @staticmethod
    def extract_single(im: np.ndarray,
                    levels: int = 4,
                    sigma: float = 5,
                    wdft_sigma: float = 0) -> np.ndarray:
        """
        Extract noise residual from a single image
        :param im: grayscale or color image, np.uint8
        :param levels: number of wavelet decomposition levels
        :param sigma: estimated noise power
        :param wdft_sigma: estimated DFT noise power
        :return: noise residual
        """

        W = PRNUProcessor.noise_extract(im, levels, sigma)
        W = PRNUProcessor.rgb2gray(W)
        W = PRNUProcessor.zero_mean_total(W)
        W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
        W = PRNUProcessor.wiener_dft(W, W_std).astype(np.float32)

        return W

    @staticmethod
    def noise_extract(im: np.ndarray, sigma: float = 5, levels: int = 4, method: str = 'Wiener filter') -> np.ndarray:
        if method == 'Wiener filter':
            return PRNUProcessor.noise_extract_wiener(im = im, levels = levels, sigma = sigma)
        elif method == 'Gaussian blur':
            return PRNUProcessor.noise_extract_gaussian(im = im, sigma = sigma)
        elif method == 'Total variation':
            return PRNUProcessor.noise_extract_tv(im = im)
        elif method == 'Non-local means':
            return PRNUProcessor.noise_extract_nlm(im = im)
        elif method == 'PCA':
            return PRNUProcessor.noise_extract_pca(im = im)
        else:
            print('Method error: Unknown noise extract method')
            return np.ndarray()

    @staticmethod
    def noise_extract_gaussian(im: np.ndarray, sigma: float = 5) -> np.ndarray:
        """
        Noise extraction using Gaussian blur.

        :param im: grayscale or color image, np.uint8
        :param sigma: standard deviation of the Gaussian blur
        :return: noise residual
        """

        assert (im.dtype == np.uint8)
        assert (im.ndim in [2, 3])

        im = im.astype(np.float32)

        # Apply Gaussian blur to the image
        blurred = cv2.GaussianBlur(im, (0, 0), sigma)

        # Calculate the noise residual by subtracting the blurred image from the original
        noise_residual = im - blurred

        return noise_residual
    @staticmethod
    def noise_extract_wiener(im: np.ndarray, levels: int = 4, sigma: float = 5) -> np.ndarray:
        """
        NoiseExtract as from Binghamton toolbox.

        :param im: grayscale or color image, np.uint8
        :param levels: number of wavelet decomposition levels
        :param sigma: estimated noise power
        :return: noise residual
        """

        assert (im.dtype == np.uint8)
        assert (im.ndim in [2, 3])

        im = im.astype(np.float32)

        noise_var = sigma ** 2

        if im.ndim == 2:
            im.shape += (1,)

        W = np.zeros(im.shape, np.float32)

        for ch in range(im.shape[2]):

            wlet = None
            while wlet is None and levels > 0:
                try:
                    wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
                except ValueError:
                    levels -= 1
                    wlet = None
            if wlet is None:
                raise ValueError('Impossible to compute Wavelet filtering for input size: {}'.format(im.shape))

            wlet_details = wlet[1:]

            wlet_details_filter = [None] * len(wlet_details)
            # Cycle over Wavelet levels 1:levels-1
            for wlet_level_idx, wlet_level in enumerate(wlet_details):
                # Cycle over H,V,D components
                level_coeff_filt = [None] * 3
                for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                    level_coeff_filt[wlet_coeff_idx] = PRNUProcessor.wiener_adaptive(wlet_coeff, noise_var)
                wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

            # Set filtered detail coefficients for Levels > 0 ---
            wlet[1:] = wlet_details_filter

            # Set to 0 all Level 0 approximation coefficients ---
            wlet[0][...] = 0

            # Invert wavelet transform ---
            wrec = pywt.waverec2(wlet, 'db4')
            try:
                W[:, :, ch] = wrec
            except ValueError:
                W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
                W[:, :, ch] = wrec

        if W.shape[2] == 1:
            W.shape = W.shape[:2]

        W = W[:im.shape[0], :im.shape[1]]

        return W
    @staticmethod
    def noise_extract_tv(im: np.ndarray, weight: float = 0.1) -> np.ndarray:
        """
        Noise extraction using Total Variation (TV) denoising.

        :param im: grayscale or color image, np.uint8
        :param weight: regularization weight (adjustable)
        :return: noise residual
        """

        assert (im.dtype == np.uint8)
        assert (im.ndim in [2, 3])

        im = im.astype(np.float32)

        # Apply Total Variation (TV) denoising to the image
        denoised = denoise_tv_chambolle(im, weight=weight)

        # Calculate the noise residual by subtracting the denoised image from the original
        noise_residual = im - denoised

        return noise_residual
    @staticmethod
    def noise_extract_nlm(im: np.ndarray, h: float = 10.0, template_window: int = 7, search_window: int = 21) -> np.ndarray:
        """
        Noise extraction using Non-Local Means (NLM) denoising.

        :param im: grayscale or color image, np.uint8
        :param h: filtering strength parameter
        :param template_window: size of the template window
        :param search_window: size of the search window
        :return: noise residual
        """

        assert (im.dtype == np.uint8)
        assert (im.ndim in [2, 3])

        im = im.astype(np.float32)

        # Apply Non-Local Means (NLM) denoising to the image
        denoised = cv2.fastNlMeansDenoisingColored(im, None, h=h, templateWindowSize=template_window, searchWindowSize=search_window)

        # Calculate the noise residual by subtracting the denoised image from the original
        noise_residual = im - denoised

        return noise_residual
    @staticmethod
    def noise_extract_pca(im: np.ndarray, n_components: int = 3) -> np.ndarray:
        """
        Noise extraction using Principal Component Analysis (PCA) denoising.

        :param im: grayscale or color image, np.uint8
        :param n_components: number of principal components to retain
        :return: noise residual
        """

        assert (im.dtype == np.uint8)
        assert (im.ndim in [2, 3])

        if im.ndim == 3:
            # Flatten the image for PCA
            im_flat = im.reshape(-1, 3)
        else:
            # If the image is grayscale, reshape to (height, width, 1)
            im_flat = im.reshape(im.shape[0], im.shape[1], 1)

        # Apply PCA to the flattened image
        pca = PCA(n_components=n_components)
        im_pca = pca.fit_transform(im_flat)

        # Reconstruct the image from the selected principal components
        im_denoised = pca.inverse_transform(im_pca)

        if im.ndim == 3:
            # Reshape the denoised image back to (height, width, 3) for color images
            im_denoised = im_denoised.reshape(im.shape)
        else:
            # Reshape for grayscale images
            im_denoised = im_denoised.reshape(im.shape)

        # Calculate the noise residual by subtracting the denoised image from the original
        noise_residual = im - im_denoised

        return noise_residual

    @staticmethod
    def noise_extract_compact(args):
        """
        Extract residual, multiplied by the image. Useful to save memory in multiprocessing operations
        :param args: (im, levels, sigma), see noise_extract for usage
        :return: residual, multiplied by the image
        """
        w = PRNUProcessor.noise_extract(*args)
        im = args[0]
        return (w * im / 255.).astype(np.float32)
    @staticmethod
    def extract_multiple_aligned(imgs: list, levels: int = 4, sigma: float = 5, processes: int = None,
                                batch_size=cpu_count(), tqdm_str: str = '') -> np.ndarray:
        """
        Extract PRNU from a list of images. Images are supposed to be the same size and properly oriented
        :param tqdm_str: tqdm description (see tqdm documentation)
        :param batch_size: number of parallel processed images
        :param processes: number of parallel processes
        :param imgs: list of images of size (H,W,Ch) and type np.uint8
        :param levels: number of wavelet decomposition levels
        :param sigma: estimated noise power
        :return: PRNU
        """

        assert (isinstance(imgs[0], np.ndarray))
        assert (imgs[0].ndim == 3)
        assert (imgs[0].dtype == np.uint8)

        h, w, ch = imgs[0].shape

        RPsum = np.zeros((h, w, ch), np.float32)
        NN = np.zeros((h, w, ch), np.float32)

        if processes is None or processes > 1:
            args_list = []
            for im in imgs:
                args_list += [(im, levels, sigma)]
            pool = Pool(processes=processes)

            for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                                desc=(tqdm_str + ' (1/2)'), dynamic_ncols=True):
                nni = pool.map(PRNUProcessor.inten_sat_compact, args_list[batch_idx0:batch_idx0 + batch_size])
                for ni in nni:
                    NN += ni
                del nni

            for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                                desc=(tqdm_str + ' (2/2)'), dynamic_ncols=True):
                wi_list = pool.map(PRNUProcessor.noise_extract_compact, args_list[batch_idx0:batch_idx0 + batch_size])
                for wi in wi_list:
                    RPsum += wi
                del wi_list

            pool.close()

        else:  # Single process
            for im in tqdm(imgs, disable=tqdm_str is None, desc=tqdm_str, dynamic_ncols=True):
                RPsum += PRNUProcessor.noise_extract_compact((im, levels, sigma))
                NN += (PRNUProcessor.inten_scale(im) * PRNUProcessor.saturation(im)) ** 2

        K = RPsum / (NN + 1)
        K = PRNUProcessor.rgb2gray(K)
        K = PRNUProcessor.zero_mean_total(K)
        K = PRNUProcessor.wiener_dft(K, K.std(ddof=1)).astype(np.float32)

        return K
    @staticmethod
    def cut_ctr(array: np.ndarray, sizes: tuple) -> np.ndarray:
        """
        Cut a multi-dimensional array at its center, according to sizes
        :param array: multidimensional array
        :param sizes: tuple of the same length as array.ndim
        :return: multidimensional array, center cut
        """
        array = array.copy()
        if not (array.ndim == len(sizes)):
            raise ArgumentError('array.ndim must be equal to len(sizes)')
        for axis in range(array.ndim):
            axis_target_size = sizes[axis]
            axis_original_size = array.shape[axis]
            if axis_target_size > axis_original_size:
                raise ValueError(
                    'Can\'t have target size {} for axis {} with original size {}'.format(axis_target_size, axis,
                                                                                        axis_original_size))
            elif axis_target_size < axis_original_size:
                axis_start_idx = (axis_original_size - axis_target_size) // 2
                axis_end_idx = axis_start_idx + axis_target_size
                array = np.take(array, np.arange(axis_start_idx, axis_end_idx), axis)
        return array
    @staticmethod
    def wiener_dft(im: np.ndarray, sigma: float) -> np.ndarray:
        """
        Adaptive Wiener filter applied to the 2D FFT of the image
        :param im: multidimensional array
        :param sigma: estimated noise power
        :return: filtered version of input im
        """
        noise_var = sigma ** 2
        h, w = im.shape

        im_noise_fft = fft2(im)
        im_noise_fft_mag = np.abs(im_noise_fft / (h * w) ** .5)

        im_noise_fft_mag_noise = PRNUProcessor.wiener_adaptive(im_noise_fft_mag, noise_var)

        zeros_y, zeros_x = np.nonzero(im_noise_fft_mag == 0)

        im_noise_fft_mag[zeros_y, zeros_x] = 1
        im_noise_fft_mag_noise[zeros_y, zeros_x] = 0

        im_noise_fft_filt = im_noise_fft * im_noise_fft_mag_noise / im_noise_fft_mag
        im_noise_filt = np.real(ifft2(im_noise_fft_filt))

        return im_noise_filt.astype(np.float32)
    @staticmethod
    def zero_mean(im: np.ndarray) -> np.ndarray:
        """
        ZeroMean called with the 'both' argument, as from Binghamton toolbox.
        :param im: multidimensional array
        :return: zero mean version of input im
        """
        # Adapt the shape ---
        if im.ndim == 2:
            im.shape += (1,)

        h, w, ch = im.shape

        # Subtract the 2D mean from each color channel ---
        ch_mean = im.mean(axis=0).mean(axis=0)
        ch_mean.shape = (1, 1, ch)
        i_zm = im - ch_mean

        # Compute the 1D mean along each row and each column, then subtract ---
        row_mean = i_zm.mean(axis=1)
        col_mean = i_zm.mean(axis=0)

        row_mean.shape = (h, 1, ch)
        col_mean.shape = (1, w, ch)

        i_zm_r = i_zm - row_mean
        i_zm_rc = i_zm_r - col_mean

        # Restore the shape ---
        if im.shape[2] == 1:
            i_zm_rc.shape = im.shape[:2]

        return i_zm_rc
    @staticmethod
    def zero_mean_total(im: np.ndarray) -> np.ndarray:
        """
        ZeroMeanTotal as from Binghamton toolbox.
        :param im: multidimensional array
        :return: zero mean version of input im
        """
        im[0::2, 0::2] = PRNUProcessor.zero_mean(im[0::2, 0::2])
        im[1::2, 0::2] = PRNUProcessor.zero_mean(im[1::2, 0::2])
        im[0::2, 1::2] = PRNUProcessor.zero_mean(im[0::2, 1::2])
        im[1::2, 1::2] = PRNUProcessor.zero_mean(im[1::2, 1::2])
        return im
    @staticmethod
    def rgb2gray(im: np.ndarray) -> np.ndarray:
        """
        RGB to gray as from Binghamton toolbox.
        :param im: multidimensional array
        :return: grayscale version of input im
        """
        rgb2gray_vector = np.asarray([0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
        rgb2gray_vector.shape = (3, 1)

        if im.ndim == 2:
            im_gray = np.copy(im)
        elif im.shape[2] == 1:
            im_gray = np.copy(im[:, :, 0])
        elif im.shape[2] == 3:
            w, h = im.shape[:2]
            im = np.reshape(im, (w * h, 3))
            im_gray = np.dot(im, rgb2gray_vector)
            im_gray.shape = (w, h)
        else:
            raise ValueError('Input image must have 1 or 3 channels')

        return im_gray.astype(np.float32)
    @staticmethod
    def threshold(wlet_coeff_energy_avg: np.ndarray, noise_var: float) -> np.ndarray:
        """
        Noise variance theshold as from Binghamton toolbox.
        :param wlet_coeff_energy_avg:
        :param noise_var:
        :return: noise variance threshold
        """
        res = wlet_coeff_energy_avg - noise_var
        return (res + np.abs(res)) / 2
    @staticmethod
    def wiener_adaptive(im: np.ndarray, noise_var: float, **kwargs) -> np.ndarray:
        """
        WaveNoise as from Binghamton toolbox.
        Wiener adaptive flter aimed at extracting the noise component
        For each input pixel the average variance over a neighborhoods of different window sizes is first computed.
        The smaller average variance is taken into account when filtering according to Wiener.
        :param x: 2D matrix
        :param noise_var: Power spectral density of the noise we wish to extract (S)
        :param window_size_list: list of window sizes
        :return: wiener filtered version of input x
        """
        window_size_list = list(kwargs.pop('window_size_list', [3, 5, 7, 9]))
        energy = im ** 2

        avg_win_energy = np.zeros(im.shape + (len(window_size_list),))
        for window_idx, window_size in enumerate(window_size_list):
            avg_win_energy[:, :, window_idx] = filters.uniform_filter(energy, window_size, mode='constant')

        coef_var = PRNUProcessor.threshold(avg_win_energy, noise_var)
        coef_var_min = np.min(coef_var, axis=2)

        # print(f'noise var: {noise_var}')
        # print(f'coef var min: {coef_var_min}')
        # print(f'coef var min + noise_var: {(coef_var_min + noise_var)}')
        # quit()

        # if coef_var_min + noise_var == 0:
        #     print(f'coef var min: {coef_var_min}')
        #     print(f'noise_var: {noise_var}')
        #     raise ValueError('Wiener adaptive wavenoise error: Divide by zero.')

        im = im * noise_var / (coef_var_min + noise_var)

        return im
    @staticmethod
    def inten_scale(im: np.ndarray) -> np.ndarray:
        """
        IntenScale as from Binghamton toolbox
        :param im: type np.uint8
        :return: intensity scaled version of input x
        """

        assert (im.dtype == np.uint8)
        T = 252
        v = 6
        out = np.exp(-1 * (im - T) ** 2 / v)
        out[im < T] = im[im < T] / T
        return out
    @staticmethod
    def saturation(im: np.ndarray) -> np.ndarray:
        """
        Saturation as from Binghamton toolbox
        :param im: type np.uint8
        :return: saturation map from input im
        """
        assert (im.dtype == np.uint8)

        if im.ndim == 2:
            im.shape += (1,)

        h, w, ch = im.shape

        if im.max() < 250:
            return np.ones((h, w, ch))

        im_h = im - np.roll(im, (0, 1), (0, 1))
        im_v = im - np.roll(im, (1, 0), (0, 1))
        satur_map = \
            np.bitwise_not(
                np.bitwise_and(
                    np.bitwise_and(
                        np.bitwise_and(
                            im_h != 0, im_v != 0
                        ), np.roll(im_h, (0, -1), (0, 1)) != 0
                    ), np.roll(im_v, (-1, 0), (0, 1)) != 0
                )
            )

        max_ch = im.max(axis=0).max(axis=0)

        for ch_idx, max_c in enumerate(max_ch):
            if max_c > 250:
                satur_map[:, :, ch_idx] = \
                    np.bitwise_not(
                        np.bitwise_and(
                            im[:, :, ch_idx] == max_c, satur_map[:, :, ch_idx]
                        )
                    )

        return satur_map
    @staticmethod
    def inten_sat_compact(args):
        """
        Memory saving version of inten_scale followed by saturation. Useful for multiprocessing
        :param args:
        :return: intensity scale and saturation of input
        """
        im = args[0]
        return ((PRNUProcessor.inten_scale(im) * PRNUProcessor.saturation(im)) ** 2).astype(np.float32)

    """
    Cross-correlation functions
    """
    @staticmethod
    def crosscorr_2d(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
        """
        PRNU 2D cross-correlation
        :param k1: 2D matrix of size (h1,w1)
        :param k2: 2D matrix of size (h2,w2)
        :return: 2D matrix of size (max(h1,h2),max(w1,w2))
        """
        assert (k1.ndim == 2)
        assert (k2.ndim == 2)

        max_height = max(k1.shape[0], k2.shape[0])
        max_width = max(k1.shape[1], k2.shape[1])

        k1 -= k1.flatten().mean()
        k2 -= k2.flatten().mean()

        k1 = np.pad(k1, [(0, max_height - k1.shape[0]), (0, max_width - k1.shape[1])], mode='constant', constant_values=0)
        k2 = np.pad(k2, [(0, max_height - k2.shape[0]), (0, max_width - k2.shape[1])], mode='constant', constant_values=0)

        k1_fft = fft2(k1, )
        k2_fft = fft2(np.rot90(k2, 2), )

        return np.real(ifft2(k1_fft * k2_fft)).astype(np.float32)
    @staticmethod
    def aligned_cc(k1: np.ndarray, k2: np.ndarray) -> dict:
        """
        Aligned PRNU cross-correlation
        :param k1: (n1,nk) or (n1,nk1,nk2,...)
        :param k2: (n2,nk) or (n2,nk1,nk2,...)
        :return: {'cc':(n1,n2) cross-correlation matrix,'ncc':(n1,n2) normalized cross-correlation matrix}
        """

        # Type cast
        k1 = np.array(k1).astype(np.float32)
        k2 = np.array(k2).astype(np.float32)

        ndim1 = k1.ndim
        ndim2 = k2.ndim
        assert (ndim1 == ndim2)

        k1 = np.ascontiguousarray(k1).reshape(k1.shape[0], -1)
        k2 = np.ascontiguousarray(k2).reshape(k2.shape[0], -1)

        assert (k1.shape[1] == k2.shape[1])

        k1_norm = np.linalg.norm(k1, ord=2, axis=1, keepdims=True)
        k2_norm = np.linalg.norm(k2, ord=2, axis=1, keepdims=True)

        k2t = np.ascontiguousarray(k2.transpose())

        cc = np.matmul(k1, k2t).astype(np.float32)
        ncc = (cc / (k1_norm * k2_norm.transpose())).astype(np.float32)

        return {'cc': cc, 'ncc': ncc}
    @staticmethod
    def pce(cc: np.ndarray, neigh_radius: int = 2) -> dict:
        """
        PCE position and value
        :param cc: as from crosscorr2d
        :param neigh_radius: radius around the peak to be ignored while computing floor energy
        :return: {'peak':(y,x), 'pce': peak to floor ratio, 'cc': cross-correlation value at peak position
        """
        assert (cc.ndim == 2)
        assert (isinstance(neigh_radius, int))

        out = dict()

        max_idx = np.argmax(cc.flatten())
        max_y, max_x = np.unravel_index(max_idx, cc.shape)

        peak_height = cc[max_y, max_x]

        cc_nopeaks = cc.copy()
        cc_nopeaks[max_y - neigh_radius:max_y + neigh_radius, max_x - neigh_radius:max_x + neigh_radius] = 0

        pce_energy = np.mean(cc_nopeaks.flatten() ** 2)

        if pce_energy * np.sign(peak_height) == 0:
            print('PCE error: zero correlation values')
            print(pce_energy)
            print(peak_height)
            return None

        out['peak'] = (max_y, max_x)
        out['pce'] = (peak_height ** 2) / pce_energy * np.sign(peak_height)
        out['cc'] = peak_height

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        from matplotlib import pyplot as plt

        if not os.path.exists('output/pce_histograms'):
            os.makedirs('output/pce_histograms')

        save_path = f'output/pce_histograms/histogram_{timestamp}.png'
        plt.hist(cc.flatten(), bins=50, color='blue', alpha=0.7)
        plt.axvline(x=peak_height, color='red', linestyle='dashed', linewidth=2, label='Peak Height')

        plt.yscale('log')

        plt.title('Cross-correlation Histogram with PCE Peak (Log Scale)')
        plt.xlabel('Cross-correlation Values')
        plt.ylabel('Frequency (Log Scale)')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

        return out

    """
    Statistical functions
    """
    @staticmethod
    def stats(cc: np.ndarray, gt: np.ndarray, ) -> dict:
        """
        Compute statistics
        :param cc: cross-correlation or normalized cross-correlation matrix
        :param gt: boolean multidimensional array representing groundtruth
        :return: statistics dictionary
        """
        assert (cc.shape == gt.shape)
        assert (gt.dtype == bool)

        assert (cc.shape == gt.shape)
        assert (gt.dtype == bool)

        fpr, tpr, th = roc_curve(gt.flatten(), cc.flatten())
        auc_score = auc(fpr, tpr)

        # EER
        eer_idx = np.argmin((fpr - (1 - tpr)) ** 2, axis=0)
        eer = float(fpr[eer_idx])

        outdict = {
            'tpr': tpr,
            'fpr': fpr,
            'th': th,
            'auc': auc_score,
            'eer': eer,
        }

        return outdict
    @staticmethod
    def gt(l1: list or np.ndarray, l2: list or np.ndarray) -> np.ndarray:
        """
        Determine the Ground Truth matrix given the labels
        :param l1: fingerprints labels
        :param l2: residuals labels
        :return: groundtruth matrix
        """

        l1 = np.array(l1)
        l2 = np.array(l2)

        assert (l1.ndim == 1)
        assert (l2.ndim == 1)

        gt_arr = np.zeros((len(l1), len(l2)), bool)

        for l1idx, l1sample in enumerate(l1):
            gt_arr[l1idx, l2 == l1sample] = True

        return gt_arr
    @staticmethod
    def compute_fingerprints(ff_dirlist, ff_device, crop):
        print('Computing fingerprints')
        fingerprint_devices = sorted(np.unique(ff_device))

        k = []
        for device in fingerprint_devices:

            if device in os.listdir('fingerprints'):
                print('Device '+device+' already registered. Loading fingerprint...')
                fingerprint = PickleHandler.load('fingerprints/' + device)
                k += [fingerprint]
                continue

            imgs = []
            for img_path in ff_dirlist[ff_device == device]:
                im = Image.open(img_path).convert('RGB')
                im_arr = np.asarray(im)
                if im_arr.dtype != np.uint8:
                    print('Error while reading image: {}'.format(img_path))
                    continue
                if im_arr.ndim != 3:
                    print('Image is not RGB: {}'.format(img_path))
                    continue
                im_cut = PRNUProcessor.cut_ctr(im_arr, crop)
                imgs += [im_cut]

            k += [PRNUProcessor.extract_multiple_aligned(imgs, processes=cpu_count())]
            # Register new fingerprint
            PickleHandler.save(PRNUProcessor.extract_multiple_aligned(imgs, processes=cpu_count()), 'fingerprints/'+device)

        return np.stack(k, 0)
    @staticmethod
    def compute_residuals(nat_dirlist, crop, image_manipulation):
        print('Computing residuals')
        imgs = []
        for img_path in nat_dirlist:
            img = Image.open(img_path).convert('RGB')
            img = PRNUProcessor.manipulateImage(img, image_manipulation)
            imgs += [PRNUProcessor.cut_ctr(np.asarray(img), crop)]

        pool = Pool(cpu_count())
        w = pool.map(PRNUProcessor.extract_single, imgs)
        pool.close()

        return np.stack(w, 0)
