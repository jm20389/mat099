import numpy as np
import pywt
from numpy.fft import fft2, ifft2
from scipy.ndimage import filters
from multiprocessing import Pool, cpu_count
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from classes.ImageProcessor import ImageProcessor

class PRNUProcessor(ImageProcessor):

    """
    PRNU functions from https://github.com/sim-pez/prnu/
    """

    """
    Extraction functions
    """

    @staticmethod
    def extract_single(im: np.ndarray, levels: int = 4, sigma: float = 5, wdft_sigma: float = 0) -> np.ndarray:
        W = PRNUProcessor.noise_extract(im, levels, sigma)
        W = PRNUProcessor.rgb2gray(W)
        W = PRNUProcessor.zero_mean_total(W)
        W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
        W = PRNUProcessor.wiener_dft(W, W_std).astype(np.float32)
        return W
    @staticmethod
    def noise_extract(im: np.ndarray, levels: int = 4, sigma: float = 5) -> np.ndarray:
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

            for wlet_level_idx, wlet_level in enumerate(wlet_details):
                level_coeff_filt = [None] * 3
                for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                    level_coeff_filt[wlet_coeff_idx] = PRNUProcessor.wiener_adaptive(wlet_coeff, noise_var)
                wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

            wlet[1:] = wlet_details_filter
            wlet[0][...] = 0
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
    def noise_extract_compact(args):
        w = PRNUProcessor.noise_extract(*args)
        im = args[0]
        return (w * im / 255.).astype(np.float32)
    @staticmethod
    def extract_multiple_aligned(imgs: list, levels: int = 4, sigma: float = 5, processes: int = None,
                                 batch_size=cpu_count(), tqdm_str: str = '') -> np.ndarray:
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

        else:
            for im in tqdm(imgs, disable=tqdm_str is None, desc=tqdm_str, dynamic_ncols=True):
                RPsum += PRNUProcessor.noise_extract_compact((im, levels, sigma))
                NN += (PRNUProcessor.inten_scale(im) * PRNUProcessor.saturation(im)) ** 2

        K = RPsum / (NN + 1)
        K = PRNUProcessor.rgb2gray(K)
        K = PRNUProcessor.zero_mean_total(K)
        K = PRNUProcessor.wiener_dft(K, K.std(ddof=1)).astype(np.float32)
        return K
    @staticmethod
    def crop(array: np.ndarray, sizes: tuple) -> np.ndarray:
        array = array.copy()
        if not (array.ndim == len(sizes)):
            raise ArgumentError('array.ndim must be equal to len(sizes)')
        for axis in range(array.ndim):
            axis_target_size = sizes[axis]
            axis_original_size = array.shape[axis]
            if axis_target_size > axis_original_size:
                raise ValueError('Can\'t have target size {} for axis {} with original size {}'.format(
                    axis_target_size, axis, axis_original_size
                ))
            if axis_target_size == axis_original_size:
                continue
            axis_offset = (axis_original_size - axis_target_size) // 2
            slice_start = axis_offset
            slice_end = axis_original_size - axis_offset
            array = np.take(array, range(slice_start, slice_end), axis=axis)
        return array
    @staticmethod
    def wiener_dft(im: np.ndarray, sigma: float) -> np.ndarray:
        im_fft = fft2(im)
        im_fft = im_fft / (np.abs(im_fft) ** 2 / im.size + sigma ** 2)
        return np.abs(ifft2(im_fft))
    @staticmethod
    def zero_mean(im: np.ndarray) -> np.ndarray:
        return im - im.mean()
    @staticmethod
    def zero_mean_total(im: np.ndarray) -> np.ndarray:
        return im - im.mean()
    @staticmethod
    def rgb2gray(im: np.ndarray) -> np.ndarray:
        assert (im.dtype == np.float32 or im.dtype == np.uint8)
        if im.ndim == 2:
            return im.copy()
        elif im.ndim == 3:
            if im.shape[2] == 1:
                return im[:, :, 0].copy()
            if im.shape[2] == 3:
                return np.dot(im, [0.299, 0.587, 0.114]).astype(np.float32)
            raise ValueError('Impossible to convert RGB to grayscale for shape: {}'.format(im.shape))
        else:
            raise ValueError('Impossible to convert image to grayscale for shape: {}'.format(im.shape))
    @staticmethod
    def threshold(wlet_coeff_energy_avg: np.ndarray, noise_var: float) -> np.ndarray:
        return np.maximum(wlet_coeff_energy_avg - noise_var, 0)
    @staticmethod
    def wiener_adaptive(x: np.ndarray, noise_var: float, **kwargs) -> np.ndarray:
        if noise_var == 0:
            return x
        beta = noise_var / x.var()
        if beta < 1:
            xstd = x.std()
            return x * (1 - beta) + xstd * np.sqrt(beta) * np.random.randn(*x.shape)
        else:
            return x
    @staticmethod
    def inten_scale(im: np.ndarray) -> np.ndarray:
        return im / 255.
    @staticmethod
    def saturation(im: np.ndarray) -> np.ndarray:
        if im.ndim == 2:
            raise ArgumentError('Saturation can be computed only on RGB images')
        im = im.astype(np.float32)
        r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
        s = np.sqrt(((r - g) * (r - g) + (r - b) * (g - b)) / ((r + g + b) * 0.01 + 1))
        return s
    @staticmethod
    def inten_sat_compact(args):
        im = args[0]
        return PRNUProcessor.inten_scale(im) * PRNUProcessor.saturation(im)

    """
    Cross-correlation functions
    """

    @staticmethod
    def crosscorr_2d(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(k1) * np.fft.fft2(k2).conj()).real
    @staticmethod
    def aligned_cc(k1: np.ndarray, k2: np.ndarray) -> dict:
        h, w = k1.shape
        k1_centered = np.roll(np.roll(k1, -(h // 2), axis=0), -(w // 2), axis=1)
        cc = PRNUProcessor.crosscorr_2d(k1_centered, k2)
        cc_centered = np.roll(np.roll(cc, h // 2, axis=0), w // 2, axis=1)
        ncc = cc_centered / np.sqrt(((k1 ** 2).sum() * (k2 ** 2).sum()))
        max_corr = np.unravel_index(np.argmax(ncc), ncc.shape)
        shift = [h // 2 - max_corr[0], w // 2 - max_corr[1]]
        return {'ncc': ncc, 'shift': shift}

    """
    Statistical functions
    """

    @staticmethod
    def stats(cc: np.ndarray, gt: np.ndarray) -> dict:
        cc_values = cc[gt > 0]
        cc_nonzero = cc_values[cc_values > 0]
        cc_zero = cc_values[cc_values == 0]

        roc_curve_fpr, roc_curve_tpr, _ = roc_curve(gt.ravel() > 0, cc.ravel())
        roc_curve_area = auc(roc_curve_fpr, roc_curve_tpr)

        return {
            'cc_min': cc_values.min(),
            'cc_max': cc_values.max(),
            'cc_mean': cc_nonzero.mean() if cc_nonzero.size > 0 else np.nan,
            'cc_median': np.median(cc_nonzero) if cc_nonzero.size > 0 else np.nan,
            'cc_std': cc_nonzero.std(ddof=1) if cc_nonzero.size > 0 else np.nan,
            'cc_zero_fraction': cc_zero.size / cc_values.size,
            'roc_curve_fpr': roc_curve_fpr,
            'roc_curve_tpr': roc_curve_tpr,
            'roc_curve_area': roc_curve_area
        }
    @staticmethod
    def greater_than(l1: list or np.ndarray, l2: list or np.ndarray) -> np.ndarray:
        if not (isinstance(l1, (list, np.ndarray)) and isinstance(l2, (list, np.ndarray))):
            raise ArgumentError('l1 and l2 must be lists or numpy arrays')
        if isinstance(l1, list):
            l1 = np.array(l1)
        if isinstance(l2, list):
            l2 = np.array(l2)
        if l1.shape != l2.shape:
            raise ArgumentError('l1 and l2 must have the same shape')
        if l1.dtype != l2.dtype:
            raise ArgumentError('l1 and l2 must have the same dtype')
        if l1.ndim != l2.ndim:
            raise ArgumentError('l1 and l2 must have the same number of dimensions')
        return (l1 > l2).astype(np.uint8)
