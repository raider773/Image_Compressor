"""Compressor module"""
from typing import List, Union

import numpy as np
from scipy.sparse import coo_matrix


class Sparse:
    """Sparse"""

    def __init__(self, data_real, data_imag, rows, cols):
        self.data_real = data_real
        self.data_imag = data_imag
        self.rows = rows
        self.cols = cols

    def toarray(self):
        return self.float_to_complex(
            coo_matrix((self.data_real, (self.rows, self.cols))).toarray(),
            coo_matrix((self.data_imag, (self.rows, self.cols))).toarray(),
        )

    @staticmethod
    def float_to_complex(real, imag):
        return np.vectorize(complex)(real, imag).astype(np.complex64)


class Compressor:
    """Compressor"""

    def __init__(
        self,
        top_percent_threshold: float = 0.025,
        scaler: int = 5,
        eps: float = 1e-16,
        compressed_dtype=np.int16,
    ):
        self.top_percent_threshold = top_percent_threshold
        self.scaler = scaler
        self.eps = eps
        self.compressed_dtype = compressed_dtype
        
        #scaler: the number the matrix is divided when compressed and multiply when decompressed
        #top_percent_threshold: the top % of individual pixels you keep after calculating argument of complex number
        #eps: when compressing the picture, use eps in the imaginary or real number to avoid errors when that particular number is 0

    def compress(self, img: np.ndarray, axis: int = 2) -> List[Sparse]:
        """Compress"""
        if len(img.shape) == 2:
            return self._compress_single(img)
        elif len(img.shape) > 2:
            return [
                self._compress_single(i)
                for i in [img[:, :, i] for i in range(img.shape[axis])]
            ]
        else:
            raise ValueError(
                f"Dimension error: {img.shape},need > 2 dimensions."
            )

    def _compress_single(self, channel: np.ndarray) -> Sparse:
        fft_coeffs = np.fft.fft2(channel / (255 * self.scaler)).astype(np.complex64)

        sorted_coeffs = np.sort(np.abs(fft_coeffs.flatten()))
        threshold = sorted_coeffs[
            int((1 - self.top_percent_threshold) * len(sorted_coeffs))
        ]

        spectral_img = fft_coeffs * (np.abs(fft_coeffs) > threshold)
        spectral_img[(spectral_img.imag == 0) & ((spectral_img.real != 0))] += complex(
            0, self.eps
        )
        spectral_img[(spectral_img.real == 0) & ((spectral_img.imag != 0))] += complex(
            self.eps, 0
        )

        sparse_real = coo_matrix(spectral_img.real).astype(self.compressed_dtype)
        sparse_imag = coo_matrix(spectral_img.imag).astype(self.compressed_dtype)
        return Sparse(
            data_real=sparse_real.data,
            data_imag=sparse_imag.data,
            rows=sparse_real.row,
            cols=sparse_real.col,
        )

    @staticmethod
    def _decompress_single(sparse_matrix):
        return np.fft.ifft2(sparse_matrix.toarray()).real

    def decompress(self, sparse: Union[Sparse, List]):
        """Decompress"""
        if isinstance(sparse, Sparse):
            return self._decompress_single(sparse) * self.scaler
        elif isinstance(sparse, List):
            decomp = [self._decompress_single(i) for i in sparse]
            reconstr_img = np.zeros(
                shape=(*decomp[0].shape, len(decomp))
            )  # hardcoded 0. Change this
            for i, channel in enumerate(decomp):
                reconstr_img[:, :, i] = channel
            return reconstr_img * self.scaler
        else:
            raise ValueError("Error in param 'sparse'")
