import os
import sys
import math
import torch
import numpy as np
import cv2
import erqa

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""
    def __init__(self):
        self.name = "PSNR"
        self.val = []

    def __call__(self, preds, gt):
        mse = np.mean((preds - gt) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        self.val.append(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

    def __str__(self):
        return f"{self.name} : {sum(self.val)/len(self.val)}"

class ERQA:
    """
    Edge Restoration Quality Assessment
    img1 and img2 have range [0, 255]
    """
    def __init__(self):
        self.name = "ERQA"
        self.val = []
        self.metric = erqa.ERQA()

    def __call__(self, preds, gt):
        return self.val.append(self.metric(preds, gt))
    
    def __str__(self):
        return f"{self.name} : {sum(self.val)/len(self.val)}"

class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"
        self.val = []

    def __call__(self, preds, gt):
        if not preds.shape == gt.shape:
            raise ValueError("Input images must have the same dimensions.")
        if preds.ndim == 2:  # Grey or Y-channel image
            return self._ssim(preds, gt)
        elif preds.ndim == 3:
            if preds.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(preds, gt))
                return self.val.append(np.array(ssims).mean())
            elif preds.shape[2] == 1:
                return self._ssim(np.squeeze(preds), np.squeeze(gt))
        else:
            raise ValueError("Wrong input image dimensions.")

    def _ssim(self, preds, gt):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        preds = preds.astype(np.float64)
        gt = gt.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(preds, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(preds ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(gt ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(preds * gt, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def __str__(self):
        return f"{self.name} : {sum(self.val)/len(self.val)}"