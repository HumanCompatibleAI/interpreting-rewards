
import numpy as np

import scipy.ndimage
import skimage.transform

"""
Note that this code closely follows that of:
    https://github.com/greydanus/visualize_atari/blob/master/saliency.py
See Greydanus et al., Visualizing and Understanding Atari Agents. Url: https://arxiv.org/abs/1711.00138
"""

def get_mask(center, size, r):
    """Creates a normalized mask (np.ndarray) of shape `size`, radius `r`, centered at `center`."""
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = scipy.ndimage.gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def occlude(img, mask):
    """Uses `mask` to occlude a region of `img`."""
    assert len(image.shape) == 4, and image.shape[0] == 1, "`img` must have shape (1, h, w, n) where k >= 1"
    img = np.copy(img)
    n = img.shape[-1]
    for k in range(n):
        I = img[0, :, :, k]
        img[0, :, :, k] = I*(1-mask) + scipy.ndimage.gaussian_filter(I, sigma=3)*mask
    return img

def compute_saliency_map(reward_model, obs, output_shape=(210, 160), stride=5, radius=5):
    """Computes a saliency map of `reward_model` over `obs`, reshaped to `ouput_shape`.""" 
    baseline = reward_model(obs).detach().cpu().numpy()
    _, h, w, _ = obs.shape
    scores = np.zeros((h // stride + 1, w // stride + 1))
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            mask = get_mask(center=(i, j), size=(h, w), r=radius)
            obs_perturbed = occlude(obs, mask)
            perturbed_reward = reward_model(obs_perturbed).detach().cpu().numpy()
            scores[i // stride, j // stride] = 0.5 * np.abs(perturbed_reward - baseline) ** 2
    # pmax = scores.max()
    scores = skimage.transform.resize(scores, output_shape=output_shape)
    scores = scores.astype(np.float32)
    # return pmax * scores / scores.max()
    return scores / scores.max()

def add_saliency_to_frame(frame, saliency, channel=1):
    """Impose saliency map `saliency` over image `frame`."""
    pmax = saliency.max()
    I = frame.astype('uint16')
    I[:, :, channel] += (frame.max() * saliency).astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I

