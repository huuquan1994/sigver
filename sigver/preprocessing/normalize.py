import numpy as np
from skimage import filters, transform
from typing import Tuple
from skimage.transform import resize
from skimage import img_as_ubyte


def preprocess_signature(img: np.ndarray,
						 canvas_size: Tuple[int, int],
						 img_size: Tuple[int, int] =(170, 242),
						 input_size: Tuple[int, int] =(150, 220)) -> np.ndarray:
	""" Pre-process a signature image, centering it in a canvas, resizing the image and cropping it.

	Parameters
	----------
	img : np.ndarray (H x W)
		The signature image
	canvas_size : tuple (H x W)
		The size of a canvas where the signature will be centered on.
		Should be larger than the signature.
	img_size : tuple (H x W)
		The size that will be used to resize (rescale) the signature
	input_size : tuple (H x W)
		The final size of the signature, obtained by croping the center of image.
		This is necessary in cases where data-augmentation is used, and the input
		to the neural network needs to have a slightly smaller size.

	Returns
	-------
	np.narray (input_size):
		The pre-processed image
	-------

	"""
	img = img.astype(np.uint8)
	centered = normalize_image(img, canvas_size)
	inverted = 255 - centered
	resized = resize_image(inverted, img_size)

	if input_size is not None and input_size != img_size:
		cropped = crop_center(resized, input_size)
	else:
		cropped = resized

	return cropped


def normalize_image(img: np.ndarray,
                    canvas_size: Tuple[int, int] = (840, 1360)) -> np.ndarray:
    """ Centers an image in a pre-defined canvas size, and remove
    noise using OTSU's method.

    Parameters
    ----------
    img : np.ndarray (H x W)
        The image to be processed
    canvas_size : tuple (H x W)
        The desired canvas size

    Returns
    -------
    np.ndarray (H x W)
        The normalized image
    """

    # 1) Crop the image before getting the center of mass

    # Apply a gaussian filter on the image to remove small components
    # Note: this is only used to define the limits to crop the image
    max_rows, max_cols = canvas_size

    blur_radius = 2
    blurred_image = filters.gaussian(img, blur_radius, preserve_range=True)

    # Binarize the image using OTSU's algorithm. This is used to find the center
    # of mass of the image, and find the threshold to remove background noise
    threshold = filters.threshold_otsu(img)

    # Find the center of mass
    binarized_image = blurred_image > threshold
    r, c = np.where(binarized_image == 0)
    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())

    # Crop the image with a tight box
    size = (400,650)
    y_min = r.min()-5 if (r.min()-5)>=0 else 0
    y_max = r.max()+5 if (r.max()+5)<=max_rows else max_rows
    x_min = c.min()-5 if (c.min()-5)>=0 else 0
    x_max = c.max()+5 if (c.max()+5)<=max_cols else max_cols
    
    cropped = img[y_min: y_max, x_min: x_max]
    cropped = img_as_ubyte(resize(cropped, size))    
    img_rows, img_cols = cropped.shape

    normalized_image = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    # 2) Center the image
    # Add the image to the blank canvas
    r_start = (max_rows-img_rows)//2 + 1
    c_start = (max_cols-img_cols)//2 + 1
    normalized_image[r_start:r_start + img_rows -2, c_start:c_start + img_cols -2] = cropped[1:-1, 1:-1]

    # Remove noise - anything higher than the threshold. Note that the image is still grayscale
    normalized_image[normalized_image > threshold] = 255

    return normalized_image


def remove_background(img: np.ndarray) -> np.ndarray:
		""" Remove noise using OTSU's method.

		Parameters
		----------
		img : np.ndarray
			The image to be processed

		Returns
		-------
		np.ndarray
			The image with background removed
		"""

		img = img.astype(np.uint8)
		# Binarize the image using OTSU's algorithm. This is used to find the center
		# of mass of the image, and find the threshold to remove background noise
		threshold = filters.threshold_otsu(img)

		# Remove noise - anything higher than the threshold. Note that the image is still grayscale
		img[img > threshold] = 255

		return img


def resize_image(img: np.ndarray,
				 size: Tuple[int, int]) -> np.ndarray:
	""" Crops an image to the desired size without stretching it.

	Parameters
	----------
	img : np.ndarray (H x W)
		The image to be cropped
	size : tuple (H x W)
		The desired size

	Returns
	-------
	np.ndarray
		The cropped image
	"""
	height, width = size

	# Check which dimension needs to be cropped
	# (assuming the new height-width ratio may not match the original size)
	width_ratio = float(img.shape[1]) / width
	height_ratio = float(img.shape[0]) / height
	if width_ratio > height_ratio:
		resize_height = height
		resize_width = int(round(img.shape[1] / height_ratio))
	else:
		resize_width = width
		resize_height = int(round(img.shape[0] / width_ratio))

	# Resize the image (will still be larger than new_size in one dimension)
	img = transform.resize(img, (resize_height, resize_width),
						   mode='constant', anti_aliasing=True, preserve_range=True)

	img = img.astype(np.uint8)

	# Crop to exactly the desired new_size, using the middle of the image:
	if width_ratio > height_ratio:
		start = int(round((resize_width-width)/2.0))
		return img[:, start:start + width]
	else:
		start = int(round((resize_height-height)/2.0))
		return img[start:start + height, :]


def crop_center(img: np.ndarray,
				size: Tuple[int, int]) -> np.ndarray:
	""" Crops the center of an image

		Parameters
		----------
		img : np.ndarray (H x W)
			The image to be cropped
		size: tuple (H x W)
			The desired size

		Returns
		-------
		np.ndarray
			The cRecentropped image
		"""
	img_shape = img.shape
	start_y = (img_shape[0] - size[0]) // 2
	start_x = (img_shape[1] - size[1]) // 2
	cropped = img[start_y: start_y + size[0], start_x:start_x + size[1]]
	return cropped


def crop_center_multiple(imgs: np.ndarray,
						 size: Tuple[int, int]) -> np.ndarray:
	""" Crops the center of multiple images

		Parameters
		----------
		imgs : np.ndarray (N x C x H_old x W_old)
			The images to be cropped
		size: tuple (H x W)
			The desired size

		Returns
		-------
		np.ndarray (N x C x H x W)
			The cropped images
		"""
	img_shape = imgs.shape[2:]
	start_y = (img_shape[0] - size[0]) // 2
	start_x = (img_shape[1] - size[1]) // 2
	cropped = imgs[:, :, start_y: start_y + size[0], start_x:start_x + size[1]]
	return cropped
