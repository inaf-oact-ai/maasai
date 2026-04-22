from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import os
import sys
import numpy as np
import io
import base64
from typing import Any

# - ASTRO/IMAGE MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
from PIL import Image

# - MAASAI MODULES
from .schemas import PreparedAsset
from maasai import logger

##################################################
###          HELPER METHODS
##################################################
def _prepare_asset(item: dict[str, Any], ctx: NodeContext) -> PreparedAsset:
	path = item["path"]
	lower = path.lower()
	
	if lower.endswith((".fits", ".fit", ".fts")):
		preview_path = None # not saving preview for the moment
		data_uint8 = _fits2png(path, save=False, outfile=preview_path)
		base64_data = _encode_image_base64(data_uint8)
		
		#print("DEBUG FITS ASSET")
		#print("path =", path)
		#print("data_uint8 is None =", data_uint8 is None)
		#print("base64_data is None =", base64_data is None)
		#print("base64 len =", 0 if base64_data is None else len(base64_data))
		
		asset= PreparedAsset(
			path=path,
			kind="fits",
			original_mime_type="application/fits",
			preview_mime_type="image/png",
			preview_path=preview_path,
			base64_data=base64_data,
			notes=["Converted from FITS using zscale preview"],
		)
		#print("DEBUG PREPARED ASSET MODEL")
		#print(asset)
		#print(asset.model_dump())
		
		return asset

	if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
		orig_mime = _guess_mime_type(path)
		base64_data = _encode_image_base64(path)
		return PreparedAsset(
			path=path,
			kind="image",
			original_mime_type=orig_mime,
			preview_mime_type=orig_mime,
			preview_path=path,
			base64_data=base64_data,
		)

	return PreparedAsset(
		path=path,
		kind="other",
		original_mime_type=_guess_mime_type(path),
		preview_mime_type=None,
		notes=["Unsupported for multimodal preview"],
	)
	
def _asset_field(asset, name, default=None):
	if isinstance(asset, dict):
		return asset.get(name, default)
	return getattr(asset, name, default)
	
def _guess_mime_type(path: str) -> str | None:
	lower = path.lower()

	if lower.endswith((".png",)):
		return "image/png"
	if lower.endswith((".jpg", ".jpeg")):
		return "image/jpeg"
	if lower.endswith((".webp",)):
		return "image/webp"
	if lower.endswith((".fits", ".fit", ".fts")):
		return "application/fits"

	return None	
	
def _encode_image_base64(image, fmt="PNG"):
	"""Encode an image to base64.

	Parameters
	----------
	image : str | np.ndarray | PIL.Image.Image
		Path to an image file, a uint8 numpy array, or a PIL image.
	fmt : str
		Output format used when image is an array/PIL image.

	Returns
	-------
	str | None
		Base64 string, or None on failure.
	"""
	try:
		# Case 1: path on disk
		if isinstance(image, str):
			with open(image, "rb") as f:
				return base64.b64encode(f.read()).decode("utf-8")

		# Case 2: numpy array
		if isinstance(image, np.ndarray):
			if image.dtype != np.uint8:
				raise ValueError(f"Expected uint8 array, got dtype={image.dtype}")

			img = Image.fromarray(image)

			buf = io.BytesIO()
			img.save(buf, format=fmt)
			return base64.b64encode(buf.getvalue()).decode("utf-8")

		# Case 3: PIL image
		if isinstance(image, Image.Image):
			buf = io.BytesIO()
			image.save(buf, format=fmt)
			return base64.b64encode(buf.getvalue()).decode("utf-8")

		raise TypeError(f"Unsupported image type: {type(image)}")

	except Exception as e:
		logger.error(f"Failed to encode image to base64 (err={str(e)})!")
		return None

def _fits2png(
	inputfile,
	hdu=0,
	zscale_data= True,
	contrast=0.25,
	subtract_bkg= False,
	clip_data= False,
	sigma_low=5.0,
	sigma_up=30.0,
	save=False,
	outfile=None
):
	""" Utility method to convert a FITS file to PNG """
	
	# - Read FITS
	try:
		data= fits.open(inputfile)[hdu].data
	except Exception as e:
		logger.error(f"Failed to open FITS file {inputfile} (hdu={hdu}), err={str(e)}!")
		return None
		
	# - Handle NaN pixels, (set to 0)
	cond= np.logical_and(data!=0, np.isfinite(data))
	data_1d= data[cond]
	data[~cond]= 0

	# - Subtract background
	if subtract_bkg:
		sigma_bkg= 3
		bkgval, _, _ = sigma_clipped_stats(data_1d, sigma=sigma_bkg)
		data_bkgsub= data - bkgval
		data= data_bkgsub

	# - Clip all pixels that are below sigma clip
	if clip_data:
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]
		res= sigma_clip(data_1d, sigma_lower=sigma_low, sigma_upper=sigma_up, masked=True, return_bounds=True)
		thr_low= float(res[1])
		thr_up= float(res[2])
		logger.debug(f"--> fits2png(): clip_data with thr_low={thr_low}, thr_up={thr_up} ...")

		data_clipped= np.copy(data)
		data_clipped[data_clipped<thr_low]= thr_low
		data_clipped[data_clipped>thr_up]= thr_up
		data= data_clipped

	# - Apply Zscale
	if zscale_data:
		transform = ZScaleInterval(contrast=contrast)
		data_stretched = transform(data)
		data= data_stretched

	# - Normalize to [0,255] range
	data_min= data.min()
	data_max= data.max()
	if data_max == data_min:
		data_uint8 = np.zeros_like(data, dtype=np.uint8)
	else:
		data_norm = 255.0 * (data - data_min) / (data_max - data_min)
		data_uint8 = data_norm.astype(np.uint8)
		
	# - Save to PNG file
	if save:
		if outfile is None:
			inputfile_base= os.path.basename(inputfile)
			inputfile_base_noext= os.path.splitext(inputfile_base)[0]
			outfile=inputfile_base_noext + '.png'
	
		# - Convert to PIL
		img = Image.fromarray(data_uint8)
		##img_rgb= Image.merge("RGB", (img, img, img))
		##img_rgb= img.convert('RGB')

		##img = Image.fromarray(data)
		
		# - Save to PNG file
		logger.info(f"Saving FITS image {inputfile} as PNG file {outfile} ...")
		##img.save(outfile, cmap='viridis')
		img.save(outfile)
		##img_rgb.save(outfile)
		##img_rgb.save(outfile, cmap='viridis')

	return data_uint8

