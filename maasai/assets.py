def _prepare_asset(item: dict[str, Any], ctx: NodeContext) -> PreparedAsset:
	path = item["path"]
	lower = path.lower()

	if lower.endswith((".fits", ".fit", ".fts")):
		preview_path = ctx.tools.fits_to_png(
			path=path,
			stretch="zscale",
			colormap="gray",
		)
		base64_data = encode_image_base64(preview_path)
		return PreparedAsset(
			path=path,
			kind="fits",
			mime_type="image/png",
			preview_path=preview_path,
			base64_data=base64_data,
			notes=["Converted from FITS using zscale preview"],
		)

	if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
		base64_data = encode_image_base64(path)
		return PreparedAsset(
			path=path,
			kind="image",
			mime_type=_guess_mime_type(path),
			preview_path=path,
			base64_data=base64_data,
		)

	return PreparedAsset(
		path=path,
		kind="other",
		notes=["Unsupported for multimodal preview"],
	)
