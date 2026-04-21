from __future__ import annotations

import re


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}")


ASTRO_KEYWORDS = {
	"astronomy", "astrophysics", "telescope", "galaxy", "star", "stellar",
	"radio", "x-ray", "xray", "spectrum", "spectra", "light curve",
	"exoplanet", "cosmology", "neutron", "sun", "solar", "flare",
	"image", "fits", "catalog", "survey", "sdss", "gaia", "alma",
}


def is_probably_english(text: str) -> bool:
	try:
		text.encode("ascii")
	except UnicodeEncodeError:
		return False
	return True


def detect_pii(text: str) -> list[str]:
	reasons: list[str] = []
	if EMAIL_RE.search(text):
		reasons.append("email")
	if PHONE_RE.search(text):
		reasons.append("phone_number")
	return reasons


def is_scientific_or_astronomy_related(text: str) -> bool:
	lowered = text.lower()
	return any(keyword in lowered for keyword in ASTRO_KEYWORDS)


def wrap_guardrail_response(message: str) -> str:
	return message.strip()
