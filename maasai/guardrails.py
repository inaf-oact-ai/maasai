from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import re

# - ADDON MODULES
from langdetect import detect, DetectorFactory
# Set seed for deterministic results (optional, for testing)
DetectorFactory.seed = 0

##################################################
###          GLOBALS
##################################################
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}")
ASTRO_KEYWORDS = {
	"astronomy", "astrophysics", "telescope", "galaxy", "star", "stellar",
	"radio", "x-ray", "xray", "spectrum", "spectra", "light curve",
	"exoplanet", "cosmology", "neutron", "sun", "solar", "flare",
	"image", "fits", "catalog", "survey", "sdss", "gaia", "alma",
}

##################################################
###          HELPERS
##################################################
def is_ascii_text(text: str) -> bool:
	try:
		text.encode("ascii")
	except UnicodeEncodeError:
		return False
	return True

def is_probably_english(text: str) -> bool:
	"""
		Checks if the provided text is English.
		Returns True if English, False otherwise.
	"""
	try:
		return detect(text) == 'en'
	except:
		# Returns False if the text is empty or non-textual
		return False

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
