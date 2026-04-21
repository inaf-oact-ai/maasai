#! /usr/bin/env python
"""
Setup for maasai
"""
import os
import sys
from setuptools import setup


def read(fname):
	"""Read a file"""
	return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
	""" Get the package version number """
	import maasai
	return maasai.__version__


PY_MAJOR_VERSION=sys.version_info.major
PY_MINOR_VERSION=sys.version_info.minor
print("PY VERSION: maj=%s, min=%s" % (PY_MAJOR_VERSION,PY_MINOR_VERSION))

reqs= []
reqs.append('numpy')
reqs.append('pydantic') 
reqs.append('mlflow')
reqs.append('langchain')
reqs.append('langchain-classic')
reqs.append('langchain-community')
reqs.append('langchain-core')
reqs.append('langchain-mcp-adapters')
reqs.append('langchain-openai')
reqs.append('langchain-text-splitters')
reqs.append('langgraph')
reqs.append('litellm')


setup(
	name="maasai",
	version=get_version(),
	author="Simone Riggi",
	author_email="simone.riggi@gmail.com",
	description="Agent-based tool for astronomical data analysis",
	license = "GPL3",
	url="https://github.com/inaf-oact-ai/maasai",
	keywords = ['astronomy', 'data', 'analysis', 'agents'],
	long_description=read('README.md'),
	long_description_content_type='text/markdown',
	download_url="https://github.com/inaf-oact-ai/maasai/archive/refs/tags/v1.0.0.tar.gz",
	packages=['maasai'],
	install_requires=reqs,
	scripts=['scripts/run.py'],
	classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Astronomy',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 3'
	]
)

