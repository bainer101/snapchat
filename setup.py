from setuptools import setup, find_packages
import os
import glob

assets = glob.glob("assets/*")

setup(
    name="filter",
    version="0.0b1",
    decription="Snapchat filters",
    long_description="Making Snapchat filters using OpenCV, simplified",
    author="Alexander Baine",
    author_email="abaine2001@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={"filter" : ["assets/*"]}
)
