from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("../requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()

setup(
    name='nudge',
    version='0.5.0',
    author='Hikaru Shindo',
    author_email='hikisan.gouv',
    packages=find_packages(),
    include_package_data=True,
    # package_dir={'': 'nudge'},
    url='tba',
    description='Neurally gUided Differentiable loGic policiEs (NUDGE)',
    long_description=long_description,
    install_requires=requirements,
)
