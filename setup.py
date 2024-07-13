from setuptools import setup

setup(
    name='ebayesthresh_torch',
    version='0.0.1',
    description='ebayesthresh pip install',
    url='https://github.com/seoyeonc/ebayesthresh_torch.git',
    author='seoyeonchoi',
    author_email='chltjdus1212@gmail.com',
    license='seoyeonc',
    packages=['ebayesthresh'],
    zip_safe=False,
    install_requires=[
        'torch==2.0.1',
        'scipy==1.10.1',
        'statsmodels==0.14.0']
)