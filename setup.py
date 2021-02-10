import setuptools

setuptools.setup(
    name='DeblendingStarfieldsDevRunjingLiu120',
    version='0.0.1',
    author='Runjing Liu',
    author_email='runjing_liu@berkeley.edu',
    packages=['deblending_runjingdev'],
    url='https://github.com/Runjing-Liu120/DeblendingStarfields',
    description='A package to reproduce experiment results in our deblending starfields paper',
    python_requires='>=3.6',
    install_requires=['astropy', 'numpy', 'pandas', 'matplotlib', 'scipy']
)
