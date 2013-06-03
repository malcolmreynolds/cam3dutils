from setuptools import setup

setup(name='cam3dutils',
      version='0.1',
      description='Utilies for Homogeneous transforms & 3d cameras',
      url='http://github.com/malcolmreynolds/cam3dutils',
      author='Malcolm Reynolds',
      author_email='malcolm.reynolds@gmail.com',
      license='BSD',
      packages=['cam3dutils'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
      ],
      zip_safe=False)
