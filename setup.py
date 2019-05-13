import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(name='omniart_eye_classifier',
                 version='0.1.2',
                 description='A CNN that can classify eyes by their respective colour',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='http://github.com/rogierknoester/omniart_eye_classifier',
                 author='Rogier Knoester',
                 author_email='knoesterrogier+omniart@gmail.com',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 zip_safe=False)
