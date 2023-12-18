# Image_Compressor
Image compressor utilizing Fourier transform, sparse matrix, and parallelization

Custom image compressor.
Given a dictionary (key: item_id, value: numpy array of a picture), I aimed to reduce the storage space of the images.
I applied Fourier transform to the images and obtained the matrix of complex numbers. I calculated the argument of each pixel and retained the best ones given a certain top percentage. When I had these pixels, I separated the real and the imaginary numbers, storing them in different sparse matrices. The image storage space was reduced by approximately 25-35%. I chose this approach over, for example, storing the pictures in JPEG because I needed to store the images as pickles in a bucket in BigQuery when not in use. Instead of storing 199M JPEGs, I stored lists of sparse matrices as pickles, divided by country and category based on my use case. In the case of an RGB picture, I compressed each channel independently, resulting in a list of 6 sparse matrices per picture (3 made of real numbers and 3 made of imaginary numbers). I then combined the decompression process (the opposite steps from the compression process) in the generator used to feed the neural net. With the storage saved from the compression, I am able to increase the batch size and avoid reaching the maximum RAM capacity.

The Jupyter notebook shows an example.
