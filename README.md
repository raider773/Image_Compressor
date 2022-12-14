# Image_Compressor
Image compressor using Fourier transform, sparse matrix and parallelization

Custom image compressor made for a particular problem at work.
Given a dictionary (key : item_id, value: numpy array of picture), we wanted to reduce the storage space of the images.
We apply Fourier transform to the images and get the matrix of complex numbers. We calculate the argument of each pixel, and keep the best ones given a certain top percentage. When we have this pixels, we separete the real and the imaginary numbers, and store them in a different sparce matrix. We reduce the image store space by 25-35% approximately. We used this instead of, for example, storing the pictures in jpeg, because we needed to store the images as pickles in a bucket in big query when not being used. Instead of storing 199M jpeg, we stored lists of sparse matrixes as pickles, divided by country and category based on our use case. In the case of an RGB picture, we compress each channel independently, so the result is a list of 6 sparce matrixes per picture. (3 made of real numbers and 3 made of imaginary numbers). We then combine the decompress process (the opposite steps from the compress process) in the generator used to feed the neural net. With the storage saved from the compression, we are able to increase batch size and avoid reaching maximum ram capacity.

Jupyter notebook shows an example.
