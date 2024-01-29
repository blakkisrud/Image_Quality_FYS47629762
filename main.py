"""
Script to show some ilustrations of spatial resolution and noise
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import sys
import os
from scipy.fft import fft2, fftshift, ifft2, fft, fftshift
from scipy import ndimage
import nibabel as nib
import nrrd

# Make a computational image of an X-ray phantom-ish-structure

NOISE_PART = False
output_path = 'output/'

if not os.path.exists(output_path):
    os.mkdir(output_path)


def gaussian_poisson_noise(image, m):
    """
    Function to add gaussian noise to an image
    """

    sigma = np.sqrt(m)

    noise = np.random.normal(m, sigma, image.shape)

    return image + noise


def make_phantom_three_circle(r1, r2, r3, c1, c2, c3):

    phantom_size = 256

    phantom = np.zeros((phantom_size, phantom_size))

    # Set the values of the pixels in the phantom

    # Set the background value
    phantom.fill(0.0)

    # Make four small circles with increasing sizes

    # Set the value of the circles
    circle_value_1 = c1
    circle_value_2 = c2
    circle_value_3 = c3

    # Set the radius of the circles
    circle_radius_1 = r1
    circle_radius_2 = r2
    circle_radius_3 = r3

    # Set the center of the circles
    circle_center_1 = 40
    circle_center_2 = 75
    circle_center_3 = 100

    # Fill in the circles

    # Get the x and y coordinates of the circles
    x = np.arange(phantom_size)
    y = np.arange(phantom_size)

    # Get the x and y coordinates of the circles
    xv, yv = np.meshgrid(x, y)

    # Calculate the distance of each pixel from the center of the circles
    r1 = np.sqrt((xv - circle_center_1)**2 + (yv - circle_center_1)**2)
    r2 = np.sqrt((xv - circle_center_2)**2 + (yv - circle_center_2)**2)
    r3 = np.sqrt((xv - circle_center_3)**2 + (yv - circle_center_3)**2)

    # Set the pixels within the circles to the circle value
    phantom[r1 <= circle_radius_1] = circle_value_1
    phantom[r2 <= circle_radius_2] = circle_value_2
    phantom[r3 <= circle_radius_3] = circle_value_3

    return phantom


def make_phantom(circle_value, square_value, background_value, output_path='output/'):

    # Set the size of the phantom
    phantom_size = 256
    # background_value = 0.0

    # Create a 2D array of the correct size filled with zeros

    phantom = np.zeros((phantom_size, phantom_size))

    # Set the values of the pixels in the phantom

    # Set the background value
    phantom.fill(background_value)

    # Make four small circles with increasing sizes

    # Set the value of the circles
    circle_value_1 = 250
    circle_value_2 = 250
    circle_value_3 = 250

    # Set the radius of the circles
    circle_radius_1 = 10
    circle_radius_2 = 5
    circle_radius_3 = 1

    # Set the center of the circles
    circle_center_1 = (40, 200)
    circle_center_2 = (75, 200)
    circle_center_3 = (100, 200)

    # Fill in the circles

    # Get the x and y coordinates of the circles
    x = np.arange(phantom_size)
    y = np.arange(phantom_size)

    # Get the x and y coordinates of the circles
    xv, yv = np.meshgrid(x, y)

    # Calculate the distance of each pixel from the center of the circles
    r1 = np.sqrt((xv - circle_center_1[0])**2 + (yv - circle_center_1[1])**2)
    r2 = np.sqrt((xv - circle_center_2[0])**2 + (yv - circle_center_2[1])**2)
    r3 = np.sqrt((xv - circle_center_3[0])**2 + (yv - circle_center_3[1])**2)

    # Set the pixels within the circles to the circle value
    phantom[r1 <= circle_radius_1] = circle_value_1
    phantom[r2 <= circle_radius_2] = circle_value_2
    phantom[r3 <= circle_radius_3] = circle_value_3

    # Set the value of the circle
    # circle_value =

    # Set the radius of the circle
    circle_radius = 25

    # Set the center of the circle
    circle_center = (40, 40)

    # Set the value of the square
    # square_value = 50

    # Set the size of the square
    square_size = 20

    # Set the center of the square
    square_center = (128, 128)

    # Fill in the circle

    # Get the x and y coordinates of the circle
    x = np.arange(phantom_size)
    y = np.arange(phantom_size)

    # Get the x and y coordinates of the circle
    xv, yv = np.meshgrid(x, y)

    # Calculate the distance of each pixel from the center of the circle
    r = np.sqrt((xv - circle_center[0])**2 + (yv - circle_center[1])**2)

    # Set the pixels within the circle to the circle value
    phantom[r <= circle_radius] = circle_value

    # Fill in the square

    # Get the x and y coordinates of the square
    x = np.arange(phantom_size)
    y = np.arange(phantom_size)

    # Get the x and y coordinates of the square
    xv, yv = np.meshgrid(x, y)

    # Calculate the distance of each pixel from the center of the square
    r = np.sqrt((xv - square_center[0])**2 + (yv - square_center[1])**2)

    # Set the pixels within the square to the square value
    phantom[(np.abs(xv - square_center[0]) <= square_size/2) &
            (np.abs(yv - square_center[1]) <= square_size/2)] = square_value

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(phantom, cmap='gray', vmin=0, vmax=255, interpolation='none')

    plt.savefig(os.path.join(output_path, 'phantom.png'))

    return phantom


def gaussian_psf(shape, sigma):

    x, y = np.meshgrid(
        np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]))
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= np.sum(psf)

    return psf


def plot_line_profile(image, x, y, direction, savefig_name=None):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    if direction == 'x':
        line_profile = image[y, :]
    elif direction == 'y':
        line_profile = image[:, x]

    ax.plot(line_profile, 'o')

    if savefig_name is not None:
        plt.savefig(os.path.join(output_path, savefig_name))

    else:

        plt.show()


def blur_phantom(phantom, psf, output_path='output/', savefig=False, filename='convolved_phantom.png'):

    # Set the mode of the convolution to 'same' to keep the output image the same size as the input image
    mode = 'same'

    # Convolve the phantom with the point spread function
    convolved_phantom = convolve2d(phantom, psf, mode=mode)

    if savefig:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(convolved_phantom, cmap='gray',
                  vmin=0, vmax=255, interpolation='none')

        plt.savefig(os.path.join(output_path, filename))

    return convolved_phantom


phantom = make_phantom(200, 200, 10)
phantom_circles = make_phantom_three_circle(25, 10, 5, 1, 2, 3)

circle_values = 155
bg_value = 130

c1_I = phantom_circles == 1
c2_I = phantom_circles == 2
c3_I = phantom_circles == 3

label_im, nb_labels = ndimage.label(phantom_circles)

slices = ndimage.find_objects(label_im)

print(slices)

roi_box = np.zeros(phantom_circles.shape)

for sl, i in zip(slices, range(len(slices))):

    sl_x, sl_y = sl
    roi_box[sl_x.start:sl_x.stop, sl_y.start:sl_y.stop] = i+1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(roi_box, cmap='gray', interpolation='none')

plt.savefig(os.path.join(output_path, 'roi_box.png'))

# Get the bounding box of the circles

bg_I = phantom_circles == 0

phantom_noise_free = phantom_circles.copy()

phantom_noise_free[c1_I] = circle_values
phantom_noise_free[c2_I] = circle_values
phantom_noise_free[c3_I] = circle_values

phantom_noise_free[bg_I] = bg_value

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(phantom_noise_free, cmap='gray',
          vmin=0, vmax=255, interpolation='none')

plt.savefig(os.path.join(output_path, 'phantom_circles.png'))

# Fill with normal noise
phantom_circles[c1_I] = np.random.normal(
    circle_values, np.sqrt(circle_values), c1_I.sum())
phantom_circles[c2_I] = np.random.normal(
    circle_values, np.sqrt(circle_values), c2_I.sum())
phantom_circles[c3_I] = np.random.normal(
    circle_values, np.sqrt(circle_values), c3_I.sum())

phantom_circles[bg_I] = np.random.normal(
    bg_value, np.sqrt(bg_value), bg_I.sum())

# phantom_circles = blur_phantom(phantom_circles, gaussian_psf((20, 20), 1e-1), savefig=True, filename='blured_phantom_circles.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(phantom_circles, cmap='gray', vmin=0, vmax=255, interpolation='none')

plt.savefig(os.path.join(output_path, 'phantom_circles_noise.png'))

nrrd.write(os.path.join(
    output_path, 'phantom_circles_noise.nrrd'), phantom_circles)

# Plot and save the distribution of the noise

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(phantom_circles[c1_I], bins=100,
        density=True, alpha=0.5, label='Circle 1')

plt.savefig(os.path.join(output_path, 'histogram_circle_1.png'))

# Calculate the SNR and CNR

# SNR = (signal - noise) / noise
# CNR = (signal1 - signal2) / noise

# CNR

mean_x = np.mean(phantom_circles[c1_I])
mean_bg = np.mean(phantom_circles[bg_I])
std_bg = np.std(phantom_circles[bg_I])

print('CNR: ', (mean_x - mean_bg) / std_bg)

# Calculate the SNR

mean_bg = np.mean(phantom_circles[bg_I])

signal_1 = (np.mean(phantom_circles[roi_box == 1])-mean_bg)
signal_2 = (np.mean(phantom_circles[roi_box == 2])-mean_bg)
signal_3 = (np.mean(phantom_circles[roi_box == 3])-mean_bg)

snr_1 = signal_1/np.std(phantom_circles[bg_I])
snr_2 = signal_2/np.std(phantom_circles[bg_I])
snr_3 = signal_3/np.std(phantom_circles[bg_I])

print(snr_1, snr_2, snr_3)

# set values

print(np.std(phantom_circles[phantom_circles == 1]))
print(np.std(phantom_circles[phantom_circles == 2]))
print(np.std(phantom_circles[phantom_circles == 3]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(phantom_circles, cmap='gray', vmin=0, vmax=255, interpolation='none')

plt.savefig(os.path.join(output_path, 'phantom_circles.png'))

# Make a fourier transform of the phantom

phantom_fft = fft2(phantom)
phantom_fft = fftshift(phantom_fft)
reconstructed_image = ifft2(fftshift(phantom_fft))
magnitude_spectrum = np.abs(phantom_fft)
magnitude_spectrum_log = np.log1p(magnitude_spectrum)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(magnitude_spectrum_log, cmap='gray', interpolation='none')

plt.savefig(os.path.join(output_path, 'phantom_fft.png'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.abs(reconstructed_image), cmap='gray', interpolation='none')

plt.savefig(os.path.join(output_path, 'reconstructed_image.png'))

psf_width_list = np.array([1e-2, 1e-1, 1])

for psf_w in psf_width_list:

    psf_size = 256
    size_of_psf = (psf_size, psf_size)
    psf = gaussian_psf(size_of_psf, psf_w)

    line_profile = psf[int(size_of_psf[0]/2), :]

    # Perform FFT on the line profile
    fft_result = fft(line_profile)

    # Shift zero frequency components to the center
    fft_result_shifted = fftshift(fft_result)

    # Calculate the magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = np.abs(fft_result_shifted)
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)

    # Display the original line profile and its FFT
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(line_profile)
    plt.title('Line Profile of the PSF')
    plt.xlabel('Pixel Position')
    plt.ylabel('Intensity')

    plt.subplot(122)
    plt.plot(np.arange(-psf_size // 2, psf_size // 2), magnitude_spectrum_log)
    plt.title('Magnitude Spectrum (Frequency Domain)')
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Magnitude (log scale)')
    plt.xlim(0, psf_size // 2)
    plt.grid(True)

    plt.savefig(os.path.join(output_path, 'psf_fft' + str(psf_w) + '.png'))


psf_1 = gaussian_psf((35, 35), 1e-1)
psf_2 = gaussian_psf((35, 35), 0.5)
psf_3 = gaussian_psf((35, 35), 1.0)

line1 = plot_line_profile(
    psf_1, 15, 15, 'x', savefig_name='psf1_line_profile.png')
line2 = plot_line_profile(
    psf_2, 15, 15, 'x', savefig_name='psf2_line_profile.png')
line3 = plot_line_profile(
    psf_3, 15, 15, 'x', savefig_name='psf3_line_profile.png')


psf_list = [psf_1, psf_2, psf_3]
i = 1

for psf in psf_list:

    # Save psf

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(psf, cmap='gray', interpolation='none')

    plt.savefig(os.path.join(output_path, 'psf' + str(i) + '.png'))

    i += 1

convolved_phantom_1 = blur_phantom(
    phantom, psf_1, savefig=True, filename='convolved_phantom_1.png')
convolved_phantom_2 = blur_phantom(
    phantom, psf_2, savefig=True, filename='convolved_phantom_2.png')
convolved_phantom_3 = blur_phantom(
    phantom, psf_3, savefig=True, filename='convolved_phantom_3.png')

if NOISE_PART:

    # Set the size of the phantom
    phantom_size = 256
    background_value = 0.0

    # Create a 2D array of the correct size filled with zeros

    phantom = np.zeros((phantom_size, phantom_size))

    # Set the values of the pixels in the phantom

    # Set the background value
    phantom.fill(background_value)

    # Set the value of the circle
    circle_value = 10

    # Set the radius of the circle
    circle_radius = 25

    # Set the center of the circle
    circle_center = (40, 40)

    # Set the value of the square
    # square_value = 50

    # Set the size of the square
    square_size = 20

    # Set the center of the square
    square_center = (128, 128)

    # Fill in the circle

    # Get the x and y coordinates of the circle
    x = np.arange(phantom_size)
    y = np.arange(phantom_size)

    # Get the x and y coordinates of the circle
    xv, yv = np.meshgrid(x, y)

    # Calculate the distance of each pixel from the center of the circle
    r = np.sqrt((xv - circle_center[0])**2 + (yv - circle_center[1])**2)

    # Set the pixels within the circle to the circle value
    phantom[r <= circle_radius] = circle_value

    # Fill in the square

    # Get the x and y coordinates of the square
    x = np.arange(phantom_size)
    y = np.arange(phantom_size)

    # Get the x and y coordinates of the square
    xv, yv = np.meshgrid(x, y)

    # Calculate the distance of each pixel from the center of the square
    r = np.sqrt((xv - square_center[0])**2 + (yv - square_center[1])**2)

    # Set the pixels within the square to the square value
    phantom[(np.abs(xv - square_center[0]) <= square_size/2) &
            (np.abs(yv - square_center[1]) <= square_size/2)] = square_value

    phantom = gaussian_poisson_noise(phantom, 200)

    # Display the phantom

    plt.imshow(phantom, cmap='gray', vmin=0, vmax=255)
    plt.show()

    plt.show()
