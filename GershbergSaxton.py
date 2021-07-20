from scipy.fft import fft, ifft
import cmath
import matplotlib.pyplot as plt
import PIL.Image as img
from numpy import asarray
import numpy as np

#%% angular spectrum methods (Fourier maths)


def padwidth(N, array): #0-padding function
    padwidth = ((int((N - array.shape[0]) / 2), int((N - array.shape[0]) / 2)),
                                    (int((N - array.shape[1]) / 2), int((N - array.shape[1]) / 2)),)
    # print("padwidth")
    # print(padwidth)
    return padwidth

def ASM_fw(f, cell_spacing, target_plane_dist, res_fac, k):
    ## Forward angular spectrum method

    # defs:
    # f is complex pressure amplitude (A) and phase (phi) in the form A*exp(i*phi)
    # cell_spacing is spacing between pixels in this complex pressure field in meters
    # target_plane_dist is the distance to the second field we want to propagate to (m)
    # keep res_fac at 1 (resolution)
    # k is wavenumber (2pi/lambda)

    f = np.kron(f, np.ones((res_fac, res_fac)))
    # Nfft = target_plane_shape[0] # new side length of array
    Nfft = len(f)
    kx = 2*np.pi*(np.arange(-Nfft/2,(Nfft)/2)/((cell_spacing/res_fac)*Nfft)) # kx vector
    kx = np.reshape(kx, (1,len(kx))) # shape correctly
    ky = kx #spacial frequencies
    f_pad = np.pad(f, padwidth(Nfft, np.zeros(f.shape)), # pad to make F the correct size
                          'constant', constant_values = 0)
    F = np.fft.fft2(f_pad) # 2D FT
    F = np.fft.fftshift(F) # Shift to the centre
    ## Propagate forwards; change signs to back-propagate
    H = np.exp(1j*np.lib.scimath.sqrt(k**2 - kx**2 - (ky**2).T)*target_plane_dist) # propagator function
    Gf = F*H # propagating the signal forward in Fourier space
    gf = np.fft.ifft2(np.fft.ifftshift(Gf)) # IFT & shift to return to real space
    return gf

def ASM_bw(f, cell_spacing, target_plane_dist, res_fac, k):
    ## Backward angular spectrum method

    f = np.kron(f, np.ones((res_fac, res_fac)))
    # Nfft = target_plane_shape[0] # new side length of array
    Nfft = len(f)
    kx = 2*np.pi*(np.arange(-Nfft/2,(Nfft)/2)/((cell_spacing/res_fac)*Nfft)) # kx vector
    kx = np.reshape(kx, (1,len(kx))) # shape correctly
    ky = kx #spacial frequencies
    f_pad = np.pad(f, padwidth(Nfft, np.zeros(f.shape)), # pad to make F the correct size
                          'constant', constant_values = 0)
    F = np.fft.fft2(f_pad) # 2D FT
    F = np.fft.fftshift(F) # Shift to the centre
    ## Propagate backwards; change signs to fwd-propagate
    H = np.exp(1j*np.lib.scimath.sqrt(k**2 + kx**2 + (ky**2).T)*target_plane_dist) # propagator function
    Gf = F*H # propagating the signal forward in Fourier space
    gf = np.fft.ifft2(np.fft.ifftshift(Gf)) # IFT & shift to return to real space
    return gf

#%% function definitions

cell_space = 1
targ_dist = 0.5
res_fac = 1

k = 2* np.pi/340 #wave number where all the waves are the same wavelength lambda = 340m 20Hz infrasound

def amplitude(plane): #using example amplitude-extracting function
    #this function takes and returns an array
    for R in range(len(plane)):
        for Ran in range(len(plane[R])):
                plane[R][Ran] = cmath.sqrt(plane[R][Ran].real**2 + plane[R][Ran].imag**2)
    return plane

def phase(plane): #using example phase-extracting function
     #this function takes and returns an array
    for R in range(len(plane)):
        for Ran in range(len(plane[R])):
            plane[R][Ran] = np.angle(plane[R][Ran])
    return plane

def produceBD(ampsrc, phasa):
    B = np.copy(ampsrc)
    for R in range(len(ampsrc)):
        for Ran in range(len(ampsrc[R])):
            B[R][Ran] = ampsrc[R][Ran] * cmath.exp((1j)) * phasa[R][Ran]
    return B

def ger_sax(Source, Target, Retrieved_Phase, iterations):
    A = ASM_bw(Target, cell_space, targ_dist,res_fac,k) #replace with angular spectrum method

    for R in range(0,iterations): #Temporary error criterion
        # may have to implement this differently as it's for arrays
        B = produceBD(amplitude(Source),phase(A))
        C = ASM_fw(B, cell_space, targ_dist, res_fac, k)
        D = produceBD(amplitude(Target),phase(C))
        A = ASM_bw(D, cell_space, targ_dist,res_fac,k)
        Retrieved_Phase = phase(A)

        # plt.imshow(abs(Retrieved_Phase))
        # plt.show()

        E = ASM_fw(A, cell_space, targ_dist,res_fac, k)

        # plt.imshow(abs(amplitude(E)))
        # plt.show()

    return (Retrieved_Phase, amplitude(E))

def gen_random_mask(size): #size is a 2d array
    randarr = np.random.rand(size[0],size[1])
    return randarr

#%% main and run

def main(img,size,iterations):
    image = plt.imread(img)
    plt.imshow(image)
    plt.show() #this is the target
    image = image[:,:,:1]/256 #image between 0 and 1
    ran_val = np.random.rand()
    image_complex = abs(image) * np.exp((1j)*ran_val*np.pi)
    image = image_complex.reshape((size,size))
    #we need an initial retrieved phase and source
    source = gen_random_mask(image.shape)*np.exp((1j)* np.pi)
    ret_phase = np.full(image.shape,(1j)*0*np.pi,np.complex)
    #show them as images then run the algorithm on them (commented out for legibility)
    # img_again = img.fromarray(data)
    # plt.imshow(img_again)
    # plt.show()
    # plt.imshow(abs(image))
    # plt.show()
    #
    # plt.imshow(abs(source))
    # plt.show()

    # blank = img.fromarray(ret_phase)
    # plt.imshow(abs(ret_phase))
    # plt.show()

    #run GS!!
    result = ger_sax(source,image,ret_phase,iterations)
    # repeat result image show!!
    plt.imshow(abs(result[0]))
    plt.show()
    plt.imshow(abs(result[1]))
    plt.show()

    return result[0] #return the phase map, and not the amplitude map

# main('GSTestData/UCL16.png',16,50)
# main('GSTestData/U16.png',16,50)
# main('GSTestData/C16.png',16,50)
# main('GSTestData/L16.png',16,50)
# main('GSTestData/UCLport16.png',16,50)
# main('GSTestData/portico16.png', 16,100)