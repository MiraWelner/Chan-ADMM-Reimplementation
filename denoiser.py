import cv2
import scipy.ndimage as ndimage
import numpy as np
import skimage
from sklearn import preprocessing as pre


def proj(x,bound=[0,1]):
    """
    Projects an image between 0 and 1
    """
    npx = np.array(x)
    npx[npx > 1] = 1
    npx[npx < 0] = 0
    return npx.tolist()

def PlugPlayADMM_deblur(y,h, gamma, rho, tol, method = 'bm3d'):
    max_itr   = 30
    dim         = y.shape        
    N           = dim[0]*dim[1]    
    Hty         =  cv2.filter2D(y, -1, h)
    eigHtH = np.abs(np.fft.fft2(h,dim))**2
    v = 0.5*np.ones(dim)
    x = v;
    u = np.zeros(dim);
    residual = np.inf;

    itr = 1;
    while(residual>tol and itr<=max_itr):
        x_old = x;
        v_old = v;
        u_old = u;
        
        x = np.real(np.fft.ifft2((np.fft.fft2(Hty + rho*(v - u))) / (rho+eigHtH)))

        vtilde = x+u;
        vtilde = proj(vtilde);
        if method == 'Non-Local Means':
            v = skimage.restoration.denoise_nl_means(np.array(vtilde),h=np.sqrt(0.005/rho))
        elif method == 'Total Variation':
            sigma = np.sqrt(0.0001/rho)
            v = skimage.restoration.denoise_tv_chambolle(np.array(vtilde),weight=1/(sigma**2))
        elif method == 'BM3D':
            v = bm3d.bm3d(vtilde,np.sqrt(0.001/rho))
        elif method == 'Median':
            v = ndimage.median_filter(vtilde, size=3)

            
        u = u + (x-v);
        rho = rho*gamma;

        residualx = (1/np.sqrt(N))*(np.sqrt(sum(sum((x-x_old)**2))))
        residualv = (1/np.sqrt(N))*(np.sqrt(sum(sum((v-v_old)**2))))
        residualu = (1/np.sqrt(N))*(np.sqrt(sum(sum((u-u_old)**2))))

        residual = residualx + residualv + residualu;
                
        itr = itr+1;
    return v, itr

def Plug_And_Play_ADMM_1D(y, gamma, rho, tol, method = 'Total Variation'):
    residuals = []
    max_itr   = 20
    dim         = y.shape        
    N           = dim[0]   
    v = np.ones(dim)
    x = v;
    u = np.zeros(dim);
    residual = np.inf;

    itr = 1;
    while(residual>tol and itr <= max_itr):
        x_old = x;
        v_old = v;
        u_old = u;
        
        x = np.real(np.fft.ifft((np.fft.fft(y + rho*(v - u))) / (rho)))
        
        vtilde = x+u;
        if method == 'Non-Local Means':
            v = skimage.restoration.denoise_nl_means(vtilde, h=(rho/95)).reshape(-1,1)
        elif method == 'Total Variation':
            v = skimage.restoration.denoise_tv_chambolle(np.array(vtilde),weight=1/(50*rho))
        elif method == 'Median':
            v = ndimage.median_filter(vtilde, mode='nearest', size=5)
        
        u = u + (x-v);

        rho = rho*gamma;

        residualx = 1/np.sqrt(N)*(np.sqrt(sum((x-x_old)**2)))
        residualv = 1/np.sqrt(N)*(np.sqrt(sum((v-v_old)**2)))
        residualu = 1/np.sqrt(N)*(np.sqrt(sum((u-u_old)**2)))

        residual = residualx + residualv + residualu;
        residuals.append([residualx, residualv, residualu])
        itr = itr+1;
        if method == 'Total Variation':
            v = v.reshape(-1,1)
    return pre.MinMaxScaler().fit_transform(v),itr