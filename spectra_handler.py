import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns
import pandas as pd
import scipy.io
from scipy.interpolate import interp1d, interp2d
import scipy.linalg as la
from scipy.optimize import curve_fit
import random
from sklearn.decomposition import NMF
from sklearn import preprocessing as pre
from scipy import ndimage

class Max_Fit:
    """
    A class used to fit the position and width of the strongest absorptive feature in a spectrum.
     
    ...

     Attributes
    ----------
    R : Reconstruction
        A Reconstruction object based on results from an NMF Reconstruction 

       
    Methods
    -------

    _fit_lorentzian()
        Private method that fits a single Lorentzian to a data set

    plot_fits()
        plots and displays fit parameters for reconstructed and ground truth spectra
        
    get_fit_data()
        Method utilized to extract fit results for usage in other scripts.      
    """    
    
    R = None   
    spectra_int = 0  
    del_peak = 10
    del_fwhm = 25
    peak_guess_g = 10
    lower_bound_0 = 0
    lower_bound_2 = -3
    upper_bound_0 = 500
    upper_bound_2 = 3
           
    def __init__(self,R, del_peak = 10,
                 del_fwhm = 25,
                 peak_guess_g = 10,
                 lower_bound_0 = 0,
                 lower_bound_2 = -3,
                 upper_bound_0 = 500,
                 upper_bound_2 = 3):
        
        #set values
        self.del_peak = del_peak
        self.del_fwhm = del_fwhm
        self.peak_guess_g = peak_guess_g
        self.lower_bound_0 = lower_bound_0
        self.lower_bound_2 = lower_bound_2
        self.upper_bound_0 = upper_bound_0
        self.upper_bound_2 = upper_bound_2
        # Get Spectra and Maximum Locations
        # Reconstructed Spectra
        self.r_spec = R.get_reconstructed()
        r_max_loc = np.argmax(self.r_spec)

        # Ground Truth
        self.g_spec = R.get_ground_truth()
        g_max_loc = np.argmax(self.g_spec)

        # Make an array of x_values
        self.x_use = np.arange(0,np.shape(self.r_spec)[0]).reshape(-1,1)

         # Define the Lorentzian
        fun_lorentzian = lambda x, a, x_o, g, z: a / (np.pi * ((x - x_o)**2 + g**2)) + z
        # Vectorize
        self.vfun_lorentzian = np.vectorize(fun_lorentzian)

        # Fit the Data
        # Ground Truth
        self.g_popt = self._fit_lorentzian(self.g_spec,g_max_loc)

        # Reconstructed
        self.r_popt = self._fit_lorentzian(self.r_spec,r_max_loc)  

        # Squared Error Peak Position
        #  Peak Position of Ground Truth
        p_g = self.g_popt[1]
        # Peak Position of Reconstructed Spectra
        p_r = self.r_popt[1]
        self.peak_serr = (p_g - p_r)**2

        # Squared Error HWHM
        #  HWHM of Ground Truth
        G_g = self.g_popt[2]
        # Peak Position of Reconstructed Spectra
        G_r = self.r_popt[2]
        self.HWHM_serr = (G_g - G_r)**2
    
    def _fit_lorentzian(self,spec_fit,peak_guess=0):
        """
        Private function that fits a spectrum to the Lorentzian function
         
         Parameters
        ----------
        spec_fit : array
                1D array of spectral intensities

        peak_guess : float
                Guess of peak location
         Returns
        -------
           
        popt : array
                Fitted parameters to be used in plot_fits
        """
        # Fitting the Function
        vfun_lorentzian = self.vfun_lorentzian
        
        # Fit only Region of reg_width pixels from peak
        reg_width = 50
        reg_fit = np.array([peak_guess - reg_width, peak_guess + reg_width])

        # Do not Let Fit Go Below reg_width
        if np.min(reg_fit)<0: 
            reg_fit[0]=0
        elif np.max(reg_fit)>np.max(self.x_use):
            reg_fit[-1]=np.max(self.x_use)

        # Parse data to Masked Region
        spec_use = spec_fit[np.min(reg_fit):np.max(reg_fit)].squeeze()
        x_fit = self.x_use[np.min(reg_fit):np.max(reg_fit)].squeeze()
        
        # Intial Guess (a,x_o,g,z)
        p_o = [1, peak_guess, self.peak_guess_g, 0]

        # Bounds 
        del_peak = self.del_peak
        del_fwhm = self.del_fwhm
        lower_bounds = [self.lower_bound_0, np.min(peak_guess) - del_peak, 0, self.lower_bound_2]
        upper_bounds = [self.upper_bound_0, np.min(peak_guess) + del_peak, del_fwhm,  self.upper_bound_2]

        # Perform the Fit
        popt = curve_fit(vfun_lorentzian, x_fit, spec_use, p0=p_o, bounds=(lower_bounds, upper_bounds))[0]

        return popt
        
    def plot_fits(self):
        """
        Plots the fitted curve alongside data for reconstructed and ground truth data
        """

        # Fitting the Function
        vfun_lorentzian = self.vfun_lorentzian

        # Get Ground Truth Fitted Spectra
        g_popt = self.g_popt

        # Get Reconstructed Fitted Spectra
        r_popt = self.r_popt

        # Get Error Terms
        p_error = (self.peak_serr)**0.5
        G_error = (self.HWHM_serr)**0.5

        # Plot Ground Truth Fit, Reconstructed Fit, and Actual Data
        plt.plot(self.x_use,vfun_lorentzian(self.x_use,r_popt[0],r_popt[1],r_popt[2],r_popt[3]))
        plt.plot(self.x_use,vfun_lorentzian(self.x_use,g_popt[0],g_popt[1],g_popt[2],g_popt[3]))
        plt.plot(self.x_use,self.g_spec,'o',color='black')
        # Label And Annotate Plot
        plt.xlabel('Pixel/Energy')
        plt.ylabel('Normalized Signal')
        plt.legend(['Recon. Fit','Ground Fit','Ground Data'],loc='best')
        err_st = r'$\Delta \omega$ Peak= '+format(p_error,'.2f')+ r', $\Delta \Gamma_{1/2}$ = '+format(G_error,'.2f')
        plt.title(err_st)
        
    def get_error(self):
        """
        Plots the fitted curve alongside data for reconstructed and ground truth data
        """

        # Fitting the Function
        vfun_lorentzian = self.vfun_lorentzian

        # Get Ground Truth Fitted Spectra
        g_popt = self.g_popt

        # Get Reconstructed Fitted Spectra
        r_popt = self.r_popt

        # Get Error Terms
        p_error = (self.peak_serr)**0.5
        G_error = (self.HWHM_serr)**0.5

        return(p_error, G_error)

    def get_fit_data(self):
        """
            Method utilized to extract fit results for usage in other scripts.    
        -------   
            Returns
        -------
            fit_out : array
                Fitted parameters and their absolute error relative to ground truth
        """  
        
        # Fit Resultsof Reconstructed Spectra
        # Peak is 1 and HWHM is 2        
        fit_r = self.r_popt[1:3]
        
        # Get Errors
        peak_serr = (self.peak_serr)**0.5
        HWHM_serr = (self.HWHM_serr)**0.5

        # Put Together Output Array
        fit_out = np.array([fit_r[0],peak_serr,fit_r[1],HWHM_serr])

        return fit_out

        
class Reconstruction:
    """
    A class used to reconstruct a spectra using the non-negative matrix factorization of a group of many
    (presumably) similar spectra
    ...

    Attributes
    ----------
    X : Multi_Spectra
        A Multi_Spectra object consisting of n spectra which the NNFM is based off
    dimensions : int
        The dimensions included in the NNMF
    spectra_int : int
        which spectra within X is being tested
    nmf : scipy.NMF
        the NFM object - it's a scipy object so you can only call scipy.NMF functions on it
    scores : 2d list
        the scaleing of the x axis provided in the file. This is corrected in the program so that the scale is 1
    ground_truth : numpy array
        the scaleing of the y axis provided in the file. This is corrected in the program so that the scale is 1
    reconstructed: numpy array
        the lowest energy at which the spectra is recoreded - lower will require zero padding

    Methods
    -------
    __make_ground_truth(self)
        A private method that creates the ground_truth attribute
        
    __make_recon(self)
        A private method that makes the reconstructed attribute
        
    plot_ground_truth()
        plots the ground truth attribute created by __make_ground_truth
        
    plot_reconstructed
        plots the reconstructed spectra created by __make_recon
    
    plot_overlay
        plots both the reconstructed spectra and the ground truth attribute and displays the mean square error
        
    get_error -> int
        returns the mean squared error that is found in plot_overlay but doesn't make a plot
    
    get_reconstructed -> array
        returns a 1D array of the reconstructed spectra
    
    get_ground_truth -> array
        returns a 1D array of the ground truth spectra
    """
    
    X = None
    dimensions = 0
    spectra_int = 0
    nmf = None
    scores = None
    ground_truth = None
    reconstructed = None
    W = None
    H = None
    
    def __init__(self,X,spectra_int,dimensions=2, solver='mu', initializer = 'nndsvda', beta_loss=2):
        self.X = X
        self.dimensions = dimensions
        self.spectra_int = spectra_int
        # Set Up the DataBase for NMF Analysis
        self.nmf = NMF(n_components=dimensions, init = initializer,
                       max_iter = 1200,
                       tol=1e-4,
                       solver = solver,
                       beta_loss=beta_loss)
        # Fit the Database to NMF Model
        self.nmf.fit(X.T)
        # Get the Weights and the "Hidden" Variables
        self.W = self.nmf.transform(X.T)
        self.H=self.nmf.components_
        self.__make_ground_truth()
        self.__make_recon()

         
    def __make_ground_truth(self):
        """
        takes the spectra that is supposed to be reconstructed and scales it between 0 and 
        1 and then stores it as an attribute
        """
        ground_truth = np.array(self.X.iloc[:,self.spectra_int]).reshape(-1, 1)
        ground_truth = pre.MinMaxScaler().fit_transform(ground_truth)
        self.ground_truth = ground_truth
        
    def __make_recon(self):
        """
        When you project the unscaled ground truth onto the NMF, each dimention in the NMF gets a score.
        The NFM objects also has components, which are rows of the factorization matrix. The ith row of the
        factorization matrix represents a spectra that would score a 1 on the ith dimention, and a 0 on other
        dimensions.
        
        Thus for each value between 0 and the number of NMF dimensions, we take the the product of the ith 
        score and the ith component. We then take the sum of these i values and that is the reconstructed
        spectra.
        
        This value is stored in the reonstructed_spectra attribute

        TB 25May23: Rather than performing this elementwise, I have changed this to a matrix multiplication.
        """
        
        # Grab the Variables
        W = self.W
        H = self.H

        # Make Reconstruction 
        reon_unscale = np.matmul(W[self.spectra_int,:],H).reshape(-1,1)
        self.reonstructed_spectra = pre.MinMaxScaler().fit_transform(reon_unscale)     
        
    def plot_ground_truth(self):
        """
        Plots the ground truth in it's own plot
        """
        plt.figure()
        plt.plot(self.ground_truth)
        plt.title("Ground Truth")
        plt.xlabel("nth wavelength")
        plt.ylabel("absorbance")
        
    def plot_reconstructed(self):
        """
        Plots the reconstructed spectra in it's own plot
        """
        plt.figure()
        plt.plot(self.reonstructed_spectra)
        plt.title("Reconstructed Spectra with " + str(self.dimensions + 1) + " NMF Dimentions")
        plt.xlabel("nth wavelength")
        plt.ylabel("absorbance")
    
    def plot_overlay(self):
        """
        Plots the reconstructed spectra and the ground truth spectra in the same plot, displays
        the mean squared error
        """
        plt.plot(self.reonstructed_spectra, label="Reconstructed")
        plt.plot(self.ground_truth, label="Ground Truth")
        plt.xlabel("nth wavelength")
        plt.ylabel("absorbance")
        plt.legend()
        mse = ((self.reonstructed_spectra - self.ground_truth) ** 2).mean(axis=None)
        plt.title("MSE = " + str(mse))
        plt.show()
    
    def get_error(self):
        """
        Returns the mean squared error between the reconstructed spectra and the ground truth spectra
        Both are scaled between 1 and 0 to ensure accuracy
        """
        return ((self.reonstructed_spectra - self.ground_truth) ** 2).mean(axis=None)
    
    def get_reconstructed(self):
        """
        Returns the reconstructed spectra
    
        """
        return self.reonstructed_spectra
    
    def get_ground_truth(self):
        """
        Returns the ground truth spectra
    
        """
        return self.ground_truth

class Spectrum:
    """
    A class used to represent a singular spectrum, and it's absorbance at different wavelengths
    
    To create an instance of this class requires a file describing the spectra. It is currently designed to
    accept data from NIST files but could be changed to accept different file formats with minimal tweaking

    ...

    Attributes
    ----------
    chemical_name : Multi_Spectra
        The name of the compound that the spectra is of - for example, 4-Nitrophenol
    molecular_composition : str
        The molecular composition of the compound that the spectra is of - for example, C6H5NO3 
    x_units : str
        the units that the x axis is in. This is typically 1/CM meaning the number of wavelenths per CM
    y_units : str
        the units that the y axis is in. This is typically absorbance
    x_factor : float
        the scaleing of the x axis provided in the file. This is corrected in the program so that the scale is 1
    y_factor : float
        the scaleing of the y axis provided in the file. This is corrected in the program so that the scale is 1
    min_energy: int
        the lowest energy at which the spectra is recoreded - lower will require zero padding
    max_energy: int
        the highest energy at which the spectra is recoreded - higher will require zero padding
    delta: int
        the interval between places that the spectra is sensed at
    raw_energy_data: list
        a list of the absorbance values taken directly from the list
    x values: list
        a list of the intervals the absorbance is analysed at constructed from the min, max, and delta

    Methods
    -------
    process_raw_energy_data(self)
        turns raw energy data from the file into the x value data - verifies internal consitancy
        
    create_dataframe(self, min_energy=None, max_energy=None, delta=None) -> pandas dataframe
        creates a pandas dataframe of the absorbancies given wavelength. Designed to be plotted in seaborn
        
    plot_single_spectrum(self, min_energy=None, max_energy=None, delta=None, scatter = False)
        uses the above create_dataframe function to make a dataframe and plot it via seaborn
    
    """
    
    chemical_name = ''
    molecular_composition = ''
    y_units = ''
    x_factor = 1
    y_factor = 1
    min_energy = 0
    max_energy = 0
    delta = 0
    raw_energy_data = 0
    y_values = np.array([])
    x_values = np.array([])
    
    def __init__(self, raw_text=None, path=None):
        """
        Parameters
        ----------
        raw_text : str
            The raw text of the file that will be parsed into internal values
        """
        if raw_text == None:
            raw_text = open(path, "r").read()
        self.chemical_name = re.findall("##TITLE=.*", raw_text)[0][8:]
        self.molecular_composition = "("+re.findall("##MOLFORM=.*", raw_text)[0][10:].replace(" ", "")+")"
        self.y_units = re.findall("##YUNITS=.*", raw_text)[0][9:].capitalize()
        self.x_units = re.findall("##XUNITS=.*", raw_text)[0][9:].capitalize()
        self.x_factor = float(re.findall("##XFACTOR=.*", raw_text)[0][10:])
        self.y_factor = float(re.findall("##YFACTOR=.*", raw_text)[0][10:])
        self.min_energy = int(re.findall("##MINX=.*", raw_text)[0][7:])
        self.max_energy = int(re.findall("##MAXX=.*", raw_text)[0][7:])
        self.delta = int(float(re.findall("##DELTAX=.*", raw_text)[0][9:]))
        
        self.raw_energy_data = raw_text.split(re.findall("##.*", raw_text)[-2])[1].split("##")[0]
        if "##TWOCOL" in raw_text:
            self.process_two_col_energy()
        else:
            self.x_values = np.array(range(self.min_energy, self.max_energy+self.delta)[::self.delta])
            self.process_raw_energy_data()
            
        self.x_values = self.x_values*self.x_factor
        self.y_values = self.y_values*self.y_factor
        
        if self.y_units == "Transmittance":
            self.y_units = "Absorbance"
            self.y_values = np.array([1-i for i in self.y_values])
            
        for itr, i in enumerate(self.y_values):
            if i < 0 or np.isnan(i):
                self.y_values[itr] = 0        
    
    def process_two_col_energy(self):
        self.delta = 1
        temp_x_values = []
        temp_y_values = []

        for line in self.raw_energy_data.split("\n")[1:-1]:
            temp_x_values.append(float(line.split(", ")[0]))
            temp_y_values.append(float(line.split(", ")[1]))
        f = interp1d(temp_x_values, temp_y_values)
        self.x_values = np.array(range(self.min_energy, self.max_energy))
        self.y_values = np.array([])
        for x in self.x_values:
            self.y_values = np.append(self.y_values, f(x))
        
    def process_raw_energy_data(self):
        """
        Takes in the raw numbers from raw_energy_data and transforms them into x_values. In the NIST files,
        the x values are formatted in lines where first there is a y value, corrsponding to the first x value
        in the line, and then several more y values that are the delta value in distance from eachother. This 
        function processes the data, removes the y values, and ensures that the given x values are what they should
        be given the delta. In all cases in the NIST database, there is no error regarding the x values
        and the delta
        """
        correct_x_value = self.min_energy
        for line in self.raw_energy_data.split("\n")[1:-1]:
            #verify that the x value is correct
            if(int(float(line.split(" ")[0])) != correct_x_value):
                print("incorrect x value")
                break
            else:
                #update correct x value and add y values to list
                for value in line.split(" ")[1:]:
                    self.y_values = np.append(self.y_values, int(value))
                    correct_x_value += self.delta
        
    def create_dataframe(self, min_energy=None, max_energy=None, delta=None):
        """
        seaborn only plots pandas dataframes so this generates a pandas dataframe. There are three rows: x values,
        y values, and the chemical name. The final row is simply a row that says the chemical name in each column.
        This is useful for when you have a dataframe with a lot of spectra which happens in the multi_spectra class.
        The dataframe can optionally be altered by min_energy and max_energy inputs which will either clip or zero
        pad the ends. Also the delta can be changed.

        Parameters
        ----------
        min_energy : int, optional
            an alternate minimum energy (default is None)
        max_energy : int, optional
            an alternate maximum energy (default is None)
        delta : int, optional
            an alternate gap between the neighboring frequencies

        Returns
        -------
        dataframe
            A 3 row pandas dataframe containing the x values, y values, and chemical name
        """
        if min_energy == None:
            min_energy = self.min_energy
        if max_energy == None:
            max_energy = self.max_energy
        if delta == None:
            delta = self.delta
            
        y_to_display = self.y_values
        x_to_display = self.x_values
        if min_energy < self.min_energy:
            #if the user wants the plot minimum to be LOWER than the lowest recorded energy, zero pad
            y_to_display = np.append(np.zeros(int(np.ceil((self.min_energy-min_energy)/self.delta))), self.y_values)
            x_values_to_add = range(min_energy, self.min_energy)[::self.delta]
            x_to_display = np.append(x_values_to_add, x_to_display)
            
        if min_energy > self.min_energy:
            #if the user wants the plot minimum to be HIGHER than the lowest recoreded energy, clip it
            y_to_display = y_to_display[(min_energy-self.min_energy)//self.delta:]
            x_to_display = x_to_display[(min_energy-self.min_energy)//self.delta:]
            
        if max_energy > self.max_energy:
            #if the user wants the plot maxiumum to be HIGHER than the highest recoreded energy, zero pad
            y_to_display = np.append(y_to_display, np.zeros(int(np.ceil((max_energy-self.max_energy)/self.delta))))
            x_values_to_add = range(self.max_energy, max_energy)[::self.delta]
            x_to_display = np.append(x_to_display, x_values_to_add)
            
        if max_energy < self.max_energy:
            #if the user wants the plot maximum to be LOWER than the highest recoreded energy, clip it
            y_to_display = y_to_display[:-1*(self.max_energy-max_energy)//self.delta]
            x_to_display = x_to_display[:-1*(self.max_energy-max_energy)//self.delta]
            
        if delta != self.delta:
            #if the user changes the delta, create a continuous function via interpolation
            #then find values at the given delta intervals
            continuous_function = interp1d(x_to_display, y_to_display)
            
            x_to_display = np.array(range(min_energy, max_energy-1)[::delta]) #-1 because NIST is awful
            new_y = np.array([])
            for x in x_to_display:
                new_y = np.append(new_y, continuous_function(x))
            y_to_display = new_y
        
        dictionary = {self.x_units: x_to_display,
                      self.y_units: y_to_display,
                      'Chemical': [self.chemical_name for x in range(len(x_to_display))]}
        dataframe = pd.DataFrame(dictionary)
        return dataframe
    
    def plot(self, min_energy=None, max_energy=None, delta=None, scatter = False, markers=[]):
        """
        Plots the spectra using seaborn by calling the create_dataframe function with the given input data
        It can make a scatter plot or lineplot. Lineplots are typically better for spectra but scatter plots 
        illustrate the delta
        
         Parameters
         ----------
         min_energy : int, optional
            an alternate minimum energy (default is None)
         max_energy : int, optional
            an alternate maximum energy (default is None)
         delta : int, optional
            an alternate gap between the neighboring frequencies
        """
        dataframe = self.create_dataframe(min_energy, max_energy, delta)
        sns.set(style='darkgrid')
        plt.rcParams['figure.figsize']=(10,7)
        if scatter:
            sns.scatterplot(x=self.x_units, y=self.y_units, data=dataframe).set(
            title=self.chemical_name + " " + self.molecular_composition)
        else:
            sns.lineplot(x=self.x_units, y=self.y_units, data=dataframe).set(
            title=self.chemical_name + " " + self.molecular_composition)
        for m in markers:
            plt.axvline(x = self.x_values[m], color = 'r')
        plt.show()

class Multi_Spectra:
    """
    A class used to take multiple spectrum objects generated by the above class and apply the same modifications
    to them before plotting them on the same axis, or analysing them as a whole
    
    ...

    Attributes
    ----------
    n : int
        the number of spectra to be taken from the directory and analysed. 
    min_min : int
        the minimum of all the minimum energies - all spectra with higher minimums are zero padded to this min
    max_max : int
        the maximum of all the maximum energies - all spectra with lower maximum are zero padded to this max
    min_delta : int
        the minimum delta - all spectra with larger deltas are interpolated to this delta
    matrix : 2D array
        a matrix where each column is a spectra and each row is the data at one energy level
    kind : string
        a string representing whether it is a scatter plot or a line plot
    legend: string
        sometimes there are too many spectra for a legend to be a good idea - this controls that

    Methods
    -------
    error_with_sv_removed(self, n) -> float
        takes all spectra in the multi-spectra dataframe and performs svd on it. Then it removes the n least
        significant singular values and reconstructs the multi-spectra dataframe. It returns a float representing
        the mean squared error between the reconstructed group of spectra
        
    plot(self) -> pandas dataframe
        creates a seaborn plot of everything
    
    """
    n = 0
    min_min = float('inf')
    max_max = 0
    min_delta = float('inf')
    spectra = []
    kind = False
    legend = 'full'
    matrix = pd.DataFrame()
    
    
    def __init__(self, directory = None,
                 spectra_list = None,
                 n = None,
                 min_energy=None,
                 max_energy=None,
                 delta=None, 
                 scatter = False,
                 legend = None):
        if scatter:
            self.kind='scatter'
        else:
            self.kind='line'
        if directory == None and spectra_list != None:
            if n != None:
                self.n = n
            else:
                self.n = len(spectra_list)
            for spectrum in spectra_list[1:n]:
                self.spectra.append(spectrum)
        elif directory != None and spectra_list == None:
            if n != None:
                self.n = n
            else:
                self.n = len(os.listdir(directory))
            
            self.spectra = [] #ensure that you don't have stuff from a previous object
            for filename in os.listdir(directory)[0:self.n]:
                if ".DX" in filename or ".txt" in filename:
                    spectra_raw_text = open(os.path.join(directory, filename), "r").read()
                    self.spectra.append(Spectrum(spectra_raw_text))            
                  
        min_list = []
        max_list = []
        delta_list = []
        
        
        for s in self.spectra:
            if min_energy == None:
                min_list.append(s.min_energy)
            if max_energy == None:
                max_list.append(s.max_energy)
            if delta == None:
                delta_list.append(s.delta)

        if min_energy == None:
            self.min_min = min(min_list)
        else:
            self.min_min = min_energy
            
        if max_energy == None:
            self.max_max = max(max_list)
        else:
            self.max_max = max_energy
        
        if delta == None: 
            self.min_delta = min(delta_list)
        else:
            self.min_delta = delta
            
        random.shuffle(self.spectra)
        dfs = [spec.create_dataframe(min_energy=self.min_min,
                                     max_energy=self.max_max,
                                     delta=self.min_delta) for spec in self.spectra]
        dataframe = pd.concat(dfs,axis=0)
        
        if (legend == None and self.n>30) or legend == False:
            self.legend = False
          
        #generate matrix
        rows_in_long_df = list(dataframe.loc[:,"1/cm"][1:]).index(float(list(dataframe.loc[:,"1/cm"])[0])) + 1
        split_df = [x for _, x in dataframe.groupby(dataframe['Chemical'])]
        head = []
        body = []
        for df_segment in split_df:
            head.append(df_segment['Chemical'].iloc[0])
            body.append(list(df_segment.loc[:,"Absorbance"][0:rows_in_long_df]))
        for column in body:
            df_col = pd.DataFrame(column)
            self.matrix=pd.concat([self.matrix, df_col], axis=1)
        self.matrix.columns = head
        self.matrix.index = dataframe['1/cm'].values[0:int(len(dataframe['1/cm'].values)//self.n)]
        
    def error_with_sv_removed(self, n):
        """
        takes all spectra in the multi-spectra dataframe and performs svd on it.  the svd creates u, sigma, and v. 
        Sigma is a diagonal matrix where the diagonal is the singular values in order from most significant to
        least significant. This function removes the n least significant singular values and reconstructs the 
        multi-spectra dataframe. It returns a float representing the mean squared error between the reconstructed 
        group of spectra and the original one
        
        svd_df is dataframe where each column corresponds to a single spectrum and the chemical names are removed,
        it is the format in which svd is taken. You may notice that the order of the columns (ie spectra) do not
        correspond to what you observe when you print the multi_spectra object's dataframe attribute. This is
        because the spectra in svd_df are sorted alphabetically, which is what happens when you do df.groupby

        Parameters
        ----------
        n : int
            the svd creates u, sigma, and v. the n least significant sigma values are removed

        Returns
        -------
        mean_squared_error: float
            the mse between the original set of spectra and the reconstructed set of spectra
        """
        u, s, v = np.linalg.svd(self.matrix, full_matrices=True)
        for i in reversed(range(n)):
            s[i] = 0
        reconstructed = u@la.diagsvd(s,*svd_df.shape)@v
        original = self.matrix.values.tolist()
        return mean_squared_error(original, reconstructed)
    
    def plot_error_with_svd_removed(self, yscale = 'linear', xscale = 'linear', markers = []):
        """
        calculate the error_with_sv_removed using the function above for every n value from 0 up until the
        max number of singular values. So the first mse should be zero and the last should be almost 1 because
        in the last one, the entire sigma matrix is made of zeros. This then plots it with linear x scale and 
        logarithmic y scale.
        
        Parameters
        ----------
        yscale : string
            if you set this to 'log' then the y axis is plotted logarithmically
        xscale : string
            if you set this to 'log' then the x axis is plotted logarithmically
        markers: int list
            the default is empty which means no markers, but every int in this list is the location of a vertical
            line in the plot. This allows you to easily emphasize specific points on the graph
        """
        spectras_error = []    
        for i in range(0,self.n):
            spectras_error.append(self.error_with_sv_removed(i))
        plt.yscale(yscale)
        plt.xscale(xscale)
        plt.rcParams["figure.figsize"] = (20,15)
        for m in markers:
            plt.axvline(x = m, color = 'r')
        plt.plot(spectras_error)
        
    def spectra_with_sv_removed(self, n):
        """
        takes spectra, performs svd, removes n least significant singular values, then ta

        Parameters
        ----------
        n : int
            the svd creates u, sigma, and v. the n least significant sigma values are removed

        Returns
        -------
        mean_squared_error: float
            the mse between the original set of spectra and the reconstructed set of spectra
        """
        df = self.dataframe
        rows_in_long_df = list(df.loc[:,"1/cm"][1:]).index(float(list(df.loc[:,"1/cm"])[0])) + 1
        split_df = [x for _, x in df.groupby(df['Chemical'])]
        for df_segment in split_df:
            self.matrix[df_segment['Chemical'].iloc[0]] = list(df_segment.loc[:,"Absorbance"][0:rows_in_long_df])
        u, s, v = np.linalg.svd(svd_df, full_matrices=True)
        for i in reversed(range(n)):
            s[i] = 0
        reconstructed = pd.DataFrame(u@la.diagsvd(s,*svd_df.shape)@v)
        
        return reconstructed
    
    def plot(self):
        """
        plot all spectra in the dataframe
        """
        sns.relplot(x='1/cm',y='Absorbance', hue='Chemical', kind = self.kind, data=self.dataframe,height=7,
                    aspect=2,
                    legend = self.legend
                   ).set(title="Multi-Spectral Plot")
        plt.show()