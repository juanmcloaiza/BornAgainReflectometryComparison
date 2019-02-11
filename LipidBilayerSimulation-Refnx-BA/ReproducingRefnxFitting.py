import numpy as np
import bornagain as ba
from bornagain import deg, angstrom
from matplotlib import pyplot as plt
from datetime import datetime as dtime
WAVELENGTH = 1.0 # Dummy for this exercise.
DATA_TO_FIT="./FitByRefnx.txt"


class SampleParameters():
    """
    Class whose only scope is to store and return values for the different
    parameters of the simulation. The defaults are set to the ones that
    "best fit" (i.e. found by a human) the curve in the fit produced by Refnx.
    The Refnx fit curve was generated using a Jupyter notebook downloaded from
    http://scripts.iucr.org/cgi-bin/paper?rg5158
    """

    # _a_ is "inner" in refnx; _f_ is "outer".
    _solvent_sld =  6.18989e-06
    _head_f_sld  =  1.88401254e-06
    _tail_f_sld  = -0.37340153e-06
    _head_a_sld  =  1.88401254e-06
    _tail_a_sld  = -0.37340153e-06
    _sio2_sld    =  3.47e-06
    _silicon_sld =  2.07e-06

    _head_f_thickness =  5.64468
    _head_a_thickness =  9.86971
    _tail_f_thickness = 13.7233
    _tail_a_thickness = 13.7233
    _sio2_thickness   = 12.0885

    # Solvent volume fractions.
    _head_f_sfv = 0.9915395153082305
    _tail_f_sfv = 0.9997855735077633
    _head_a_sfv = 0.5670808231721158
    _tail_a_sfv = 0.9997855735077633
    _sio2_sfv = 1.0 - 0.127675

    # These are obtained from "theory" according to refnx paper;
    # They Correspond to the slds AFTER solvation.
    # Reported here only as an aside note, they are calculated later.
    # head_f_overall_sld =  1.920442349526814e-06
    # tail_f_overall_sld = -0.3719941909455789e-06
    # head_a_overall_sld =  3.7481094650427435e-06
    # tail_a_overall_sld = -0.3719941909455789e-06
    # sio2_overall_sld   =  3.81726195575e-06

    _rough1 = 4.80879
    _rough2 = 3.00
    _rough3 = 3.00
    _rough4 = 0.00
    _rough5 = 3.00
    _rough6 = 1.64969
    _rough7 = 0.00


    def __init__(self,params={}):
        self.solvent_sld = params["solvent_sld"] if "solvent_sld" in params else self._solvent_sld
        self.head_f_sld  = params["head_f_sld"]  if "head_f_sld"  in params else self._head_f_sld
        self.tail_f_sld  = params["tail_f_sld"]  if "tail_f_sld"  in params else self._tail_f_sld
        self.tail_a_sld  = params["tail_a_sld"]  if "tail_a_sld"  in params else self._tail_a_sld
        self.head_a_sld  = params["head_a_sld"]  if "head_a_sld"  in params else self._head_a_sld
        self.sio2_sld    = params["sio2_sld"]    if "sio2_sld"    in params else self._sio2_sld
        self.silicon_sld = params["silicon_sld"] if "silicon_sld" in params else self._silicon_sld

        self.sio2_thickness   = params["sio2_thickness"]   if "sio2_thickness"   in params else self._sio2_thickness
        self.head_a_thickness = params["head_a_thickness"] if "head_a_thickness" in params else self._head_a_thickness
        self.tail_a_thickness = params["tail_a_thickness"] if "tail_a_thickness" in params else self._tail_a_thickness
        self.tail_f_thickness = params["tail_f_thickness"] if "tail_f_thickness" in params else self._tail_f_thickness
        self.head_f_thickness = params["head_f_thickness"] if "head_f_thickness" in params else self._head_f_thickness

        self.rough1 = params["rough1"] if "rough1" in params else self._rough1
        self.rough2 = params["rough2"] if "rough2" in params else self._rough2
        self.rough3 = params["rough3"] if "rough3" in params else self._rough3
        self.rough4 = params["rough4"] if "rough4" in params else self._rough4
        self.rough5 = params["rough5"] if "rough5" in params else self._rough5
        self.rough6 = params["rough6"] if "rough6" in params else self._rough6
        self.rough7 = params["rough7"] if "rough7" in params else self._rough7

        self.head_f_sfv  = params["head_f_sfv"]  if "head_f_sfv"  in params else self._head_f_sfv
        self.tail_f_sfv  = params["tail_f_sfv"]  if "tail_f_sfv"  in params else self._tail_f_sfv
        self.tail_a_sfv  = params["tail_a_sfv"]  if "tail_a_sfv"  in params else self._tail_a_sfv
        self.head_a_sfv  = params["head_a_sfv"]  if "head_a_sfv"  in params else self._head_a_sfv
        self.sio2_sfv    = params["sio2_sfv"]    if "sio2_sfv"    in params else self._sio2_sfv


def get_real_data(filename):
    """
    - Get a numpy array from the data file.
    - Translate the coordinate from qvector to incident angle.
    - Normalize intensity values.
    """
    real_data = np.loadtxt(filename, usecols=(0, 1))
    q = real_data[:, 0]
    real_data[:, 0] =  np.arcsin(WAVELENGTH * q /(4.0*np.pi))
    real_data[:, 1] /= real_data[0, 1]
    return real_data


def get_real_data_axis(filename):
    """
    Get axis coordinates of the experimental data
    :return: 1D array with axis coordinates
    """
    return get_real_data(filename)[:, 0]


def get_real_data_values(filename):
    """
    Get experimental data values as a 1D array
    :return: 1D array with experimental data values
    """
    return get_real_data(filename)[:, 1]


def sld_with_solvent(original_sld, solvent_sld, solvent_fraction):
    """
    :param original_sld: sld of the original material
    :param solvent_sld: sld of the solvent material
    :param solvent_fraction: fraction volume of solvent
    :return: overall sld of solvent + original material
    """
    if solvent_fraction < 0 or solvent_fraction > 1:
        print("original sld: {}".format(original_sld))
        print("solvent sld: {}".format(solvent_sld))
        print("solvent_fraction: {}".format(solvent_fraction))
        raise ValueError("Non physical solvent fraction found : {}"
                .format(solvent_fraction))

    osld = original_sld
    ssld = solvent_sld
    sf = solvent_fraction
    rsld = osld * sf + (1.0 - sf) * ssld
    return rsld


def get_sample(params):
    """
    Define the multilayer sample, using the parameters given.
    :return:  Multilayer
    """

    xpar = SampleParameters(params)

    # materials by SLD
    m_solvent = ba.MaterialBySLD("solvent", xpar.solvent_sld, 0.0)
    m_head_f  = ba.MaterialBySLD("head_f",  sld_with_solvent(xpar.head_f_sld, xpar.solvent_sld, xpar.head_f_sfv),  0.0)
    m_tail_f  = ba.MaterialBySLD("tail_f",  sld_with_solvent(xpar.tail_f_sld, xpar.solvent_sld, xpar.tail_f_sfv),  0.0)
    m_tail_a  = ba.MaterialBySLD("tail_a",  sld_with_solvent(xpar.tail_a_sld, xpar.solvent_sld, xpar.tail_a_sfv),  0.0)
    m_head_a  = ba.MaterialBySLD("head_a",  sld_with_solvent(xpar.head_a_sld, xpar.solvent_sld, xpar.head_a_sfv),  0.0)
    m_sio2    = ba.MaterialBySLD("sio2",    sld_with_solvent(xpar.sio2_sld, xpar.solvent_sld, xpar.sio2_sfv),  0.0)
    m_silicon = ba.MaterialBySLD("silicon", xpar.silicon_sld, 0.0)

    # Layers with some thickness
    silicon_layer = ba.Layer(m_silicon)
    sio2_layer    = ba.Layer(m_sio2,   xpar.sio2_thickness   * angstrom)
    head_a_layer  = ba.Layer(m_head_a, xpar.head_a_thickness * angstrom)
    tail_a_layer  = ba.Layer(m_tail_a, xpar.tail_a_thickness * angstrom)
    tail_f_layer  = ba.Layer(m_tail_f, xpar.tail_f_thickness * angstrom)
    head_f_layer  = ba.Layer(m_head_f, xpar.head_f_thickness * angstrom)
    solvent_layer = ba.Layer(m_solvent)

    # Multilayer with interlayer roughness
    roughness_Silicon_sio2  = ba.LayerRoughness(xpar.rough1*angstrom,0.0,0.0)
    roughness_sio2_HeadA    = ba.LayerRoughness(xpar.rough2*angstrom,0.0,0.0)
    roughness_HeadA_TailA   = ba.LayerRoughness(xpar.rough3*angstrom,0.0,0.0)
    roughness_TailA_TailF   = ba.LayerRoughness(xpar.rough4*angstrom,0.0,0.0)
    roughness_TailF_HeadF   = ba.LayerRoughness(xpar.rough5*angstrom,0.0,0.0)
    roughness_HeadF_solvent = ba.LayerRoughness(xpar.rough6*angstrom,0.0,0.0)
    roughness_Zero          = ba.LayerRoughness(xpar.rough7*angstrom,0.0,0.0)

    multi_layer = ba.MultiLayer()
    multi_layer.addLayerWithTopRoughness(silicon_layer, roughness_Zero)
    multi_layer.addLayerWithTopRoughness(sio2_layer,    roughness_Silicon_sio2)
    multi_layer.addLayerWithTopRoughness(head_a_layer,  roughness_sio2_HeadA)
    multi_layer.addLayerWithTopRoughness(tail_a_layer,  roughness_HeadA_TailA)
    multi_layer.addLayerWithTopRoughness(tail_f_layer,  roughness_TailA_TailF)
    multi_layer.addLayerWithTopRoughness(head_f_layer,  roughness_TailF_HeadF)
    multi_layer.addLayerWithTopRoughness(solvent_layer, roughness_HeadF_solvent)

    return multi_layer

def get_simulation(params):
    """
    - Defines the instrument and the beam.
    - Define the sample.
    - :return:  SpecularSimulation
    """
    wavelength = WAVELENGTH * ba.angstrom
    sampling_var = get_real_data_axis(DATA_TO_FIT)

    simulation = ba.SpecularSimulation()
    simulation.setBeamParameters(wavelength, sampling_var)
    simulation.setSample(get_sample(params))
    return simulation

def comparison_plot(fit_objective, title):
    """
    Given a fit (through the fit_objective), save a plot
    to visually compare it with the original data.
    The plot will be saved in the directory containing this
    script.
    """
    plt.figure()
    fit_result = fit_objective.simulationResult()
    data = fit_result.data()
    xbins = np.array(data.getAxis(0).getBinCenters())

    # We don't use get_real_data, as we want to compare
    # the results to the original data, which is in q space.
    qdata = np.loadtxt(DATA_TO_FIT)[:,0]
    idata = np.loadtxt(DATA_TO_FIT)[:,1]

    X1 = qdata
    Y1 = idata

    X2 = 4.0*np.pi*np.sin(xbins*np.pi/180.0)/WAVELENGTH
    Y2 = data.getArray()

    plt.semilogy(X1,Y1)
    plt.semilogy(X2,Y2)
    plt.xlim([0,0.6])
    plt.title(title)

    save_file=title+"_"+str(dtime.now().date())+"_"+str(dtime.now().time())+".png"
    plt.savefig(save_file)


def print_finish_message():
    print("   ,(   ,(   ,(   ,(   ,(   ,(   ,(   ,(")
    print("`-'  `-'  `-'  `-'  `-'  `-'  `-'  `-'  `")
    print("   ,(   ,(   ,(   ,(   ,(   ,(   ,(   ,(")
    print("`-'  `-'  `   Finished   `-'  `-'  `-'  `")
    print("   ,(   ,(   ,(   ,(   ,(   ,(   ,(   ,(")
    print("`-'  `-'  `-'  `-'  `-'  `-'  `-'  `-'  `")
    print("   ,(   ,(   ,(   ,(   ,(   ,(   ,(   ,(")



if __name__ == '__main__':
    # Uncomment the parameters you wish to fit
    # The values reported here produce the "best fit" (fit by human sight).
    # SLD values refer to the "effective" SLD of the layer, neglecting
    # any kind of solvation.
    # Implementing solvation treatment is work in progress.
    tunable_parameters = {
        "solvent_sld": 6.18989e-06,
    #   "head_f_sld":  1.88401254e-06,
    #   "head_a_sld":  3.60e-06,
    #   "tail_f_sld": -0.25e-06,
    #   "tail_a_sld": -0.37340153e-06,
    #   "sio2_sld":    3.47e-06,
    #   "silicon_sld": 2.07e-06,

    #   "head_f_thickness":  5.64468,
    #   "head_a_thickness":  9.86971,
    #   "tail_f_thickness": 13.7233,
    #   "tail_a_thickness": 13.7233,
    #   "sio2_thickness  ": 12.0885,

    #   "rough1": 4.80879,
    #   "rough2": 3.00,
    #   "rough3": 3.00,
    #   "rough4": 1.00e-7,
    #   "rough5": 3.00,
    #   "rough6": 1.64969,
    #   "rough7": 1.00e-7,

    #   "head_f_sfv": 0.9915395153082305,
    #   "tail_f_sfv": 0.9997855735077633,
    #   "head_a_sfv": 0.5670808231721158,
    #   "tail_a_sfv": 0.9997855735077633,
    #   "sio2_sfv": 1.0 - 0.127675,
    }
    params = ba.Parameters()
    for k,v in tunable_parameters.items():
        params.add(k, v, min=0.1*v, max=10*v)

    # Obtain the dataset to fit as a numpy array.
    real_data = get_real_data_values(DATA_TO_FIT)

    # Link the real data with a simulation.
    fit_objective = ba.FitObjective()
    fit_objective.addSimulationAndData(get_simulation, real_data, 1.0)
    fit_objective.initPrint(10)

    # Do the fitting.
    # To play around with the minimizers available, visit:
    # https://www.bornagainproject.org/documentation/working-with-python/fitting/fitting-highlights/minimizers/
    minimizer = ba.Minimizer()
    minimizer.setMinimizer("GSLMultiMin","BFGS2")
    result = minimizer.minimize(fit_objective.evaluate, params)
    fit_objective.finalize(result)

    # Plot a comparison of the fit results and the actual data:
    plot_title = "FitPlot"
    comparison_plot(fit_objective, plot_title)

    # Let the user know that the program finished:
    print_finish_message()
