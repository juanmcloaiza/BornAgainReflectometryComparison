import numpy as np
import bornagain as ba
from bornagain import deg, angstrom
from matplotlib import pyplot as plt
from os import path
WAVELENGTH = 1.0 # Dummy for this exercise.
DATA_TO_FIT="./FitByRefnx.txt"


class SampleParameters():
    """
    Class whose only scope is to return store and return values
    for the different parameters of the simulation.
    """
    _solvent_sld =  6.18989e-06
    _head_f_sld  =  1.88401254e-06
    _head_a_sld  =  3.60e-06
    _tail_f_sld  = -0.25e-06
    _tail_a_sld  = -0.37340153e-06
    _sio2_sld    =  3.47e-06
    _silicon_sld =  2.07e-06

    _head_f_thickness =  5.64468
    _head_a_thickness =  9.86971
    _tail_f_thickness = 13.7233
    _tail_a_thickness = 13.7233
    _sio2_thickness   = 12.0885

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


def get_real_data():
    """
    Loading data from genx_interchanging_layers.dat
    Returns a Nx2 array (N - the number of experimental data entries)
    with first column being coordinates,
    second one being values.
    """
    if not hasattr(get_real_data, "data"):
        filename = DATA_TO_FIT
        real_data = np.loadtxt(filename, usecols=(0, 1))

        # translate from 2 times incident angle (degrees)
        # to incident angle (radians)
        q = real_data[:, 0]
        real_data[:, 0] =  np.arcsin(WAVELENGTH * q /(4.0*np.pi))
        real_data[:, 1] *= real_data[0, 1]
        get_real_data.data = real_data
    return get_real_data.data.copy()

def get_real_data_axis():
    """
    Get axis coordinates of the experimental data
    :return: 1D array with axis coordinates
    """
    return get_real_data()[:, 0]

def get_real_data_values():
    """
    Get experimental data values as a 1D array
    :return: 1D array with experimental data values
    """
    return get_real_data()[:, 1]

def get_sample(params):
    """
    Defines sample and returns it
    """

    xpar = SampleParameters(params)
    print(xpar.solvent_sld)

    #materials by SLD
    m_solvent = ba.MaterialBySLD("solvent", xpar.solvent_sld, 0.0)
    m_head_f  = ba.MaterialBySLD("head_f",  xpar.head_f_sld,  0.0)
    m_tail_f  = ba.MaterialBySLD("tail_f",  xpar.tail_f_sld,  0.0)
    m_tail_a  = ba.MaterialBySLD("tail_a",  xpar.tail_a_sld,  0.0)
    m_head_a  = ba.MaterialBySLD("head_f",  xpar.head_a_sld,  0.0)
    m_sio2    = ba.MaterialBySLD("sio2",    xpar.sio2_sld,    0.0)
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
    Create and return specular simulation with its instrument defined
    """
    wavelength = WAVELENGTH * ba.angstrom
    sampling_var = get_real_data_axis()

    simulation = ba.SpecularSimulation()
    simulation.setBeamParameters(wavelength, sampling_var)
    simulation.setSample(get_sample(params))
    return simulation

def my_plot(fit_objective, title):
    plt.figure()
    fit_result = fit_objective.simulationResult()
    data = fit_result.data()
    xbins = np.array(data.getAxis(0).getBinCenters())

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

    plt.savefig(title+".png")

tunable_parameters = {"solvent_sld":6.0e-6}
real_data = get_real_data_values()
fit_objective = ba.FitObjective()
fit_objective.addSimulationAndData(get_simulation, real_data, 1.0)
minimizer = ba.Minimizer()
params = ba.Parameters()

for k,v in tunable_parameters.items():
    params.add(k, v, min=0.1*v, max=10*v)

plot_observer = ba.PlotterSpecular()
fit_objective.initPrint(10)
fit_objective.initPlot(10, plot_observer)

result = minimizer.minimize(fit_objective.evaluate, params)
fit_objective.finalize(result)

plot_title = ""
for k,v in tunable_parameters.items():
    plot_title += k + "(before fitting)=" + "{:.2f}".format(1e6*v)
my_plot(fit_objective, plot_title)
print("Finished")
