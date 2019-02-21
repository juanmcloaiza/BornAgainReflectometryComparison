import numpy as np
import bornagain as ba
from bornagain import deg, angstrom
from matplotlib import pyplot as plt
from datetime import datetime as dtime
WAVELENGTH = 1.0 # Dummy wavelength.


class SampleParameters():
    """
    Class whose only scope is to store and return values for the different
    parameters of the simulation. The defaults are set to the ones found
    by fits produced using Refnx.
    The Refnx fit curve was generated using a Jupyter notebook downloaded from
    http://scripts.iucr.org/cgi-bin/paper?rg5158
    """

    def __init__(self, solv_sld=None):
        self.solvent_sld = 6.18989e-06
        if solv_sld is not None:
            self.solvent_sld = solv_sld

        self.head_f_sld = 1.88401254e-06
        self.tail_f_sld = -0.37340153e-06
        self.head_a_sld = 1.88401254e-06
        self.tail_a_sld = -0.37340153e-06
        self.sio2_sld = 3.47e-06
        self.silicon_sld = 2.07e-06

        # Thicknesses:
        self.head_f_thickness = 5.64468 * angstrom
        self.head_a_thickness = 9.86971 * angstrom
        self.tail_f_thickness = 13.7233 * angstrom
        self.tail_a_thickness = 13.7233 * angstrom
        self.sio2_thickness = 12.0885 * angstrom

        # Solvent volume fractions.
        # dry_fraction = V / A / t
        self.head_a_sfv = 319 / 56.9956 / self.head_a_thickness * angstrom
        self.tail_a_sfv = 782 / 56.9956 / self.tail_a_thickness * angstrom
        self.head_f_sfv = 319 / 56.9956 / self.head_f_thickness * angstrom
        self.tail_f_sfv = 782 / 56.9956 / self.tail_f_thickness * angstrom
        # self.head_f_sfv = 0.9915395153082305
        # self.tail_f_sfv = 0.9997855735077633
        # self.head_a_sfv = 0.5670808231721158
        # self.tail_a_sfv = 0.9997855735077633
        self.sio2_sfv = 1.0 - 0.127675

        # Roughnesses:
        factor = np.sqrt(1.0)
        # sig_ba = lambda x: np.sqrt(- x * x / 2. / np.log(0.5))
        self.sio2_rough = 4.80879 * angstrom * factor
        self.head_a_rough = 3.00 * angstrom * factor
        self.tail_a_rough = 3.00 * angstrom * factor
        self.tail_f_rough = 0.00 * angstrom * factor
        self.head_f_rough = 3.00 * angstrom * factor
        self.solvent_rough = 1.64969 * angstrom * factor

        # self.sio2_rough = sig_ba(4.80879 * angstrom)
        # self.head_a_rough = sig_ba(3.00 * angstrom)
        # self.tail_a_rough = sig_ba(3.00 * angstrom)
        # self.tail_f_rough = sig_ba(0.00 * angstrom)
        # self.head_f_rough = sig_ba(3.00 * angstrom)
        # self.solvent_rough = sig_ba(1.64969 * angstrom)


def get_real_data(filename):
    """
    - Get a numpy array from the data file.
    - Translate the coordinate from qvector to incident angle.
    - Normalize intensity values.
    """
    real_data = np.loadtxt(filename, usecols=(0, 1))
    q = real_data[:, 0]
    real_data[:, 0] = np.arcsin(WAVELENGTH * q / (4.0 * np.pi))
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


def sld_with_solvent(original_sld, solvent_sld, dry_fraction):
    """
    :param original_sld: sld of the original material
    :param solvent_sld: sld of the solvent material
    :param dry_fraction: volume fraction of dry material
    :return: overall sld of solvent + original material
    """
    if dry_fraction < 0 or dry_fraction > 1:
        print("original sld: {}".format(original_sld))
        print("solvent sld: {}".format(solvent_sld))
        print("dry_fraction: {}".format(dry_fraction))
        raise ValueError("Non physical dry fraction found : {}"
                .format(dry_fraction))

    osld = original_sld
    ssld = solvent_sld
    df = dry_fraction
    rsld = osld * df + (1.0 - df) * ssld
    return rsld


def get_sample(xpar):
    """
    Define the multilayer sample, using the sample parameters given.
    :param xpar: instance of SampleParameters, relevant parameters for the simulation.
    :return: instance of the generated Multilayer.
    """
    assert isinstance(xpar, SampleParameters), "argument of get_sample must be an instance of SampleParameters"
    head_f_sld = sld_with_solvent(xpar.head_f_sld, xpar.solvent_sld, xpar.head_f_sfv)
    tail_f_sld = sld_with_solvent(xpar.tail_f_sld, xpar.solvent_sld, xpar.tail_f_sfv)
    tail_a_sld = sld_with_solvent(xpar.tail_a_sld, xpar.solvent_sld, xpar.tail_a_sfv)
    head_a_sld = sld_with_solvent(xpar.head_a_sld, xpar.solvent_sld, xpar.head_a_sfv)
    sio2_sld = sld_with_solvent(xpar.sio2_sld, xpar.solvent_sld, xpar.sio2_sfv)

    # materials by SLD
    m_solvent = ba.MaterialBySLD("solvent", xpar.solvent_sld, 0.0)
    m_head_f = ba.MaterialBySLD("head_f", head_f_sld, 0.0)
    m_tail_f = ba.MaterialBySLD("tail_f", tail_f_sld, 0.0)
    m_tail_a = ba.MaterialBySLD("tail_a", tail_a_sld, 0.0)
    m_head_a = ba.MaterialBySLD("head_a", head_a_sld, 0.0)
    m_sio2 = ba.MaterialBySLD("sio2", sio2_sld, 0.0)
    m_silicon = ba.MaterialBySLD("silicon", xpar.silicon_sld, 0.0)

    # Layers with some thickness
    silicon_layer = ba.Layer(m_silicon)
    sio2_layer = ba.Layer(m_sio2, xpar.sio2_thickness)
    head_a_layer = ba.Layer(m_head_a, xpar.head_a_thickness)
    tail_a_layer = ba.Layer(m_tail_a, xpar.tail_a_thickness)
    tail_f_layer = ba.Layer(m_tail_f, xpar.tail_f_thickness)
    head_f_layer = ba.Layer(m_head_f, xpar.head_f_thickness)
    solvent_layer = ba.Layer(m_solvent)

    # Interlayer roughness
    roughness_Silicon_sio2 = ba.LayerRoughness(xpar.sio2_rough, 0.0, 0.0)
    roughness_sio2_HeadA = ba.LayerRoughness(xpar.head_a_rough, 0.0, 0.0)
    roughness_HeadA_TailA = ba.LayerRoughness(xpar.tail_a_rough, 0.0, 0.0)
    roughness_TailA_TailF = ba.LayerRoughness(xpar.tail_f_rough, 0.0, 0.0)
    roughness_TailF_HeadF = ba.LayerRoughness(xpar.head_f_rough, 0.0, 0.0)
    roughness_HeadF_solvent = ba.LayerRoughness(xpar.solvent_rough, 0.0, 0.0)

    # Define multilayer
    multi_layer = ba.MultiLayer()
    multi_layer.addLayer(silicon_layer)
    multi_layer.addLayerWithTopRoughness(sio2_layer, roughness_Silicon_sio2)
    multi_layer.addLayerWithTopRoughness(head_a_layer, roughness_sio2_HeadA)
    multi_layer.addLayerWithTopRoughness(tail_a_layer, roughness_HeadA_TailA)
    multi_layer.addLayerWithTopRoughness(tail_f_layer, roughness_TailA_TailF)
    multi_layer.addLayerWithTopRoughness(head_f_layer, roughness_TailF_HeadF)
    multi_layer.addLayerWithTopRoughness(solvent_layer, roughness_HeadF_solvent)

    return multi_layer


def get_simulation(filename, bckgr=None):
    """
    - Defines the instrument and the beam.
    - Define the sample.
    - :return:  SpecularSimulation
    """
    wavelength = WAVELENGTH * angstrom
    theta_range = get_real_data_axis(filename)
    n_sig = 3
    n_points = 25
    simulation = ba.SpecularSimulation()
    # Add footprint
    simulation.setBeamParameters(wavelength, theta_range)

    # Add beam divergence
    simulation.addParameterDistribution("*/Beam/Wavelength",
                                        ba.DistributionGaussian(wavelength, 0.02 * angstrom),
                                        n_points, n_sig)

    if bckgr is not None:
        backgr = ba.ConstantBackground(bckgr)
        simulation.setBackground(backgr)

    return simulation


def save_text_file(data, filename):
    intensity = data.getArray()
    x_axis = data.getAxis(0).getBinCenters()
    array_to_export = np.column_stack((x_axis, intensity))
    np.savetxt(filename, array_to_export)
    return


def comparison_plot(datasim, datafile, title):
    """
    Given a fit (through the fit_objective), save a plot
    to visually compare it with the original data.
    The plot will be saved in the directory containing this
    script.
    """
    # Font sizes and line widths
    FigSize = 10
    FontSize = 20
    BorderWidth = 3
    plt.rcParams.update({'font.size': FontSize})
    plt.rcParams.update({'axes.linewidth': BorderWidth})
    plt.figure(figsize=(1.5 * FigSize, FigSize))
    plt.tick_params(width=BorderWidth, length=FontSize, which='major')
    plt.tick_params(width=BorderWidth, length=0.3 * FontSize, which='minor')

    # We don't use get_real_data, as we want to compare
    # the results to the original data, which is in q space.
    qdata = np.loadtxt(datafile)[:, 0]
    idata = np.loadtxt(datafile)[:, 1]
    error = np.loadtxt(datafile)[:, 2]

    Xdata = qdata
    Ydata = idata

    xbins = np.array(datasim.getAxis(0).getBinCenters())
    Xsimu = 4.0 * np.pi * np.sin(xbins * np.pi / 180.0) / WAVELENGTH
    Ysimu = datasim.getArray()

    rel_error = 2. * np.abs(Ydata - Ysimu) / np.abs(Ydata + Ysimu)
    mean_err = rel_error.mean()

    plt.errorbar(Xdata, Ydata, yerr=error, linewidth=0.2 * FontSize, markersize=FontSize, label="Fit by Refnx")
    plt.errorbar(Xsimu, Ysimu, linewidth=0.2 * FontSize, markersize=FontSize, label="Simulation by BornAgain")

    plt.semilogy(Xsimu, rel_error, 'k--', label="Relative Error", alpha=0.3)
    plt.axhline(y=mean_err, color='k', label="Mean Relative Error = $" + str(np.round(mean_err, 3)) + "$")

    plt.yscale('log')
    plt.xlim([0, 0.3])
    plt.title(title)

    plt.ylabel('Reflectivity')
    plt.xlabel('Q /$\AA^{-1}$')
    plt.legend()

    fig = plt.gcf()
    save_file = title + "_" + str(dtime.now().date()) + "_" + str(dtime.now().time()) + ".png"
    plt.savefig(save_file)
    plt.show()


sld_solvents = {
    'd2o': 6.18989e-06,
    'h2o': -0.56e-06,
    'hdmix': SampleParameters().silicon_sld
}
wavelength_spread = 0.02
beam_divergence = 0.02

names = [
    r'd2o', r'h2o', r'hdmix'
]

backgrounds = [None, 6.16827e-08, 2.60714e-07]

simulation_data = {}
for name, background in zip(names, backgrounds):
    sld_solv  = sld_solvents[name]
    in_RefnxFitFile = "./FitByRefnx_"+name+".txt"
    sample_parameters = SampleParameters(sld_solv)
    sample = get_sample(sample_parameters)
    simulation = get_simulation(in_RefnxFitFile, background)
    simulation.setSample(sample)
    simulation.runSimulation()
    results = simulation.result()
    simulation_data[name] = results.data()

FigSize = 15
FontSize = 15
BorderWidth = 0


fig, axs = plt.subplots(2, 2, figsize=(FigSize, FigSize))

for i, name in enumerate(names):
    sld = sld_solvents[name]

    # Real data points:
    qdata = np.loadtxt("./RefnxData_" + name + ".dat")[:, 0]
    idata = np.loadtxt("./RefnxData_" + name + ".dat")[:, 1]
    Xdata = qdata
    Ydata = idata

    # Bornagain simulation:
    datasim = simulation_data[name]
    xbins = np.array(datasim.getAxis(0).getBinCenters())
    Xsimu = 4.0 * np.pi * np.sin(xbins * np.pi / 180.0) / WAVELENGTH
    Ysimu = datasim.getArray()

    # Fit with Refnx model:
    qdata = np.loadtxt("./FitByRefnx_" + name + ".txt")[:, 0]
    idata = np.loadtxt("./FitByRefnx_" + name + ".txt")[:, 1]
    Xrfnx = qdata
    Yrfnx = idata

    # CURRENT SUBPLOT:
    I = (i // 3) // 2
    J = (i // 3) % 2
    shift = 0.01 ** (i % 3)

    # data:
    axs[I, J].tick_params(width=BorderWidth, length=0.3 * FontSize, which='major')
    axs[I, J].tick_params(width=BorderWidth, length=0.3 * FontSize, which='minor')

    # refnx fit:
    axs[I, J].semilogy(Xrfnx, Yrfnx * shift,
                       linewidth=0.2 * FontSize,
                       ls='--',
                       markersize=FontSize,
                       label=str(name) + " - Refnx")

    # bornagain simulation:
    axs[I, J].semilogy(Xsimu, Ysimu * shift,
                       linewidth=0.2 * FontSize,
                       ls='-',
                       markersize=FontSize,
                       label=str(name) + " - BornAgain")

    # Scale, titles, limits and labels:
    axs[I, J].set_yscale("log", nonposy='clip')
    axs[I, J].set_xlim([0, 0.3])
    axs[I, J].set_ylim([1e-12, 1e1])

    if J == 0:
        axs[I, J].set_ylabel('Reflectivity')

    if I == 1:
        axs[I, J].set_xlabel('Q /$\AA^{-1}$')
    # axs[I,J].legend()
    current_plot = (I, J)

fig = plt.gcf()
plt.show()