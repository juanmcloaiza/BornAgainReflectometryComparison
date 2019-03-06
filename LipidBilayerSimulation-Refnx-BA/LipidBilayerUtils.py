import numpy as np
import bornagain as ba
from bornagain import deg, angstrom
from matplotlib import pyplot as plt
from datetime import datetime as dtime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
WAVELENGTH = 1.0

class CustomFitObjective(ba.FitObjective):
    def __init__(self):
        ba.FitObjective.__init__(self)

    def evaluate(self, params):

        # Evaluate residuals needs to be called always:
        bla = self.evaluate_residuals(params)

        sim = (np.asarray(self.simulation_array()))
        exp = (np.asarray(self.experimental_array()))

        l_sim = -np.log(sim)
        l_exp = -np.log(exp)
        eps = (np.sum(np.abs(l_exp))/l_exp.size) * 1e-14
        l_sim_exp_diff = ((l_sim - l_exp)/(eps + l_sim + l_exp))**2

        return l_sim_exp_diff.sum()

class SampleParameters():
    """
    Class whose only scope is to store and return values for the different
    parameters of the simulation. The defaults are set to the ones found
    by fits produced using Refnx.
    The Refnx fit curve was generated using a Jupyter notebook downloaded from
    http://scripts.iucr.org/cgi-bin/paper?rg5158
    """

    # Refnx weird parameters:
    _apm = 56.96436214318241
    _b_heads = 0.000601
    _v_heads = 319
    _b_tails = -0.000292
    _v_tails = 782

    # Scattering length densities:
    _d2o_sld = 6.189356429014113e-06
    _h2o_sld = -0.56e-06
    _tail_sld  = _b_tails/_v_tails
    _head_sld  = _b_heads/_v_heads
    _sio2_sld    =  3.47e-06
    _si_sld =  2.07e-06

    #Thicknesses:
    _head_f_thickness =  5.648173678129387
    _head_a_thickness =  9.815135921819564
    _tail_f_thickness = 13.730779217746703
    _tail_a_thickness = 13.730779217746703
    _sio2_thickness   = 12.159613125614271
    _solvent_thickness = 0.0

    # Solvent volume fractions.
    _head_f_sfv = _v_heads / _apm / _head_f_thickness
    _tail_f_sfv = _v_tails / _apm / _tail_f_thickness
    _head_a_sfv = _v_heads / _apm / _head_a_thickness
    _tail_a_sfv = _v_tails / _apm / _tail_a_thickness
    _sio2_sfv = 1.0 - 0.12836029654068992

    # Roughnesses:
    _sio2_rough    = 4.328939010929362
    _head_a_rough  = 3.00
    _tail_a_rough  = 3.00
    _tail_f_rough  = 0.00
    _head_f_rough  = 3.00
    _solvent_rough = 1.9634717423676802

    # Background:
    _d2o_bkg = 0.0
    _h2o_bkg = 6.16827e-08
    _hdmix_bkg = 2.60714e-07


    def __init__(self,params={}):
        assert isinstance(params, dict), "SampleParameters class must be initialized by an instance of dict"


        self.solvent_sld = params["solvent_sld"] if "solvent_sld" in params else self._si_sld
        self.d2o_sld = params["d2o_sld"] if "d2o_sld" in params else self._d2o_sld
        self.h2o_sld = params["h2o_sld"] if "h2o_sld" in params else self._h2o_sld
        self.head_sld  = params["head_sld"]  if "head_sld"  in params else self._head_sld
        self.tail_sld  = params["tail_sld"]  if "tail_sld"  in params else self._tail_sld
        self.sio2_sld    = params["sio2_sld"]    if "sio2_sld"    in params else self._sio2_sld
        self.si_sld = params["si_sld"] if "si_sld" in params else self._si_sld

        self.sio2_thickness   = params["sio2_thickness"]   if "sio2_thickness"   in params else self._sio2_thickness
        self.head_a_thickness = params["head_a_thickness"] if "head_a_thickness" in params else self._head_a_thickness
        self.tail_a_thickness = params["tail_a_thickness"] if "tail_a_thickness" in params else self._tail_a_thickness
        self.tail_f_thickness = params["tail_f_thickness"] if "tail_f_thickness" in params else self._tail_f_thickness
        self.head_f_thickness = params["head_f_thickness"] if "head_f_thickness" in params else self._head_f_thickness
        self.solvent_thickness = params["solvent_thickness"] if "solvent_thickness" in params else self._solvent_thickness

        self.sio2_rough = params["sio2_rough"] if "sio2_rough" in params else self._sio2_rough
        self.head_a_rough = params["head_a_rough"] if "head_a_rough" in params else self._head_a_rough
        self.tail_a_rough = params["tail_a_rough"] if "tail_a_rough" in params else self._tail_a_rough
        self.tail_f_rough = params["tail_f_rough"] if "tail_f_rough" in params else self._tail_f_rough
        self.head_f_rough = params["head_f_rough"] if "head_f_rough" in params else self._head_f_rough
        self.solvent_rough = params["solvent_rough"] if "solvent_rough" in params else self._solvent_rough

        self.head_f_sfv  = params["head_f_sfv"]  if "head_f_sfv"  in params else self._head_f_sfv
        self.tail_f_sfv  = params["tail_f_sfv"]  if "tail_f_sfv"  in params else self._tail_f_sfv
        self.tail_a_sfv  = params["tail_a_sfv"]  if "tail_a_sfv"  in params else self._tail_a_sfv
        self.head_a_sfv  = params["head_a_sfv"]  if "head_a_sfv"  in params else self._head_a_sfv
        self.sio2_sfv    = params["sio2_sfv"]    if "sio2_sfv"    in params else self._sio2_sfv


        self.d2o_bkg    = params["d2o_bkg"]    if "d2o_bkg"    in params else self._d2o_bkg
        self.h2o_bkg    = params["h2o_bkg"]    if "h2o_bkg"    in params else self._h2o_bkg
        self.hdmix_bkg    = params["hdmix_bkg"]    if "hdmix_bkg"    in params else self._hdmix_bkg


def overall_sld(original_sld,solvent_sld,solvent_fraction):
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
        raise ValueError("Non physical solvent fraction found : {}".format(solvent_fraction))

    osld = original_sld
    ssld = solvent_sld
    sf = solvent_fraction
    rsld = osld * sf + ssld - sf * ssld
    return rsld


def MyLipidLeaflet(top_material, top_thickness, top_sfv, rough_top_with_external,
                   bottom_material, bottom_thickness, bottom_sfv, rough_top_bottom,
                   solvent_material):

    # generate new materials with solvent-corrected sld
    top_mat_solv_sld = overall_sld(top_material.materialData(), solvent_material.materialData(), top_sfv)
    bot_mat_solv_sld = overall_sld(bottom_material.materialData(), solvent_material.materialData(), bottom_sfv)
    top_mat_solv = ba.MaterialBySLD(top_material.getName(), top_mat_solv_sld.real,top_mat_solv_sld.imag)
    bot_mat_solv = ba.MaterialBySLD(bottom_material.getName(), bot_mat_solv_sld.real,bot_mat_solv_sld.imag)

    lipid_leaflet = []
    top_layer = ba.Layer(top_mat_solv, top_thickness * angstrom)
    bottom_layer = ba.Layer(bot_mat_solv, bottom_thickness * angstrom)

    rough_1 = ba.LayerRoughness(rough_top_with_external * angstrom, 0.0, 0.0)
    rough_2 = ba.LayerRoughness(rough_top_bottom * angstrom, 0.0, 0.0)

    lipid_leaflet.append((top_layer,  rough_1))
    lipid_leaflet.append((bottom_layer, rough_2))

    return lipid_leaflet


def MyLipidBilayer(top_head_material, top_head_thickness, top_head_sfv, rough_top_with_external,
                   top_tail_material, top_tail_thickness, top_tail_sfv, rough_top_head_tail,
                   bottom_tail_material, bottom_tail_thickness, bottom_tail_sfv, rough_top_bottom_tails,
                   bottom_head_material, bottom_head_thickness, bottom_head_sfv, rough_bottom_head_tail,
                   solvent_material):

    top_leaflet = MyLipidLeaflet(top_head_material, top_head_thickness, top_head_sfv, rough_top_with_external,
                                 top_tail_material, top_tail_thickness, top_tail_sfv, rough_top_head_tail,
                                 solvent_material)

    bottom_leaflet = MyLipidLeaflet(bottom_tail_material, bottom_tail_thickness, bottom_tail_sfv, rough_top_bottom_tails,
                                    bottom_head_material, bottom_head_thickness, bottom_head_sfv, rough_bottom_head_tail,
                                    solvent_material)


    return top_leaflet + bottom_leaflet
#
# Utility functions
#

def get_real_data(filename):
    """
    - Get a numpy array from the data file.
    - Normalize intensity values.
    """
    real_data = np.loadtxt(filename, usecols=(0, 1))
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

def save_text_file(x,y, filename):
    intensity = y
    x_axis = x
    array_to_export = np.column_stack((x_axis, intensity))
    np.savetxt(filename,array_to_export)
    return

def comparison_plot(datafile,
                    refnx_sim_file,
                    ba_sim_file,
                    title="", shift = 0.0,
                    nrows=2, ncols=2, datasets_per_plot=3,
                    I = 0, J = 0,
                    fig = None, axs = None, inset = None):
    """
    Given a fit (through the fit_objective), save a plot
    to visually compare it with the original data.
    The plot will be saved in the directory containing this
    script.
    """
    FigSize = 20
    FontSize = 20
    BorderWidth = 3
    if (fig is None) and (axs is None):
        plt.rcParams.update({'font.size': FontSize})
        plt.rcParams.update({'axes.linewidth': BorderWidth})
        fig, axs = plt.subplots(nrows, ncols, figsize=(FigSize,FigSize))#,sharex=True,sharey=True)
        inset = {}

    if ncols * nrows < 2:
        axs = np.atleast_2d(np.array(axs))

    # Real data points (Q space):
    Xdata = np.loadtxt(datafile)[:,0]
    Ydata = np.loadtxt(datafile)[:,1]
    try:
        erbar = np.loadtxt(datafile)[:,2]
    except:
        erbar = np.zeros_like(Ydata)

    # Refnx model data ():
    Xrfnx = np.loadtxt(refnx_sim_file)[:,0]
    Yrfnx = np.loadtxt(refnx_sim_file)[:,1]

    # Bornagain model data:
    Xsimu = np.loadtxt(ba_sim_file)[:,0]#np.array(datasim.getAxis(0).getBinCenters())
    Ysimu = np.loadtxt(ba_sim_file)[:,1]#np.array(datasim.getAxis(0).getBinCenters())#datasim.getArray()

    #Refnx-Bornagain models difference:
    refnx_ba_diff  = 2.*np.abs(Yrfnx-Ysimu)/np.abs(Yrfnx+Ysimu)
    mean_diff = refnx_ba_diff.mean()

    #data:
    axs[I,J].tick_params(width=BorderWidth, length=0.3*FontSize, which='major')
    axs[I,J].tick_params(width=BorderWidth, length=0.3*FontSize, which='minor')
    base_line = axs[I,J].errorbar(Xdata, Ydata*shift, yerr=erbar*shift,
                      linewidth = 0.1*FontSize,
                      linestyle = ':',
                      marker = '.',
                      markersize=0.1*FontSize,
                      label = "Data")
    current_color=base_line.lines[0].get_color()

    #refnx fit:
    axs[I,J].errorbar(Xrfnx, Yrfnx*shift,
                      color = current_color,
                      linewidth = 0.2*FontSize,
                      ls = '--',
                      markersize=FontSize,
                      label = "Refnx")

    #bornagain simulation:
    axs[I,J].errorbar(Xsimu, Ysimu*shift,
                      color = current_color,
                      linewidth = 0.2*FontSize,
                      ls = '-',
                      markersize=FontSize,
                      label = "BornAgain")

    # CURRENT INSET:
    if not (I,J) in inset.keys():
        inset[(I,J)] = inset_axes(axs[I,J],
                        width="30%",
                        height="30%",
                        )
    current_inset = inset[(I,J)]
    current_inset.semilogy(Xsimu, refnx_ba_diff,'.', color = current_color, label = "Relative Error", alpha = 0.3)
    current_inset.axhline(y=mean_diff,color=current_color,ls='--', label="Mean Relative Error = $" + str(np.round(mean_diff,3)) + "$")
    #current_inset.set_ylim([1e-2,5])
    #current_inset.set_xlim([0,0.3])
    current_inset.tick_params(width=BorderWidth, length=0.2*FontSize, which='minor')
    current_inset.tick_params(width=BorderWidth, length=0.3*FontSize, which='major')
    #current_inset.set_xticks(np.linspace(0,0.2,3),minor=False)
    #current_inset.set_yticks(np.linspace(1e-1,0.2,3),minor=False)
    #current_inset.grid()



    # Scale, titles, limits and labels:
    axs[I,J].set_yscale("log", nonposy='clip')
    #axs[I,J].set_xlim([0,0.3])
    #axs[I,J].set_ylim([1e-12,1e1])
    axs[I,J].set_title(title)

    if J == 0:
        axs[I,J].set_ylabel('Reflectivity')

    if I == ncols-1:
        axs[I,J].set_xlabel('Q /$\AA^{-1}$')
    #axs[I,J].legend()
    current_plot = (I,J)

    return (fig, axs, inset)

def get_simulation(filename, params):
    """
    Create and return specular simulation with its instrument defined
    """
    wavelength = WAVELENGTH * ba.angstrom
    qvec = get_real_data_axis(filename)

    theta = np.arcsin( WAVELENGTH * qvec /(4.0*np.pi) )

    simulation = ba.SpecularSimulation()
    simulation.setBeamParameters(wavelength, theta)
    xpar = SampleParameters(params)
    simulation.setSample(get_lipid_bilayer_sample(xpar))
    return simulation

def plot_comparison_filenames(DataFile, OtherCodeSimFile, BornAgainSimFile, to_title):
    rfig, raxs, rinset = None, None, None
    for k in [0]:
        i = (k//3)//2
        j = (k//3)%2
        yshift = 0.01**(k%3)
        rfig, raxs, rinset = comparison_plot(
                        DataFile,
                        OtherCodeSimFile,
                        BornAgainSimFile,
                        title=to_title,
                        shift = yshift,
                        nrows=1, ncols=1, datasets_per_plot=3,
                        I = i, J = j,
                        fig = rfig, axs = raxs, inset = rinset)

def get_lipid_bilayer_sample(samp_par):
    assert isinstance(samp_par, SampleParameters), "samp_par argument must be an instance of a SampleParameters class"

    m_air = ba.MaterialBySLD("air", 0.0, 0.0)
    m_solvent = ba.MaterialBySLD("solvent",samp_par.solvent_sld,0.0)
    m_heads = ba.MaterialBySLD("head",samp_par.head_sld,0.0)
    m_tails = ba.MaterialBySLD("tail",samp_par.tail_sld,0.0)
    m_sio2_solvent   = ba.MaterialBySLD("sio2_solvent",  overall_sld(samp_par.sio2_sld, samp_par.solvent_sld, samp_par.sio2_sfv),0.0)
    m_si = ba.MaterialBySLD("si",samp_par.si_sld,0.0)


    lipid_bilayer_solvent = MyLipidBilayer(m_heads, samp_par.head_a_thickness, samp_par.head_a_sfv, samp_par.head_a_rough,
                                           m_tails, samp_par.tail_a_thickness, samp_par.tail_a_sfv, samp_par.tail_a_rough,
                                           m_tails, samp_par.tail_f_thickness, samp_par.tail_f_sfv, samp_par.tail_f_rough,
                                           m_heads, samp_par.head_f_thickness, samp_par.head_f_sfv, samp_par.head_f_rough,
                                           m_solvent)

    air_layer = (ba.Layer(m_air,0),
                     ba.LayerRoughness(0.0,0.0,0.0))

    si_layer = (ba.Layer(m_si),
                     ba.LayerRoughness(0.0,0.0,0.0))

    sio2_solvent_layer = (ba.Layer(m_sio2_solvent, samp_par.sio2_thickness * angstrom),
                  ba.LayerRoughness(samp_par.sio2_rough * angstrom,0.0,0.0))

    solvent_layer = (ba.Layer(m_solvent),
                 ba.LayerRoughness(samp_par.solvent_rough * angstrom,0.0,0.0))

   # solvent_layers =   [air_layer, si_layer, sio2_solvent_layer] +   lipid_bilayer_solvent +   [solvent_layer]
    solvent_layers =   [si_layer, sio2_solvent_layer] +   lipid_bilayer_solvent +   [solvent_layer]


    solvent_multilayer = ba.MultiLayer()


    for lay in solvent_layers:
        solvent_multilayer.addLayerWithTopRoughness(lay[0],lay[1])

    return solvent_multilayer
