"""
Run single simulation.
"""
import bornagain as ba
import matplotlib.pyplot as plt
import numpy as np


def beam_and_detector_setup(params):
    """
    Defines and returns a specular simulation.
    """
    simulation = ba.SpecularSimulation()
    scan_size = params["scan_size"]
    q_min = params["q_min"]
    q_max = params["q_max"]
    q_vals = np.linspace(q_min,q_max,scan_size,endpoint=False)
    scan = ba.QSpecScan(q_vals)
    simulation.setScan(scan)
    return q_vals, simulation


def run_simulation(params):
    """
    Runs simulation and returns its result.
    """
    q_vals, simulation = beam_and_detector_setup(params)
    sample = get_sample(params)

    simulation.setSample(sample)
    simulation.runSimulation()
    depth, slds = get_sld_profile_ba(params)
    return depth, slds, q_vals, simulation.result().array()


def get_sample(params):
    """
    Defines sample and returns it
    """

    if len(params["thickness"]) != len(params["sld"]):
        raise Exception("Number of thicknesses and number of slds differ")

    total_thickness = sum(params["thickness"])
    thickness_rescaling = params["total_sample_depth"] / total_thickness
    multi_layer = ba.MultiLayer()

    for index, (thickness, sld) in enumerate(zip(params["thickness"], params["sld"])):
        material = ba.MaterialBySLD(str(index), sld, 0.0)
        layer_thickness = thickness*thickness_rescaling
        layer = ba.Layer(material, layer_thickness)
        roughness = ba.LayerRoughness()
        roughness.setSigma(params["roughness"])
        multi_layer.addLayerWithTopRoughness(layer,roughness)

    return multi_layer


def get_sld_profile_ba(params, sample = None):
    n_depth_points = params["n_depth_points"]
    depth_min_to_sample = -1.0625 * params["total_sample_depth"]
    depth_max_to_sample =  2.0625 * params["total_sample_depth"]

    if sample is None:
        sample = get_sample(params)

    depth_values = np.linspace(depth_min_to_sample, depth_max_to_sample, n_depth_points, endpoint=False)
    zpoints, slds = ba.MaterialProfile(sample, len(depth_values), -depth_values[0], -depth_values[-1])
    depth = - np.flip(np.array(zpoints))
    slds = np.flip(np.real(np.array(slds)))
    return depth, slds


def build_sample_from_sld_prof(z_vals,sld_vals):
    """
    Defines sample and returns it
    """
    multi_layer = ba.MultiLayer()
    assert np.all(z_vals[1:] - z_vals[:-1] == z_vals[1] - z_vals[0])
    layer_thickness = abs(z_vals[1] - z_vals[0])
    zero_thickness = 0

    # Fronting
    material = ba.MaterialBySLD(str(-1), 0.0, 0.0)
    layer = ba.Layer(material, zero_thickness)
    multi_layer.addLayer(layer)

    for index, (z, sld) in enumerate(zip(z_vals, sld_vals)):
        material = ba.MaterialBySLD(str(index), sld, 0.0)
        layer = ba.Layer(material,layer_thickness)
        multi_layer.addLayer(layer)

    # Backing
    material = ba.MaterialBySLD(str(index+1), 1.0e-6, 0.0)
    layer = ba.Layer(material, zero_thickness)
    multi_layer.addLayer(layer)


    return multi_layer


def run_simulation_from_sldprof(params, z_vals=None, sld_vals=None):
    """
    Runs simulation and returns its result.
    """
    if z_vals is None: z_vals, _ = get_sld_profile_ba(params)
    if sld_vals is None: _, sld_vals = get_sld_profile_ba(params)

    q_vals, simulation = beam_and_detector_setup(params)
    sample = build_sample_from_sld_prof(z_vals, sld_vals)
    #z_new, sld_new = get_sld_profile_ba(params, sample)
    simulation.setSample(sample)
    simulation.runSimulation()

    return z_vals, sld_vals, q_vals, simulation.result().array()



def get_default_par_values_ba():
    default_values = {
        "thickness": np.array([0, 1, 1, 1, 0]),
        "sld": np.array([0, 5, 7, 3, 1])*1e-6,
        "scan_size": 1024,
        "q_min": 0.0,
        "q_max": 1.0,
        "total_sample_depth": 64,
        "n_depth_points": 1024,
        "roughness": 2.5
    }
    return default_values


def show_bornagain_inconsistency(filename=None):

    for _n_ in range(5,12):
        n_depth_points = 2**_n_

        params = get_default_par_values_ba()
        params["n_depth_points"] = n_depth_points

        d,s,q,r = run_simulation(params)
        d2,s2,q2,r2 = run_simulation_from_sldprof(params, d, s)

        params["roughness"] = 0

        d3,s3,q3,r3 = run_simulation(params)
        d4,s4,q4,r4 = run_simulation_from_sldprof(params)


        # plot:
        ax = plt.figure(figsize = (18,6))
        ax.suptitle(f"Number of depth points = {n_depth_points}")

        ax.add_subplot(121)
        plt.title("SLD profile")
        plt.plot(d,s*1e6, label="Layers with roughness")
        plt.plot(d2,s2*1e6+1, label="Continous SLD profile 1")
        plt.plot(d3,s3*1e6+2, label="Layers with no roughness")
        plt.plot(d4,s4*1e6+3, label="Continuous SLD profile 2")
        plt.legend()

        ax.add_subplot(122)
        plt.title("Reflectivity signal")
        plt.plot(q,np.log(r), label="Layers with roughness")
        plt.plot(q2,np.log(r2), label="Continous SLD profile 1")
        plt.plot(q3,np.log(r3), label="Layers with no roughness")
        plt.plot(q4,np.log(r4), label="Continuous SLD profile 2")
        plt.legend()

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename+str(n_depth_points)+".png")
            print(f"Figure written to {filename}_{str(n_depth_points)}.png")

#print(d)
#print(d2)
#print(q)
#print(q2)
if __name__ == "__main__":
    show_bornagain_inconsistency()
    #show_bornagain_inconsistency("./BornAgainInconsistency.png")
