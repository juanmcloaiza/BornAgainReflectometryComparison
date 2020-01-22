import refnx
from refnx.reflect import SLD, ReflectModel, Slab, Structure
import numpy as np
import matplotlib.pyplot as plt


def refnx_reflectivity(structure,qvals):
    return ReflectModel(structure, scale=1.0, bkg=0.0, dq=0.0).model(qvals)


def get_structure(params):
    total_thickness = sum(params["thickness"])
    thickness_rescaling = params["total_sample_depth"] / total_thickness
    thick_list, sld_list = params["thickness"], params["sld"]
    rough_list = [0 for _ in sld_list]

    assert len(sld_list) == len(thick_list), "slds and thicknesses have different lengths"
    assert len(sld_list) == len(rough_list), "slds list must be longer than roughnesses list by one"

    thickness_arr = np.array(thick_list)*thickness_rescaling
    sld_arr = np.array(sld_list)
    roughness_arr = params["roughness"]*np.ones(len(rough_list))

    slab = np.vectorize(Slab)(
        thickness_arr,
        sld_arr,
        roughness_arr
    )
    structure = Structure()
    np.vectorize(structure.append)(slab)
    return structure


def build_structure_from_sld_prof(depth_values, sld_values):

    assert np.all(depth_values[1:] - depth_values[:-1] == depth_values[1] - depth_values[0])
    layer_thickness = depth_values[1] - depth_values[0]
    zero_thickness = 0

    thickness_arr = np.zeros(len(depth_values)+2)
    thickness_arr[0] = zero_thickness
    thickness_arr[1:-1] = layer_thickness
    thickness_arr[-1] = zero_thickness

    sld_arr = np.zeros(len(depth_values)+2)
    sld_arr[1:-1] = sld_values
    sld_arr[-1] = 1

    rough_arr = np.zeros(thickness_arr.shape)

    slab = np.vectorize(Slab)(
        thickness_arr,
        sld_arr,
        rough_arr,
    )
    structure = Structure()
    np.vectorize(structure.append)(slab)
    return structure



def get_sld_profile_refnx(params, structure=None):
    n_depth_points = params["n_depth_points"]
    depth_min_to_sample = -1.0625 * params["total_sample_depth"]
    depth_max_to_sample =  2.0625 * params["total_sample_depth"]

    if structure is None:
        structure = get_structure(params)

    depth_values = np.linspace(depth_min_to_sample, depth_max_to_sample, n_depth_points, endpoint=False)
    z, rho = refnx.reflect.sld_profile(structure.slabs(), depth_values)
    return z, rho


def sld_profile_and_reflectivity(params):
    """
    Given some lists of slds, thicknesses and roughnesses, this function
    returns depth, sld_values, qvalues and reflectivity values.
    The sld_profile returned (i.e. depth, sld_values) DOES NOT correspond
    to the sld_profile used for the calculations and instead is a proxy
    using the given thicknesses and roughnesses.
    """

    qvals = np.linspace(params["q_min"],params["q_max"],params["scan_size"],endpoint=False)
    structure = get_structure(params)
    depth, sld_values = get_sld_profile_refnx(params, structure)

    reflectivity_values = refnx_reflectivity(structure,qvals)

    return depth, sld_values, qvals, reflectivity_values



def sld_profile_and_reflectivity_from_sld_profile(params, depth_values=None, sld_values=None):
    """
    Given some lists of slds, thicknesses and roughnesses, this function
    returns depth, sld_values, qvalues and reflectivity values.
    The sld_profile returned (i.e. depth, sld_values) DOES NOT correspond
    to the sld_profile used for the calculations and instead is a proxy
    using the given thicknesses and roughnesses.
    """
    qvals = np.linspace(params["q_min"],params["q_max"],params["scan_size"],endpoint=False)

    if depth_values is None: depth_values, _ = get_sld_profile_refnx(params)
    if sld_values is None: _, sld_values = get_sld_profile_refnx(params)

    structure = build_structure_from_sld_prof(depth_values, sld_values)
    reflectivity_values = refnx_reflectivity(structure,qvals)

    return depth_values, sld_values, qvals, reflectivity_values


def get_default_par_values_refnx():
    default_values = {
        "thickness": np.array([0, 1, 1, 1, 0]),
        "sld": np.array([0, 5, 7, 3, 1]),
        "scan_size": 1024,
        "q_min": 0.0,
        "q_max": 1.0,
        "total_sample_depth": 64,
        "n_depth_points": 1024,
        "roughness": 2.5
    }
    return default_values

def show_refnx_inconsistency(filename=None):

    for _n_ in range(5,12):
        n_depth_points = 2**_n_

        params = get_default_par_values_refnx()
        params["n_depth_points"] = n_depth_points

        d,s,q,r = sld_profile_and_reflectivity(params)
        d2,s2,q2,r2 = sld_profile_and_reflectivity_from_sld_profile(params, d, s)

        params["roughness"] = 0

        d3,s3,q3,r3 = sld_profile_and_reflectivity(params)
        d4,s4,q4,r4 = sld_profile_and_reflectivity_from_sld_profile(params, d3, s3)

        ax = plt.figure(figsize = (18,6))

        ax.add_subplot(121)
        plt.title("SLD profile")
        plt.plot(d,s, label="Layers with roughness")
        plt.plot(d2,s2+1, label="Continous SLD profile 1")
        plt.plot(d3,s3+2, label="Layers with no roughness")
        plt.plot(d4,s4+3, label="Continuous SLD profile 2")
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

if __name__ == "__main__":
    show_refnx_inconsistency("./RefnxInconsistency.png")
