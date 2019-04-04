#ifndef REFLECTAWESOMETRY_H
#define REFLECTAWESOMETRY_H

//#include <QCoreApplication>
//#include <QFile>
//#include <QTextStream>
//#include <QTimer>
//#include <cmath>
#include <fstream>
#include <complex>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

#define PI 3.14159265358979323846
// TINY is required to make sure a complex sqrt takes the correct branch
// if you choose too small a number for tiny then the complex square root
// takes a lot longer.
#define TINY 1e-30

inline void matmul(std::complex<double> a[2][2], std::complex<double> b[2][2],
                   std::complex<double> c[2][2])
{
    c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
    c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
    c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
    c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
}

template <typename T> std::vector<T> linspace(T a, T b, size_t N)
{
    T h = (b - a) / static_cast<T>(N - 1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}

std::vector<double> calculate_reflectometry(std::vector<double> qvals,
                                                 std::vector<double> thicknesses,
                                                 std::vector<double> slds,
                                                 std::vector<double> roughnesses)
{
    using namespace std;
    auto npoints = qvals.size();
    std::vector<double> reflectometry(npoints);

    double scale, bkg;
    double num = 0, den = 0, answer = 0;

    complex<double> super;
    complex<double> sub;
    complex<double> oneC = complex<double>(1., 0.);
    complex<double> MRtotal[2][2];
    complex<double> MI[2][2];
    complex<double> temp2[2][2];
    complex<double> qq2;
    complex<double>* SLD = nullptr;
    complex<double>* thickness = nullptr;
    double* rough_sqr = nullptr;

    size_t nlayers = size_t(thicknesses.size());

    scale = 1.0;
    bkg = 0.0;
    sub = complex<double>(2.07, 0.0 + TINY);
    super = complex<double>(0.00, 0.0);

    try {
        SLD = new complex<double>[nlayers + 2];
        thickness = new complex<double>[nlayers];
        rough_sqr = new double[nlayers + 1];
    } catch (...) {
        goto done;
    }
    // fill out all the SLD's for all the layers
    for (int ii = 1; ii < int(nlayers + 1); ii += 1) {
        SLD[ii] = 4e-6 * PI * (complex<double>(slds[size_t(ii - 1)], 0.0) + TINY) - super;
        thickness[ii - 1] = complex<double>(0, thicknesses[size_t(ii - 1)]);
        rough_sqr[ii - 1] = -2 * roughnesses[size_t(ii - 1)] * roughnesses[size_t(ii - 1)];
    }

    SLD[0] = complex<double>(0, 0);
    SLD[nlayers + 1] = 4e-6 * PI * (sub - super);
    rough_sqr[nlayers] = 0.0;

    for (size_t j = 0; j < npoints; j++) {
        complex<double> beta, rj;
        complex<double> kn, kn_next;

        qq2 = complex<double>(qvals[j] * qvals[j] / 4, 0);

        // now calculate reflectivities and wavevectors
        kn = std::sqrt(qq2);
        for (int ii = 0; ii < int(nlayers + 1); ii++) {
            // wavevector in the layer
            kn_next = std::sqrt(qq2 - SLD[ii + 1]);

            // reflectance of the interface
            rj = (kn - kn_next) / (kn + kn_next) * std::exp(kn * kn_next * rough_sqr[ii]);

            if (!ii) {
                // characteristic matrix for first interface
                MRtotal[0][0] = oneC;
                MRtotal[0][1] = rj;
                MRtotal[1][1] = oneC;
                MRtotal[1][0] = rj;
            } else {
                // work out the beta for the layer
                beta = std::exp(kn * thickness[ii - 1]);
                // this is the characteristic matrix of a layer
                MI[0][0] = beta;
                MI[0][1] = rj * beta;
                MI[1][1] = oneC / beta;
                MI[1][0] = rj * MI[1][1];

                // multiply MRtotal, MI to get the updated total matrix.
                memcpy(temp2, MRtotal, sizeof(MRtotal));
                matmul(temp2, MI, MRtotal);
            }
            kn = kn_next;
        }

        num = std::norm(MRtotal[1][0]);
        den = std::norm(MRtotal[0][0]);
        answer = (num / den);
        answer = (answer * scale) + bkg;

        reflectometry[j] = answer;
    }

done:
    if (SLD)
        delete[] SLD;
    if (thickness)
        delete[] thickness;
    if (rough_sqr)
        delete[] rough_sqr;

    return reflectometry;
}

void write_to_file(std::string filename, std::vector<double> qvals, std::vector<double> reflectometry)
{
    std::ofstream myfile;
    myfile.open(filename);
    //TODO: raise exception if qvals and reflectometry have different sizes
    size_t nq = qvals.size();
    for (size_t i = 0; i < nq; i++)
    {
        myfile << qvals[i] << " " << reflectometry[i] << std::endl;
    }
    myfile.close();
}

struct layer {
    double Thickness;
    double SLD;
    double roughness;
};

std::vector<double> run_my_model(const size_t number_of_layers, const std::vector<double> &qvalues)
{
    layer Ti;
    Ti.Thickness = 30;
    Ti.SLD = -2.0;
    Ti.roughness = 0.0;

    layer Ni;
    Ni.Thickness = 70;
    Ni.SLD = 10.0;
    Ni.roughness = 0.0;

    std::vector<double> thicknesses(2*number_of_layers);
    std::vector<double> slds(2*number_of_layers);
    std::vector<double> roughnesses(2*number_of_layers);

    for (size_t i = 0; i < 2*number_of_layers; i+=2) {
        thicknesses[i] = Ti.Thickness;
        slds[i] = Ti.SLD;
        roughnesses[i] = Ti.roughness;

        thicknesses[i+1] = Ni.Thickness;
        slds[i+1] = Ni.SLD;
        roughnesses[i+1] = Ni.roughness;
    }
/*
    std::cout << "Nlayers : " << number_of_layers << std::endl;
    for (auto t : thicknesses) {
        std::cout << t << " ";
    }
    std::cout << std::endl;

    for (auto t : slds) {
        std::cout << t << " ";
    }
    std::cout << std::endl;

    for (auto t : roughnesses) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
*/
    std::vector<double> reflectometry =
        calculate_reflectometry(qvalues, thicknesses, slds, roughnesses);
    return reflectometry;
}

#endif // REFLECTAWESOMETRY_H
