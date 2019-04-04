#include "Reflectawesometry.h"

int main(int argc, char* argv[])
{
    std::cout << "Hello World!" << std::endl;

    size_t numberOfLayers;
    if(argc < 2){
        numberOfLayers = 1;
    }else{
     numberOfLayers = size_t(atoi(argv[1]));
    }

    double qmin = 0.0;
    double qmax = 0.5;
    size_t npoints = 1025;
    std::vector<double> qvals = linspace(qmin,qmax,npoints);
    auto reflectometry = run_my_model(numberOfLayers,qvals);
    write_to_file("ReflectawesometryOutput.txt", qvals, reflectometry);
    std::cout << "Good bye World!" << std::endl;

    return 0;
}

