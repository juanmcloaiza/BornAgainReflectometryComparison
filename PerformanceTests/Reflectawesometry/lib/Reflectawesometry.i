%module Reflectawesometry
%{
#include "Reflectawesometry.h"
%}

%include "std_vector.i"

// Instantiate templates used by Reflectawesometry
namespace std {
   %template(Vector)  vector <double>;
}

// Include the header file with above prototypes
%include "Reflectawesometry.h"
