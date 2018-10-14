
#include <boost/python.hpp>



char const* greet()
{
   return "hello, world";
}


//'s important that the library file is named like you declare the module here:
BOOST_PYTHON_MODULE(cpp_python_test)
{
    using namespace boost::python;
    def("greet", greet);
}

