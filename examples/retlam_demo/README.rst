Running the example
-------------------

Example runs with the default parameters by executing the shell script.

Shell script copies or creates some necessary files the first time
it is executed but otherwise it is a convenient wrapper 
for the respective python script.

Configuration of example
------------------------

Simulations can be configured by a configuration file with the fields shown in
specification file. This file is the specification of a valid configuration 
file and in addition documents parameters and provides default values in case 
a field is missing.

Script generates an empty configuration file named <example>_default.cfg, if
it does not exist, which is a valid configuration that uses all default values.
User can edit this file or use a different one but in the latter case should
provide the name to the script.

Try
$ ./run_retlam_demo.sh -h
or
$ python retlam_demo.py -h
for all the available options

Results
-------
Most of the results are stored in HDF5 format with reasonable default
naming to reflect their content.

Information about each generated file follow

Result Files
------------
All file names will be suffixed with a text specified in configuration
which by default is empty, unless it is stated otherwise.

*   grid_dima.h5/grid_dimb.h5: coordinates of screen grid 
    where the image is projected (not subject to a suffix)

*   retina_elev<id>.h5/retina_azim<id>.h5: spherical coordinates of ommatidia
    on retina, id is a numeric identifier of the retina

*   retina_dima<id>.h5/retina_dimb<id>.h5: coordinates of ommatidia
    on screen, id is a numeric identifier of the retina

*   intensities.h5: values of input on screen points


*   retina_input<id>.h5: inputs of retina, id
    is a numeric indentifier of the retina, in case there are more than 1
    (subject to a suffix)

*   retina_output<id>_gpot.h5: graded potential outputs of retina, id
    is a numeric indentifier of the retina, in case there are more than 1
    (subject to a suffix which will be appended before _gpot) 

*   lamina_output<id>_gpot.h5: graded potential outputs of lamina, id
    is a numeric indentifier of the lamina, in case there are more than 1
    (subject to a suffix which will be appended before _gpot)
