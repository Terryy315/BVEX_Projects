This folder contains parts of my research on using a K-band satellite (TDRS-12) for beam calibration of the telescope.

The main code to generate those scan, integrate5.py, is a more developed and reliable version of "integrate2.py" in the software folder.

Each of the example scan shows two very important integrated power plots, up scan (when the telescope moves up), and down scan (when the telescope moves down). Each plot shows the power received from the TDRS-12 satellite as a function of the telescope's elevation.

A curve fit has been performed to the data. The offset between the two Gaussian peaks gives us some important insights into the accuracy and precision of our elevation pointing. It can tell us about the pointing offset and backlash in some mechanical components. The width of the Gaussian (which is one of the CLI outputs of the script) tells the focus of the telescope.
