<table><tr><td><img src='./fig6_F2SEvolution.pdf'></td></tr></table>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworld-community/Knight-Himalayas/master)

About
-----
Welcome! This repo contains the python setup file for the Himalayan reconstruction convergence models. The model can be run online through the binder link or downloaded and used within a Underworld docker container.

Files
-----
File | Purpose
--- | ---
`HimalayasReconstructionModel.py` | Python script to run the model. Can be modified to recreate any case presented in the manuscript. 
`fig6_F2SEvolution.pdf` | Evolution of the fast-to-slow model, which recreates similar structures that are seen in the Himalayas.

Tests
-----
**_Please specify how your repository is tested for correctness._**
**_Tests are not required for `laboratory` tagged repositories, although still encouraged._**
**_All other repositories must include a test._**


Parallel Safe
-------------
**_Please specify if your model will operate in parallel, and any caveats._**

Yes, test result should be obtained in both serial and parallel operation.

Check-list
----------
- [Y] (Required) Have you replaced the above sections with your own content? 
- [Y] (Required) Have you updated the Dockerfile to point to your required UW/UWG version? 
- [Y] (Required) Have you included a working Binder badge/link so people can easily run your model?
                 You probably only need to replace `template-project` with your repo name. 
- [Y] (Optional) Have you included an appropriate image for your model? 
