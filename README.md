# J0437-Scintillation-arcs

Repository containing data and code used for the analysis of scintillation arcs in MeerKAT observations of J0437-4715 (Reardon et al., submitted, Nature Astronomy).

The data provided are the normalised secondary spectra (Hough transform) forms, to reduce file size. The large raw dynamic spectra will be made available for download elsewhere, and a link provided here.

The codes were used to identify individual arcs across multiple epochs (select_and_tune.py), fit their curvatures (fit_orbit_campaign.py), and produce a screen velocity model (model_curvatures.py). The outputs from these analyses are included in this repository.

Note that the codes and outputs are provided for transparency and reproducibility only and these interactive tools are not intended for use with other data sets (although they should work if the data is reduced to the same format). For any further queries, please contact Daniel Reardon at: dreardon@swin.edu.au

The resulting screen models are shown for each observation in the animation below. Each frame shows the secondary spectrum from a 12 hour observation on a single day, for the six consecutive days of the observing campaign. Models are overlaid on one half of the spectrum only, to enable visual comparison to the data.
![](https://github.com/danielreardon/J0437-Scintillation-arcs/blob/main/animations/model_animation.gif?raw=true)
