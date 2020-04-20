# battery-parameter-spaces

NOTE: This repository precedes the repository located [here](https://github.com/chueh-ermon/battery-fast-charging-optimization), associated with [this paper](https://doi.org/10.1038/s41586-020-1994-5):

> Closed-loop optimization of fast-charging protocols for batteries with machine learning  
> Peter Attia\*, Aditya Grover\*, Norman Jin, Kristen Severson, Todor Markov, Yang-Hung Liao, Michael Chen, Bryan Cheong, Nicholas Perkins, Zi Yang, Patrick Herring, Muratahan Aykol, Stephen Harris, Richard Braatz, Stefano Ermon, William Chueh  
> *Nature*, 2020  

\* equal contribution

This repository is currently unmaintained.

Files:
-   closed_loop_oed_hyperparameters.py: hyperparamater optimization code
-   closed_loop_oed.py: OED code
-   contour_lifetimes.py: Generates contour plot of lifetimes
-   contour_points_general.py: Generates contour plot of points (script)
-   contour_points.py: Generates contour plot of points (function)
-   generate_predictions.py: Used in hyperparameter optimization to generate predictions using the simulator
-   policies.py: Creates parameter spaces
-   sim_with_seed.py: Lifetime simulator for hyperparamater optimization
