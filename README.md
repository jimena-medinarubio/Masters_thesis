# Masters_thesis
This private repository contains the models outputs and codes used for the analysis in my Master's thesis conducted at the Physics Department of Imperial College London. The thesis focuses on investigating the transport pathways and dynamics of ocean currents in the coastal region near the island of Curaçao (South Caribbean) over the month of March 2022. 
The following information can be found in this repository:

**DELFT3D hydrodynamic model of the ocean currents around the island of Curaçao**
  - Code for extraction of wind datasets from ECMWF
  - Attribute files necessary to run the model (grid, bathymetry, open boundaries, boundary conditions, wind forcing, MDF file). See https://oss.deltares.nl/web/delft3d/manuals for more information about each of them.
  - Animation of the velocity field at each timestep (snapshots included in thesis).
  - *Model output in netCDF file too heavy (13GB). Time series of velocity components and surface elevation at each timestep.*
  
**Parcels simulation of Lagrangian trajectories**
  - Code needed to run the model with the velocity field as input.
  - Model output of the Lagrangian trajectories compressed from original size (3.5GB).
  - Animation showing the particles advection over time.
  
**Kernel Density Estimation (KDE) analysis**
Code to calculate the KDE from Lagrangian trajectories
  
**Supervised Machine Learning method to classify trajectories (KNN)**
  - Given a domain parition, code to classify trajectories and obtain their itinerary at each timestep.
  Using the itineraries of the particles, the following metrics are calculated:
  
  **a) Exposure time**
  - Code to calculate the exposure time of particles in each region of the partitioned domain using their itineraries.

  **b)Lagrangian Flow Network**
  - Code to compute the transition matrix of the trajectories over the partitioned domain and obtain a graph where nodes represent each   region and the edges, the probability of transport between them.
  - Code to calculate the *betweenness centrality* of each region of the partitioned domain
  - *Infomap* algorithm used to find distinct commnities within the network.
  
