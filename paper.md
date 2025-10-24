---
title: 'pyRES: A Python Package for Time-Dependent Energy Analysis and and Economic Assessment of Renewable Energy Communities'
tags:
  - Python
  - Renewable energy communities
  - Solar panels
  - Battery energy storage system
  - economics
authors:
  - name: Isabella Pizzuti
    orcid: 0009-0005-6881-7396
    equal-contrib: true
    affiliation: 1 
    corresponding: true
  - name: Giovanni Delibra
    equal-contrib: false
    affiliation: 1
affiliations:
 - name: Sapienza University of Rome,Department of Mechanical and Aerospace Engineering,  Italy
   index:  1

date: 24 October 2025
bibliography: paper.bib

---

# Summary
Energy communities represent a new paradigm for producing and sharing renewable energy, promoting a bottom-up energy 
transition. These initiatives allow citizens, businesses, and local authorities to collaborate in generating, storing, 
and consuming clean energy, often integrating innovative energy solutions. Among the most commonly implemented systems 
are small-scale generation units and storage solutions, but their optimal integration depends on site-specific 
conditions, energy demand profiles, and local policies. Effective planning demands tools capable of 
simulating time-dependent interactions among distributed generation, storage, and consumption, while also accounting for
local constraints. Framing the planning of energy communities as high-resolution energy flow modeling provides a powerful 
tool to support decision-makers based on a detailed reconstruction of energy production, consumption, and exchange dynamics. 
The tool allows to evaluate how constraints —as the availability of rooftops or land, the cost of storage systems, 
and regulations on energy sharing — affect energy and economic performance. By providing a quantitative basis for planning, 
these models help stakeholders design energy communities.


pyRES is a framework to support the time-dependent energy analysis and assess 
the economic performance of renewable energy communities (RECs) composed of consumers and prosumers equipped with 
photovoltaic systems (PV) and battery energy storage systems (BESS). pyRES’s key features include photovoltaic simulation,
using the PVlib library [@anderson2023pvlib] and meteorological data; 
BESS simulation, modelling charge and discharge cycles based on PV generation and user demand; Prosumer and REC modelling, 
managing self-consumption, surplus generation, energy sharing, grid exchanges, and evaluating economic performance at the 
individual and community level. pyRES is based on an object-oriented approach, where the superclass System defines common 
attributes and methods for all energy systems. Specific technologies are implemented as subclasses that inherit from the superclass 
System and extend its functionality.This structure allows for easy extension: new energy systems can be added by creating 
additional subclasses, each implementing their specific models, permitting analyses ranging from single urban districts to countries.

A pyRES model consists of a collection of YAML and CSV files that define technologies,
locations, consumers, prosumers, and community configurations. The preprocessing stage integrates this information 
by identifying the system’s location, retrieving corresponding meteorological data via the PVlib library, and 
organizing datasets with numpy and pandas. Subsequently, the model is instantiated through Python class objects that 
represent system components. The computational phase executes core methods for energy 
yield assessment and for the evaluation of economic indicators. Finally, the results are processed 
and exported using pandas and Matplotlib, generating CSV files for further processing and PNG 
visualizations as shown in \autoref{fig:example}.

pyRES is developed in the open on GitHub ([@fidgit]) and each
release is archived on Zenodo ([![DOI](https://zenodo.org/badge/1082548694.svg)](https://doi.org/10.5281/zenodo.17435948)).

# Statement of need

pyRES was designed to be used by researchers, students in energy engineering courses and professionals working in RECs 
design. It has already been applied in a number of scientific publications exploring various aspects of RECs and used 
in graduated courses on energy modelling. Among the main studies, pyRES supported the analysis of community energy
models on small Italian islands , evaluating the synergy between self-generated electricity and clean water production [@corsini2023challenges]. 
Additionally, pyRES was used to investigate the integration of photovoltaic panels with cogeneration and biomass systems 
in a community located in a mountainous area [@pizzuti2024integration], and to assess the impact of heat pumps and electrification policies on 
community energy performance [@pizzuti2025design]. Ongoing research projects using pyRES include the forecasting of demand curves and the 
development of hydrogen modules to extend its functionality by leveraging its modular structure.

![Example summary visualisation of annual energy perfomance 
from a REC model, created with the matplotlib 
in pyRES.\label{fig:example}](docs/ex_results.png)



# Acknowledgements

This project is supported by the Ministry of University and Research
(MUR) as part of the European Union program NextGenerationEU,
PNRR-M4C2 - ECS_00000024 “Rome Technopole” in Flagship Project 1
“Clean energy transition and circular economy: materials, bioenergy,
green chemistry, green hydrogen and alternative fuels, renewable energy communities, isolated energy systems and smaller islands“, under
the program PE2. Scenari energetici del futuro. PE0000021″ NEST −
Network 4 Energy Sustainable Transition“ and by Regione Lazio under
agreement with Department of Mechanical and Aerospace Engineering
of Sapienza University of Rome on research and support to Renewable
Energy Communities.

# References
