# project29

#  Materials Discovery from Data Mining to Design Novel Semiconductor Quantum Dots for Optoelectronics

# MSE 395, Senior design
# Group members: Sam Kaufman, Grace Pakeltis, Bonan Shen, Dawei Shi
# Advisors: Prof. Moonsub Shim, Prof. Andre Schleife
# Department of Materials Science and Engineering

# FILES (Author/Source)
# AllDescriptors.csv (Materials Project)
# MPQuery-2.py (Ethan)
# NeuralNetwork_Structure_EHull.py (Ethan)
# FindDescriptors.py (Ethan)

# PROJECT DESCRIPTION
# Materials science increasingly relies on online collections of materials properties, both from experiment and theory. This is an exciting opportunity, because it allows covering the largest possible space in accelerated materials design. However, the available data is not complete: Oftentimes specific properties, needed to find a well-suited material for a particular application, are missing. Since it can be expensive and cumbersome to measure or accurately compute those for a large number of materials, machine learning is used as an alternative. This approach is familiar, e.g. from social networks, and establishes a mathematical relation between known (“descriptors”) and unknown (“target”) properties, based on training data. The team working on this project will gather training data (e.g. for Young’s modulus, exciton binding energies, plasmon frequencies, or optical-absorption) either from the literature or by running density-functional theory calculations and use this data to train machine-learning models to predict otherwise unknown target properties. Descriptors will be used from large online databases that comprise of as many as 80,000 individual entries (Materials Project). By combining their own results with existing machine-learning models (e.g. for band gaps), the team will be able to identify materials for high-mobility electron- and hole-transport layers, needed for novel semiconductor devices. Training data and trained models willbe published, so they are usable in the future by the international research community, e.g. via the Citrination web site (http://citrination.com).


