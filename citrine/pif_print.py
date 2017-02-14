from pypif import pif
from pypif.obj import *


chemical_system = ChemicalSystem()
chemical_system.chemical_formula = 'MgO2'

band_gap = Property()
band_gap.name = 'Band gap'
band_gap.scalars = 7.8
band_gap.units = 'eV'

chemical_system.properties = band_gap

print(pif.dumps(chemical_system, indent=4))
