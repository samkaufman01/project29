'''
Created on Jan 27, 2017

@author: ethan
'''
from pymatgen.matproj.rest import MPRester, MPRestError
import numpy as np

class NoMaterialAtIDError(Exception):
    pass

API_Key='fdhGQJEPevF5MsiD'
mp=MPRester(API_Key)

MaterialIDs=np.genfromtxt('ALL-IDLIST.txt',dtype=str)
MPIDs=['mp-'+a for a in MaterialIDs]

data = mp.query(criteria={"task_id": {"$in": MPIDs}}, properties=['task_id','elasticity','structure'])
for i in range(0,10):#len(data)):
    ID=data[i]['task_id']
    el=data[i]['elasticity']
    alatt=data[i]['structure'].lattice.a #a lattice constant
    print(ID,el,alatt)

