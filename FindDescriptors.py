'''
Created on Dec 22, 2016

@author: ethan
'''
#Finds all descriptors for used material ids in materials project

import math
import json
import os.path
import ast
from pymatgen.core import periodic_table
from pymatgen.matproj.rest import MPRester, MPRestError
from pymatgen.analysis import bond_valence
import numpy as np
import itertools
import re, itertools
import matplotlib.pyplot as plt
from mendeleev import element


NobleConfigs=[0,2,10,18,36,54,86] #number of electrons in noble metal configuration

'''
determine whether the value belong to the float range.
'''
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
# output the array list of separated test name from the data
def NameAnalyzer(NameToTest):
    NameToTest= [ y for y in list(itertools.chain(*[re.split(r'\"(.*)\"', x) 
        for x in re.split(r'\((.*)\)', NameToTest)])) 
        if y != ''] #this splits at parenthases
    
    ParenSeparatedName=[]
    for j in range(0,len(NameToTest)):#This will check if first character is a number, due to parenthases
        firstChar=NameToTest[j][0]
        if firstChar.isdigit()==True:
            ParenSeparatedName.extend([a for a in re.split(r'([A-Z][a-z]*\d*)', NameToTest[j]) if a])
        else:
            #print NameToTest[j]
            ParenSeparatedName.append(NameToTest[j])
   
_____SeparatedName=[]
    for j in range(0,len(ParenSeparatedName)):
        TempSegment=[a for a in re.split(r'([A-Z][a-z]*)', ParenSeparatedName[j]) if a]
        multiplier=1.0
        if isfloat(TempSegment[0])==False:
            if j<len(ParenSeparatedName)-1 and isfloat(ParenSeparatedName[j+1])==True:
                multiplier=float(ParenSeparatedName[j+1])
            for k in range(0,len(TempSegment)):
                #print(TempSegment[k],TempSegment[k].isnumeric())
                if isfloat(TempSegment[k])==False:
                    if k<len(TempSegment)-1 and isfloat(TempSegment[k+1])==True:
                        SeparatedName.append(TempSegment[k])
                        SeparatedName.append(str(float(TempSegment[k+1])*multiplier))
                    elif k<len(TempSegment)-1 and isfloat(TempSegment[k+1])==False:
                        SeparatedName.append(TempSegment[k])
                        SeparatedName.append(str(multiplier))
                    elif k==len(TempSegment)-1 and isfloat(TempSegment[k])==False:
                        SeparatedName.append(TempSegment[k])
                        SeparatedName.append(str(multiplier))
    return(SeparatedName)

# output the electronegativity difference
# output the average electronegativity cost ?? I may be a little bit confused about this
def CostFunction(CostName):
    ENegC=[]
    ENegC2=0
    for k in range(0,len(CostName),2):
        OrderedElementIndex=OrderedElements.index(CostName[k])
        ENegC.append(Electronegativity[OrderedElementIndex])
        ENegC2+=Electronegativity[OrderedElementIndex]*float(CostName[k+1])
    ENegC2=ENegC2/totalAtoms
    ENegDif=max(ENegC)-min(ENegC)  
    
    return ENegDif,ENegC2

def AtomisticProperties(NameVec,NumAt,RelWeight):
    #properties=[atomic_volume,dipole_polarizability,ionenergy,atomic_radius,thermal_conductivity,vdw_radius]
    Vol=[element(a).atomic_volume for a in NameVec[0::2]]
    TotVol=sum([a*b for a,b in zip(Vol,RelWeight)])#*NSites
    VolFrac=TotVol/float(Volume)
    
    Polar=[element(a).dipole_polarizability for a in NameVec[0::2]]
    TotPolar=sum([a*b for a,b in zip(Polar,RelWeight)])#*NSites
    AvgPolar=np.mean([a*b for a,b in zip(Polar,RelWeight)])
    DiffPolar=max(Polar)-min(Polar)

    ElecAffin=[element(a).electron_affinity for a in NameVec[0::2]]
    TotAffin=sum([a*b for a,b in zip(ElecAffin,RelWeight)])#*NSites
    AvgAffin=np.mean([a*b for a,b in zip(ElecAffin,RelWeight)])
    DiffAffin=max(ElecAffin)-min(ElecAffin)

    IonEn=[]
    for k in range(0,len(NameVec),2):
        OrderedElementIndex=IonElements.index(NameVec[k])
        IonEn.append(IonizationEns[OrderedElementIndex])
    TotIonEn=sum([a*b for a,b in zip(IonEn,RelWeight)])#*NSites
    AvgIonEn=np.mean([a*b for a,b in zip(IonEn,RelWeight)])
    DiffIonEn=max(IonEn)-min(IonEn)
    
    Mass=[element(a).mass for a in NameVec[0::2]]
    TotMass=sum([a*b for a,b in zip(Mass,RelWeight)])#*NSites
    AvgMass=np.mean([a*b for a,b in zip(Mass,RelWeight)])
    RedMass=(sum([b/a for a,b in zip(Mass,RelWeight)]))**-1 #reduced mass
    DiffMass=max(Mass)-min(Mass)
    
    VDWRad=[element(a).vdw_radius for a in NameVec[0::2]]
    TotVDW=sum([a*b for a,b in zip(VDWRad,RelWeight)])
    AvgVDW=np.mean([a*b for a,b in zip(VDWRad,RelWeight)])
    DiffVDW=max(VDWRad)-min(VDWRad)
    
    for values in [TotVol,TotPolar,AvgPolar,DiffPolar,TotAffin,AvgAffin
                   ,DiffAffin,TotIonEn,AvgIonEn,DiffIonEn,TotMass,AvgMass,DiffMass,RedMass,TotVDW,AvgVDW,DiffVDW]:
        fM.write("%s," % values)
        
    return


ENegData=np.genfromtxt('Electronegativity.txt',dtype=None)
Elements=[]
OrderedElements=[]
Electronegativity=[]
for i in range(0,len(ENegData)):
    Elements.append(ENegData[i][0].decode("utf-8"))
    OrderedElements.append(ENegData[i][0].decode("utf-8"))
    Electronegativity.append(ENegData[i][1])
IonizationData=np.genfromtxt('FirstIonizationEnergies.txt',dtype=None)
IonElements=[]
IonizationEns=[]
for i in range(0,len(IonizationData)):
    IonElements.append(IonizationData[i][0].decode("utf-8"))
    IonizationEns.append(IonizationData[i][1])

def CountElectrons(mpid):
    oxidizedstates = mp.query('mp-'+mpid, ["bv_structure"])[0]['bv_structure'].get_primitive_structure()
    AtomList=[spec for spec in oxidizedstates.species]
    AtomSet=set(AtomList)
    Occurances=[AtomList.count(Atom) for Atom in AtomSet]
    #print(AtomSet,Occurances)
    
    nsTot=0 #Total number of s,p,d,f electrons in unit cell
    npTot=0
    ndTot=0
    nfTot=0
    for i in range(0,len(AtomSet)):
        specie=list(AtomSet)[i] #specify the ionic species
        multiplicity=Occurances[i] #corresponding multiplicity in formula
        try: #check if provides oxidation state or neutral
            num_electrons = int(specie.Z-specie.oxi_state)
        except AttributeError:
            num_electrons = specie.Z
        #print(specie,num_electrons)
        ns=0
        np=0
        nd=0
        nf=0
        if num_electrons in NobleConfigs and specie.oxi_state>0: #check if species has noble gas due to e loss
            pass
        else:
            orbitals = periodic_table.get_el_sp(int(num_electrons)).full_electronic_structure
            #print(orbitals)
            max_n=orbitals[-1][0] #finds maximum principle quantum number, module orders by n so this is ok to do
            s_placehold=[item for item in orbitals if 's' in item]
            ns=s_placehold[-1][2] #outermost s orbital, always in valence
            p_placehold=[item for item in orbitals if 'p' in item]
            if p_placehold: #check if there are any p electrons
                if p_placehold[-1][0]==max_n: #checks if outermost filled p orbital is valence or core
                    np=p_placehold[-1][2]
                else:
                    np=0
                
                d_placehold=[item for item in orbitals if 'd' in item]
                if d_placehold: #check if any d electrons
                    if d_placehold[-1][0]==max_n-1: #checks if outermost filled d orbital is valence or core
                        nd=d_placehold[-1][2]
                    elif d_placehold[-1][0]==max_n: #this is an exception for materials in which outermost s electrons are moved to the d orbital
                        nd=d_placehold[-1][2]
                        ns=0
                        np=0
                    else:
                        nd=0
                    
                    f_placehold=[item for item in orbitals if 'f' in item]
                    if f_placehold: #check if any d electrons
                        nf=f_placehold[-1][2]                
        #print(ns,np,nd,nf)
        nsTot+=ns*multiplicity
        npTot+=np*multiplicity
        ndTot+=nd*multiplicity
        nfTot+=nf*multiplicity
    return nsTot, npTot, ndTot, nfTot
# determine the material strucutre according to the group number
def SymmGroupFinder(GroupNum):
    if 1<=GroupNum<=2:
        Group='Triclinic'
    elif 3<=GroupNum<=15:
        Group='Monoclinic'
    elif 16<=GroupNum<=74:
        Group='Orthorhombic'
    elif 75<=GroupNum<=142:
        Group='Tetragonal'
    elif 143<=GroupNum<=167:
        Group='Trigonal'
    elif 168<=GroupNum<=194:
        Group='Hexagonal'
    elif 195<=GroupNum<=230:
        Group='Cubic'
    else:
        Group='Amorphous'
    return Group


class NoMaterialAtIDError(Exception):
    pass
    #Don't need anything here, just define it as an exception

def DatabaseGap(IDNum): #Charge Neutrality from materialsproject
    Name='mp-'+IDNum
    try:
        banddata = mp.get_data(Name,prop="bandstructure")
        
        #print(IDNum,banddata)
        if len(banddata) == 0:
            raise NoMaterialAtIDError       
        bandstruct = banddata[0]["bandstructure"]
        #print(bandstruct.bands)
        #print(list(bandstruct.bands.keys()))
        keys=list(bandstruct.bands.keys())
        #print(bandstruct.bands)
        #print(bandstruct.bands[keys[0]])
        EG=[]
        for qq in range(0,len(keys)):
            for i in range(bandstruct.nb_bands):
                if (    np.mean(bandstruct.bands[keys[qq]][i+1]) > bandstruct.efermi
                    and abs(max(bandstruct.bands[keys[qq]][i+1])-max(bandstruct.bands[keys[qq]][i]))>0.001
                    ): #check that i+1 band is on average above Ef and that the maxima of i+1 and i do not coincide
                    cbbottom = i+1
                    vbtop = i
                    break #bands found first time condition is satisfied
            ConBanMinUp=min(bandstruct.bands[keys[qq]][cbbottom])
            ValBanMaxUp=max(bandstruct.bands[keys[qq]][vbtop])
            EGUp=ConBanMinUp-ValBanMaxUp
            EG.append(EGUp)

        #plt.plot(bandstruct.bands[Spin.up][vbtop])
        #plt.plot(bandstruct.bands[Spin.up][cbbottom])
        #plt.plot(bandstruct.bands[Spin.up][vbtop-1])
        #plt.show()
        
    except NoMaterialAtIDError:
        return
    except MPRestError:
        return
    #print(EG)
    return(EG)



IDList=np.genfromtxt('idlist.txt',dtype=None)

API_Key='HbQtI8zsBEk7R91T'
mp=MPRester(API_Key)

MetalThreshold=-0.3 #threshold gap to be considered a metal and retained



fM=open('MetalDescriptors7.csv','w')

fM.write('MPID'+','+'Formula'+','+'EnergyPerAtom'+','+'Volume'+','+'Density'+','+'NSites'+','+'SpaceGroup'+','+'Symmetry'+','+'FormationEnergy'+','
         +'EHull'+','+'a'+','+'b'+','+'c'+','+'s'+','+'p'+','+'d'+','+'f'+','+'ENegDiff'+','+'ENegAvg'+','
         +'TotVol'+','+'TotPolar'+','+'AvgPolar'+','+'DiffPolar'+','+'TotAffin'+','+'AvgAffin'+','+'DiffAffin'+','
         +'TotIonEn'+','+'AvgIonEn'+','+'DiffIonEn'+','+'TotMass'+','+'AvgMass'+','+'DiffMass'+','+'RedMass'+','
         +'TotVDW'+','+'AvgVDW'+','+'DiffVDW'+','
         +'Gap(VASP)'+','+'IsMetal'+'\n')

for MPID in IDList:
    if 861483<MPID<1000000000:
        try:
            data = mp.query(criteria={"task_id": 'mp-'+str(MPID)}, properties=["pretty_formula",'volume','spacegroup.number','formation_energy_per_atom',
                                            'structure','density','e_above_hull','nsites','energy_per_atom'])
            formula=str(data[0]['pretty_formula'])
            print(MPID,formula)
            Volume=str.format("{0:.6f}",data[0]['volume'])
            symmGroup=str(data[0]['spacegroup.number'])
            symmetry=SymmGroupFinder(int(symmGroup))
            formE=str.format("{0:.6f}",data[0]['formation_energy_per_atom'])
            lattA=str.format("{0:.6f}",data[0]['structure'].lattice.a)
            lattB=str.format("{0:.6f}",data[0]['structure'].lattice.b)
            lattC=str.format("{0:.6f}",data[0]['structure'].lattice.c)
            density=str.format("{0:.6f}",data[0]['density'])
            eHull=str.format("{0:.6f}",data[0]['e_above_hull'])
            NSites=str(int(data[0]['nsites']))
            formE=str.format("{0:.6f}",data[0]['formation_energy_per_atom'])
            energy=str.format("{0:.6f}",data[0]['energy_per_atom'])
            s,p,d,f=CountElectrons(str(MPID))
            MaterialGap=DatabaseGap(str(MPID))
            #print(MaterialGap)
            if MaterialGap is None: #no band structure
                continue #next material if current material does not have band structure
            elif all(q>=MetalThreshold for q in MaterialGap): #non metal
                IsMetal='NonMetal'
                Gap=min(MaterialGap) #smallest gap value if spin-dependent bs
            elif any(q<=MetalThreshold for q in MaterialGap): #metal
                IsMetal='Metal'
                Gap=0
            for item in[str(MPID),formula,energy,Volume,density,NSites,symmGroup,symmetry,formE,eHull,lattA,lattB,lattC,s,p,d,f]:
                fM.write("%s," % item)
            #find atomsitic descriptors
            SeparatedName=NameAnalyzer(formula)
            #print(SeparatedName)
            AtomicWeights=[float(k) for k in SeparatedName[1::2]] #number weights in formula
            totalAtoms=sum([float(k) for k in SeparatedName[1::2]]) #total number of atoms in formula unit
            ElectronegativityDifference,ElectronegativityAverage=CostFunction(SeparatedName)
            for item in[ElectronegativityDifference,ElectronegativityAverage]:
                fM.write("%s," % item)
            AtomisticProperties(SeparatedName,totalAtoms,AtomicWeights)
            
            #Write whether metal and gap if not
            for item in[Gap,IsMetal]:
                fM.write("%s," % item)
                   
            fM.write('\n')
    
        except:
            pass
        
fM.close()
