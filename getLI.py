import pandas as pd
import numpy as np
from scipy import stats
big_data = pd.read_csv("AllDescriptors.csv", header=0, sep = ',')
formulars = big_data["Formula"].tolist()
all_elements = "Li Be B C N O F Na Mg Al Si P S Cl K Ca Cs Tl V Cr Mn Fe Co Nl Cu Zn Ga Ge As Se Br Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi"
print (formulars)
all_elements = all_elements.split()
print (all_elements)
Li_Bi_list = []
for Formula in formulars:
	upper_num = 0
	flag = 0
	for c in Formula:
		if c in all_elements:
			flag+=1
		if c.isupper():
			upper_num+=1
	if upper_num>=3 or upper_num<2 or flag !=2 : 
		continue
	Li_Bi_list.append(Formula)
Li_Bi_list.extend(all_elements)

print (Li_Bi_list)
thefile = open('Li_Bi_list_formula', 'w')
for item in Li_Bi_list:
  thefile.write("%s\n" % item)
res = big_data
res = res[res['Formula'].isin(Li_Bi_list)]
res = res[['MPID','Formula','a','b','c']]
#print res
res.to_csv("Li_Bi_list_2")
