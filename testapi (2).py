from citrination_client import *
from operator import itemgetter
import numpy as np

compounds = np.genfromtxt('Li_Bi_list_2',delimiter=',',skip_header=1,usecols=(1,2,3,4,5))
effectiveMasses = np.genfromtxt('OpticalProperties.txt',skip_header=1,usecols=(0,2))

compounds = np.append(compounds,np.zeros([len(compounds),1]),1)
compoundIdx = 0
for i in range(len(effectiveMasses)):
    if effectiveMasses[i][0] == compounds[compoundIdx][0]:
        compounds[compoundIdx][5] = effectiveMasses[i][1]
        compoundIdx += 1

bandInputs = []
bulkInputs = []
for compound in compounds:
    bandInputs.append({'Chemical formula': compound[1]})
    bulkInputs.append({'Element': compound[1]})

client = CitrinationClient('qw02gowcoQpDEC1EcMUcegtt')
resp = client.predict("279", bandInputs)
resp2 = client.predict("149", bulkInputs)
# effectiveMass = np.full(len(bandInputs),0.5)
mobility = np.array([d['Bulk Modulus'][0] for d in resp2['candidates']]) * np.power(effectiveMass,-2.5) * 1.2 * pow(10,-14)
results = []
idx = 0
for d in resp['candidates']:
    if d['Band Gap'][0] + d['Band Gap'][1] >= 1 and d['Band Gap'][0] - d['Band Gap'][1] <= 1.7:
        results.append([d['Chemical formula'][0], d['Band Gap'][0], d['Band Gap'][1], mobility[idx]])
    idx += 1

results = sorted(results, key=itemgetter(3), reverse=True)

print("Done")
