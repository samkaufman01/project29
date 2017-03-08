from pymatgen import Composition, Element
from numpy import zeros, mean
from sklearn import linear_model, cross_validation, metrics, ensemble


# Training file containing band gaps extracted from Materials Project
# created in previous blog post and linked here
trainFile = open("bandgapDFT.csv","r").readlines()

# Input: pymatgen Composition object
# Output: length-100 vector representing any chemical formula

def naiveVectorize(composition):
    vector = zeros((MAX_Z))
    for element in composition:
        fraction = composition.get_atomic_fraction(element)
        vector[element.Z - 1] = fraction
    return(vector)

# Extract materials and band gaps into lists, and construct naive feature set
materials = []
bandgaps = []
naiveFeatures = []

MAX_Z = 100 # maximum length of vector to hold naive feature set

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    materials.append(material) #store chemical formulas
    naiveFeatures.append(naiveVectorize(material)) #create features from chemical formula
    bandgaps.append(float(split[1])) #store numerical values of band gaps

# Establish baseline accuracy by "guessing the average" of the band gap set
# A good model should never do worse.
baselineError = mean(abs(mean(bandgaps) - bandgaps))
print("The MAE of always guessing the average band gap is: " + str(round(baselineError, 3)) + " eV")


# Train linear ridge regression model using naive feature set
#alpha is a tuning parameter affecting how regression deals with collinear inputs
linear = linear_model.Ridge(alpha = 0.5)  

cv = cross_validation.ShuffleSplit(len(bandgaps),\
            n_iter=10, test_size=0.1, random_state=0)

scores = cross_validation.cross_val_score(linear, naiveFeatures,\
            bandgaps, cv=cv, scoring='neg_mean_absolute_error')

print("The MAE of the linear ridge regression band gap model using the naive feature set is: "\
    + str(round(abs(mean(scores)), 3)) + " eV")

# Let's see which features are most important for the linear model

print("Below are the fitted linear ridge regression coefficients for each feature (i.e., element) in our naive feature set")

linear.fit(naiveFeatures, bandgaps) # fit to the whole data set; we're not doing CV here

print("element: coefficient")

for i in range(MAX_Z):
    element = Element.from_Z(i + 1)
    print(element.symbol + ': ' + str(linear.coef_[i]))
