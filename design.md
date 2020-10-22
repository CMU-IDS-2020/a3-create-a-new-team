IDS Assignment 3


features = {EYE, HAIR, SEX, GSM}



$\Omega:$ a subset of features, e.g. {EYE, HAIR}
$\omega:$ an instantiation of a certain $\Omega$, e.g. {RED, BLOND}

$c:$ a certain character

Goal: investigate the relations between $\Omega$s and (ID or Align)

(or $\Omega$s and (ID and Align)? not sure)

 

#### Charts

##### Bubble plot

- One bubble for each character
- Size: #appearance
- Color: World (DC or Marvel)
- filters
  - Year (slider)
  - GENDER (ticker)
  - Align (ticker)  (alternatively we can use Align for color and World as a filter)


##### Stacked Barplot 
- x: Align or ID
- y: #character
- color: one feature
- filters
    - the distinguishing feature for color (single-choice ticker)
    - Alive (ticker)
    - Year (slider)
    - World (DC/Marvel/Both) (ticker)



##### Bubble plot

- Aim: visualize stereotypes of differente align or ID
- $f(c):$  log(#appearance of $c$)
- $g(\omega ) = \sum_{c\_matches\_\omega} f(c)$
- One bubble per $\omega$
- Size: $g(\omega)$
- Color: World
- filters
    - Year (slider)
    - $\Omega$ (ticker)
    - whether to operate upon Align or ID (ticker?)

##### Heatmap
- correlations between features and (Align or ID)
- filter
    - $\Omega$ (ticker)
    - Year (slider)
    - World (DC or Marvel or Both) (ticker)


##### Confusion matrix (?) of predictions of linear models
- 80% train, 20% test
- Models
    - $features \rightarrow Align$
    - $features \rightarrow ID$
    - $features + ID \rightarrow Align$


#### Issues

##### How to deal with NaNs

- when a column is used in a filter and the filter is activated, remove rows with the value of that column being NaN
- only consider rows with values of columns of interest all being non-NaN..
- treat NaN as a possible value in the range of a filter

##### The relations among features, ID and Align
- the following follows (x, y)
- (features, ID)
- (features, Align)
- (features, ID $\times$ Align)
- (featuers + ID, Align)
- (featuers + Align, ID)
- .....?
