# CSE150A_Project
My Project PEAS:
P: Performance - Accuracy by binary prediction, and P(yes), P(no),  Expected profit = (subscription value × P(y=‘yes’)) − (call cost)
E: Environment - Database of 41 k potential clients with features above; dynamic economic context, in real life, it could demonstrate real called strategy 
A: Actuators - Dial/not-dial decision (and possibly scheduling a call month/day)
S: Sensors- All dataset attributes plus real-time updates (new campaign counts, macro-indices)

What problem are you solving? Why does probabilistic modeling make sense to tackle this problem?
We must rank bank clients by likelihood of subscribing to a term-deposit so call-center agents target profitable leads. Probabilistic models output calibrated probabilities, handle uncertainty, combine categorical and numerical evidence, and support expected-profit decisions instead of crude binary yes-or-no classifications.
