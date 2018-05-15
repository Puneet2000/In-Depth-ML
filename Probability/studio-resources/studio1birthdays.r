#-----------
# In studio we showed  studio1.r. This is extra code used to make the birthday plot we showed them
#---------------------------

#Need colMatches
source('colMatches.r')

#Set real parameters
# Look for 2 matching birthdays
ndays = 365
npeople = 20
ntrials = 10000
sizematch = 2

# Run ntrials together
year = 1:ndays
y =  sample(year, npeople*ntrials, replace=TRUE)
trials = matrix(y, nrow=npeople, ncol=ntrials)
w = colMatches(trials,sizematch)
mean(w)

#Repeat

#---------------------------
# Look for triples
ndays = 365
npeople = 100
ntrials = 10000
sizematch = 3

# Run ntrials together
year = 1:ndays
y =  sample(year, npeople*ntrials, replace=TRUE)
trials = matrix(y, nrow=npeople, ncol=ntrials)
w = colMatches(trials,sizematch)
mean(w)

#Repeat
#---------------------------
# Plot probability of match vs npeople
ndays = 365
ntrials = 200
sizematch = 2
maxpeople = 100

# Run ntrials together
people = 1:maxpeople;
p = rep(0, length(people))
for (npeople in people)
{
    year = 1:ndays
    y =  sample(year, npeople*ntrials, replace=TRUE)
    trials = matrix(y, nrow=npeople, ncol=ntrials)
    w = colMatches(trials,sizematch)
    p[npeople] = mean(w)
}
plot(people,p,type='l', col='blue', lwd=2)

#Repeat
