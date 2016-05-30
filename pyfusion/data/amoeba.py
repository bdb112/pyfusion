"""
# Amoeba uses the simplex method of Nelder and Mead to maximize a
# function of 1 or more variables.
#
#   Copyright (C) 2005  Thomas R. Metcalf
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
"""
def amoeba(var,scale,func,ftolerance=1.e-4,xtolerance=1.e-4,itmax=500,data=None):
    '''Use the simplex method to maximize a function of 1 or more variables.
    
       Input:
              var = the initial guess, a list with one element for each variable
              scale = the search scale for each variable, a list with one
                      element for each variable.
              func = the function to maximize.
              
       Optional Input:
              ftolerance = convergence criterion on the function values (default = 1.e-4)
              xtolerance = convergence criterion on the variable values (default = 1.e-4)
              itmax = maximum number of iterations allowed (default = 500).
              data = data to be passed to func (default = None).
              
       Output:
              (varbest,funcvalue,iterations)
              varbest = a list of the variables at the maximum.
              funcvalue = the function value at the maximum.
              iterations = the number of iterations used.

       - Setting itmax to zero disables the itmax check and the routine will run
         until convergence, even if it takes forever.
       - Setting ftolerance or xtolerance to 0.0 turns that convergence criterion
         off.  But do not set both ftolerance and xtolerance to zero or the routine
         will exit immediately without finding the maximum.
       - To check for convergence, check if (iterations < itmax).
              
       The function should be defined like func(var,data) where
       data is optional data to pass to the function.

       Example:
       
       >>> import amoeba
       >>> def afunc(var,data=None): return 1.0-var[0]*var[0]-var[1]*var[1]
       >>> print amoeba.amoeba([0.25,0.25],[0.5,0.5],afunc)
       ([0.0, 0.0], 1.0, 17)

       Version 1.0 2005-March-28 T. Metcalf
               1.1 2005-March-29 T. Metcalf - Use scale in simsize calculation.
                                            - Use func convergence *and* x convergence
                                              rather than func convergence *or* x
                                              convergence.
               1.2 2005-April-03 T. Metcalf - When contracting, contract the whole
                                              simplex.
      bdb added doctest                                         
      '''

    nvar = len(var)       # number of variables in the minimization
    nsimplex = nvar + 1   # number of vertices in the simplex
    
    # first set up the simplex

    simplex = [0]*(nvar+1)  # set the initial simplex
    simplex[0] = var[:]
    for i in range(nvar):
        simplex[i+1] = var[:]
        simplex[i+1][i] += scale[i]

    fvalue = []
    for i in range(nsimplex):  # set the function values for the simplex
        fvalue.append(func(simplex[i],data=data))

    # Ooze the simplex to the maximum

    iteration = 0
    
    while 1:
        # find the index of the best and worst vertices in the simplex
        ssworst = 0
        ssbest  = 0
        for i in range(nsimplex):
            if fvalue[i] > fvalue[ssbest]:
                ssbest = i
            if fvalue[i] < fvalue[ssworst]:
                ssworst = i
                
        # get the average of the nsimplex-1 best vertices in the simplex
        pavg = [0.0]*nvar
        for i in range(nsimplex):
            if i != ssworst:
                for j in range(nvar): pavg[j] += simplex[i][j]
        for j in range(nvar): pavg[j] = pavg[j]/nvar # nvar is nsimplex-1
        simscale = 0.0
        for i in range(nvar):
            simscale += abs(pavg[i]-simplex[ssworst][i])/scale[i]
        simscale = simscale/nvar

        # find the range of the function values
        fscale = (abs(fvalue[ssbest])+abs(fvalue[ssworst]))/2.0
        if fscale != 0.0:
            frange = abs(fvalue[ssbest]-fvalue[ssworst])/fscale
        else:
            frange = 0.0  # all the fvalues are zero in this case
            
        # have we converged?
        # print(iteration, frange, simscale)
        if (((ftolerance <= 0.0 or frange < ftolerance) and    # converged to maximum
             (xtolerance <= 0.0 or simscale < xtolerance)) or  # simplex contracted enough
            (itmax and iteration >= itmax)):             # ran out of iterations
            return simplex[ssbest],fvalue[ssbest],iteration

        # reflect the worst vertex
        pnew = [0.0]*nvar
        for i in range(nvar):
            pnew[i] = 2.0*pavg[i] - simplex[ssworst][i]
        fnew = func(pnew,data=data)
        if fnew <= fvalue[ssworst]:
            # the new vertex is worse than the worst so shrink
            # the simplex.
            for i in range(nsimplex):
                if i != ssbest and i != ssworst:
                    for j in range(nvar):
                        simplex[i][j] = 0.5*simplex[ssbest][j] + 0.5*simplex[i][j]
                    fvalue[i] = func(simplex[i],data=data)
            for j in range(nvar):
                pnew[j] = 0.5*simplex[ssbest][j] + 0.5*simplex[ssworst][j]
            fnew = func(pnew,data=data)
        elif fnew >= fvalue[ssbest]:
            # the new vertex is better than the best so expand
            # the simplex.
            pnew2 = [0.0]*nvar
            for i in range(nvar):
                pnew2[i] = 3.0*pavg[i] - 2.0*simplex[ssworst][i]
            fnew2 = func(pnew2,data=data)
            if fnew2 > fnew:
                # accept the new vertex in the simplex
                pnew = pnew2
                fnew = fnew2
        # replace the worst vertex with the new vertex
        for i in range(nvar):
            simplex[ssworst][i] = pnew[i]
        fvalue[ssworst] = fnew
        iteration += 1
        #if __debug__: print ssbest,fvalue[ssbest]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
     
