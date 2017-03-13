# image_inpainting
Demonstrating the use of linearly stabilized schemes for image inpainting. 

## Current state
In the [figs](figs) folder are two figures, originally from the public domain, but now artificially "corrupted". These are just begging to be restored using image inpainting techniques. 

In [tv](tv), we have so far files covering explicit and implicit solvers, solving the diffusion and TV based inpainting models.

In [tvhneg](tvhneg), we have so far ... nothing.

## Plan 
The first step is to present the use of linearly stabilized schemes on two PDE-based inpainting models.

One is the TV model (Chan and Shen, 2001):

$$ u_t = div(grad(u) / abs(grad(u))) + lambda_{Omega\D}(u_0 - u),$$

and the other is the TV-H^{-1} (Bertozzi and Schoenlieb, 2011)

$$ u_t = Delta div(grad(u) / abs(grad(u))) + lambda_{Omega\D}(u_0 - u). $$

The techniques we use to evolve these models are the subject of [my thesis](https://github.com/kevstter/Thesis-B)

At some point down the road, it would be nice to look at the performance vs the direct energy minimization from a variational model as in [Getreuer](http://www.ipol.im/pub/art/2012/g-tvi/) and [Papafitsoros, Schoenlieb, and Sengul](http://www.ipol.im/pub/art/2013/40/), both of whom use the split Bregman algorithm to minimize an energy functional. 
