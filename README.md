# image_inpainting
Demonstrating the use of linearly stabilized schemes for image inpainting. 

## Plan 
The first step is to present the use of linearly stabilized schemes on two PDE-based inpainting models.

One is the TV model (Chan and Shen, 2001):

$$ u_t = div(grad(u) / abs(grad(u))) + lambda_{Omega\D}(u_0 - u),$$

and the other is the TV-H^{-1} (Bertozzi and Schoenlieb, 2011)

$$ u_t = Delta div(grad(u) / abs(grad(u))) + lambda_{Omega\D}(u_0 - u). $$

The techniques we use to evolve these models are the subject of [my thesis](https://github.com/kevstter/Thesis-B)

