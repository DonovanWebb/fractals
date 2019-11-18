# Import libraries for simulation
import tensorflow as tf
import numpy as np
from numpy import arange, pi, exp

# Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import Image, display
import matplotlib.pyplot as plt
import time
import imageio

def nthRootsOfUnity(n):
    return np.round_(exp(2j * pi / n * arange(n)), 2)

def DisplayFractal(a, b, iterations, i ):
    """Display an array of iteration counts as a
       colorful picture of a fractal."""
    # a = (np.real(a) + 2*np.imag(a)) + np.clip((iterations-b), 0, 20)/10
    a = iterations - b
    #imageio.imwrite('background_fract_col.png', a)
    plt.figure(i)
    plt.imshow(a, cmap='inferno')
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    #plt.savefig('bg_fract.png', dpi=1000)

def main(order, iterations, step):


    start = time.time()

    sess = tf.InteractiveSession()

    # Use NumPy to create a 2D array of complex numbers

    X, Y = np.mgrid[-2:1:step, -2.5:2.5:step]
    Z = X+1j*Y

    xs = tf.constant(Z.astype(np.complex64))
    zs = tf.Variable(xs)
    ns = tf.Variable(tf.zeros_like(xs, tf.float32))

    tf.global_variables_initializer().run()

    # Compute the new values of z: z^2 + x
    # zs_ = (zs - ((zs**order-1) / (order*zs**(order-1)))) # Basic
    # zs_ = (zs - ((zs**order-1) / (order*zs**(order-2)))) # NICE ONE
    zs_ = (zs - ((zs**order-1) / (order*zs**(order-2))))

    # Have we diverged with this new value?
    not_diverged = tf.abs(zs_) < 4

    # Operation to update the zs and the iteration count.
    #
    # Note: We keep computing zs after they diverge! This
    #       is very wasteful! There are better, if a little
    #       less simple, ways to do this.
    #
    step = tf.group(
        zs.assign(zs_),
        ns.assign_add(tf.cast(not_diverged, tf.float32))
        )

    for i in range(iterations): step.run()


    colors = np.array([])
    roots = nthRootsOfUnity(order)
    complexlist = []
    for root in roots:
        complexlist.append(np.complex(root))
    endplot = time.time()
    print(f'took {endplot-start}s')

    return zs_.eval(), ns.eval()
    #DisplayFractal(ns.eval())

for i in range(500,530):
    iterations = 100
    # 0.005 takes ~5 seconds
    step = 0.005
    zs, ns = main(i/100, iterations, step)
    DisplayFractal(zs, ns, iterations, i)
plt.show()
