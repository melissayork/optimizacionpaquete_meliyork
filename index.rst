.. Paquete de optimizacion documentation master file, created by
   sphinx-quickstart on Fri Jul 12 16:06:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bienvenido a la documentación del paquete de optimización
=========================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Descripción
===========
Se presenta la documentación para el paquete "optimizacionpaquete_meliyork_1," el cual incluye algoritmos de optimización y funciones objetivo. Este paquete está diseñado para resolver problemas de optimización tanto de una variable como multivariados. A continuación, se detallan los algoritmos de optimización incluidos en el paquete:

- **Métodos para funciones de una variable:**

  - **Métodos de eliminación de regiones:**
      - Método de división de intervalos por la mitad
      - Búsqueda de Fibonacci
      - Método de la sección dorada 

  - **Métodos basados en la derivada:**
      - Método de Newton-Raphson
      - Método de bisección
      - Método de la secante

- **Métodos para funciones multivariadas:**

  - **Métodos directos:**
      - Caminata aleatoria
      - Método de Nelder y Mead (Simplex)
      - Método de Hooke-Jeeves

  - **Métodos de gradiente:**
      - Método de Cauchy
      - Método de Fletcher-Reeves
      - Método de Newton


Instalación
===========

.. code-block:: bash

    pip install optimizacionpaquete-meliyork-1

Para más información sobre el paquete, revisar: https://pypi.org/project/optimizacionpaquete-meliyork-1/

Ejemplo de cómo importar las funciones del paquete
==================================================

.. code-block:: python

   from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesMultivariables import himmelblau
   from optimizacionpaquete_meliyork_1.metodosMultivariables.hookeJeeves import hooke_jeeves

   xb_inicial = [-5, -2.5]
   delta = [0.5, 0.25]
   alpha = 2
   epsilon = 0.1
   resul = hooke_jeeves(himmelblau, xb_inicial, delta, alpha, epsilon)
   print(f"Resultado de booth: x = {resul}, f(x) = {himmelblau(resul)}")

Primero se accede al paquete mediante "optimizacionpaquete_meliyork_1", en este caso se desea importar la funcion "himmelblau" que se encuentra
en la carpeta "funcionesPrueba", en el archivo "funcionesMultivariables".

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modulo1
   modulo2