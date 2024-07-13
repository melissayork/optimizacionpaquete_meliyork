Instalacion
=====

.. _Instalación:

Puedes instalar la última versión de `optimizacionpaquete_meliyork_1` desde PyPI usando pip:

.. code-block:: bash

    pip install optimizacionpaquete-meliyork-1

**Requisitos de instalacion**

Antes de descargar asegurate de tener instalado:
- Python 3.6 o superior
- pip (el instalador de paquetes de Python)

Si requieres verificar la version de Python y de pip, ejecuta los siguientes comandos en tu terminal:

.. code-block:: bash

    python --version
    pip --version

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