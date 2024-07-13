.. _funcionesObjetivo:

Funciones 
======================================

Funciones univariables 
----------------------------------

.. code-block:: python
    
    def f1(x):
    return (x**2) + 54/x

    def f2(x):
        return (x**3) + 2 * x - 3

    def f3(x):
        return (x**4) + (x**2) - 33

    def f4(x):
        return 3 * (x**4) - 8 * (x**3) - 6 * (x**2) + 12 * x

**Ejemplo de cómo importar las funciones**

.. code-block:: python

    
    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesUnivariables import f1, f2, f3, f4

Funciones objetivo
----------------------------------

Rastrigin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    def rastrigin(x):
        
        """
        La función Rastrigin es una función objetivo comúnmente utilizada en pruebas de algoritmos de optimización. Es una función no convexa con un mínimo global en el origen, y su diseño es adecuado para evaluar la capacidad de los algoritmos para explorar un espacio de búsqueda con múltiples óptimos locales.

        :param x: Un vector de números reales en el cual se evalúa la función Rastrigin.
        :type x: numpy.ndarray
        :return: El valor de la función Rastrigin evaluada en el vector `x`.
        :rtype: float
    
        """
        A=10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
**Ejemplo de uso**

.. code-block:: python
    
    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesMultivariables import rastrigin
    x = np.array([0.1, 0.2])
    valor = rastrigin(x)
    print(f"Valor de Rastrigin en x: {valor}")


Ackley
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    def ackley(x):
    
        """
        La función Ackley es una función objetivo utilizada en pruebas de algoritmos de optimización. Es una función no convexa con un mínimo global en el origen, y su diseño es adecuado para evaluar la capacidad de los algoritmos para explorar un espacio de búsqueda con múltiples óptimos locales.

        :param x: Un vector de números reales en el cual se evalúa la función Ackley.
        :type x: numpy.ndarray
        :return: El valor de la función Ackley evaluada en el vector `x`.
        :rtype: float

        :Ejemplo:

        >>> x = np.array([0.1, 0.2])
        >>> valor = ackley(x)
        >>> print(f"Valor de Ackley en x: {valor}")

        """
        a = 20
        b = 0.2
        c = 2 * np.pi
        suma1 = x[0]**2 + x[1]**2
        suma2 = np.cos(c * x[0]) + np.cos(c * x[1])
        term1 = -a * np.exp(-b * np.sqrt(0.5 * suma1))
        term2 = -np.exp(0.5 * suma2)
        resul = term1 + term2 + a + np.exp(1)
        return resul

