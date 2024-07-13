.. _algoritmos:

Métodos para funciones de una variable
======================================

Métodos de eliminación de regiones
----------------------------------

Método de división de intervalos por la mitad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    def interval_halving_method(f, e, a, b):
        
        """
        Encuentra el mínimo de una función unidimensional utilizando el método de reducción del intervalo.

        El método de reducción del intervalo, o **Interval Halving**, busca minimizar una función unidimensional reduciendo el intervalo de búsqueda en cada iteración, eligiendo entre dos puntos que se encuentran a la mitad de la longitud del intervalo.

        :param f: La función objetivo.
        :type f: function
        :param e: La tolerancia para la longitud del intervalo de búsqueda, el proceso se detiene cuando \(|b - a| < e\).
        :type e: float
        :param a: El límite inferior del intervalo de búsqueda.
        :type a: float
        :param b: El límite superior del intervalo de búsqueda.
        :type b: float
        :return: El punto en el intervalo \([a, b]\) donde la función tiene su mínimo.
        :rtype: float

        """
        L = b - a
        xm = (a + b) / 2

        while True:
            x1 = a + (L / 4)
            x2 = b - (L / 4)

            fx1 = f(x1)
            fx2 = f(x2)
            fxm = f(xm)

            if fx1 < fxm:
                b = xm
                xm = x1
            else:
                if fx2 < fxm:
                    a = xm
                    xm = x2
                else:
                    a = x1
                    b = x2

            L = b - a
            if abs(L) < e:
                return x1+x2/2 
            
**Ejemplo**

.. code-block:: python   

    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesUnivariables import f1
    from optimizacionpaquete_meliyork_1.metodosUnivariables.intervalHalving import interval_halving_method
    
    e=1e-5
    a=0
    b=4
    result= interval_halving_method(f1, e, a, b)
    print(result)




Búsqueda de Fibonacci
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   
    def fibonacci_search(f, e, a, b):
    
    """
    Encuentra el mínimo de una función unidimensional utilizando el método de búsqueda de Fibonacci.

    La búsqueda de Fibonacci es un método de optimización unidimensional que utiliza la serie de Fibonacci para reducir el intervalo de búsqueda de manera eficiente.

    :param f: La función objetivo que se desea minimizar.
    :type f: function
    :param e: El número deseado de evaluaciones en la serie de Fibonacci.
    :type e: int
    :param a: El límite inferior del intervalo de búsqueda.
    :type a: float
    :param b: El límite superior del intervalo de búsqueda.
    :type b: float
    :return: El punto en el intervalo \([a, b]\) donde la función tiene su mínimo.
    :rtype: float

    """
    
    L = b - a

    fib = [0, 1]
    while len(fib) <= e +2:
        fib.append(fib[-1] + fib[-2])

    
    k = 2

    while k < e:
        Lk = (fib[e - k + 2] / fib[e+ 2]) * L

        x1 = a + Lk
        x2 = b - Lk

        fx1 = f(x1)
        fx2 = f(x2)

        if fx1 < fx2:
            b = x2
        elif fx1 > fx2:
            a = x1
        elif fx1 == fx2:
            a=x1
            b=x2

        
        k += 1

    return a+b/2



**Ejemplo**

.. code-block:: python   

    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesUnivariables import f1
    from optimizacionpaquete_meliyork_1.metodosUnivariables.fibonacci import fibonacci_search
    
    e=3
    a=0
    b=4
    result = fibonacci_search(f1, e, a, b)
    print(f"Resultado: x = {result}, f(x) = {f1(result)}")


Método de la sección dorada (Búsqueda Dorada)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python 
    
    def busquedaDorada(funcion, e:float, a:float=None, b:float=None)->float:
    
    """
    Encuentra el mínimo de una función utilizando el método de búsqueda dorada.

    La búsqueda dorada es un método de optimización unidimensional basado en la proporción áurea para reducir el intervalo de búsqueda de manera eficiente.

    :param funcion: La función objetivo.
    :type funcion: function
    :param e: La tolerancia para el criterio de convergencia del método.
    :type e: float
    :param a: El límite inferior del intervalo de búsqueda. Si no se proporciona, se debe especificar.
    :type a: float, opcional
    :param b: El límite superior del intervalo de búsqueda. Si no se proporciona, se debe especificar.
    :type b: float, opcional
    :return: El punto en el intervalo \([a, b]\) donde la función tiene su mínimo.
    :rtype: float

    """
    
    def regla_eliminacion(x1, x2, fx1, fx2, a, b)->tuple[float, float]:
        if fx1>fx2:
            return x1, b
        
        if fx1<fx2:
            return a, x2
        
        return x1, x2 

    def w_to_x(w:float, a, b)->float:
        return w*(b-a)+a 
    
    phi=(1 + np.math.sqrt(5) )/ 2 - 1
    aw, bw=0,1
    Lw=1
    k=1

    while Lw>e:
        w2=aw+phi*Lw
        w1=bw-phi*Lw
        aw, bw=regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), funcion(w_to_x(w2, a, b)), aw, bw)
        k+=1
        Lw=bw-aw

    return(w_to_x(aw, a, b)+w_to_x(bw, a, b))/2


**Ejemplo**

.. code-block:: python 

    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesUnivariables import f1
    from optimizacionpaquete_meliyork_1.metodosUnivariables.busquedaDorada import busquedaDorada
    
    e=0.1
    a=0
    b=4
    
    resul = busquedaDorada(f1, e, a, b)
    print(f"Resultado: x = {resul}, f(x) = {f1(resul)}")

 
Métodos basados en la derivada
----------------------------------

Método de Newton-Raphson
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python 
    def newton_raphson(x_0, f, E):
    """
    Encuentra una raíz de una función unidimensional utilizando el método de Newton-Raphson.

    El método de Newton-Raphson es un método iterativo para encontrar soluciones de ecuaciones no lineales. En cada iteración, el método utiliza la derivada de la función para aproximar una mejor solución a la raíz de la ecuación.

    :param x_0: El valor inicial para el punto de partida del método iterativo.
    :type x_0: float
    :param f: La función objetivo.
    :type f: function
    :param E: La tolerancia para el criterio de convergencia, el proceso se detiene cuando \(|f'(x_{\text{next}})| < E\).
    :type E: float
    :return: El valor de \(x\) que aproxima una raíz de la función.
    :rtype: float

    """
    def primera_derivada(x, f):
        delta = 0.0001
        return (f(x + delta) - f(x - delta)) / (2 * delta)

    def segunda_derivada(x, f):
        delta = 0.0001
        return (f(x + delta) - 2 * f(x) + f(x - delta)) / (delta ** 2)
    
    k = 1

    while True:
        f_primera = primera_derivada(x_0, f)
        f_segunda = segunda_derivada(x_0, f)
        x_next = x_0 - (f_primera / f_segunda)
        f_prima_next = primera_derivada(x_next, f)
        
        if abs(f_prima_next) < E:
            break
        
        k += 1
        x_0 = x_next

    return x_next

**Ejemplo**

.. code-block:: python

    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesUnivariables import f1
    from optimizacionpaquete_meliyork_1.metodosUnivariables.newtonRaphson import newton_raphson
    
    x_0=1
    E=0.1
    resul= newton_raphson(x_0, f1, E)
    print(f"Resultado: x = {resul}, f(x) = {f1(resul)}")


Método de bisección
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python 
    
    def biseccion(f, e, a, b):
    """
    Realiza la búsqueda de la raíz de la derivada de la función `f` utilizando el método de bisección.

    Este método encuentra un punto donde la primera derivada de la función `f` es cero, lo cual puede indicar un máximo o un mínimo local.

    :param f: La función objetivo
    :type f: function
    :param e: La tolerancia para el criterio de convergencia.
    :type e: float
    :param a: El límite inferior del intervalo de búsqueda.
    :type a: float
    :param b: El límite superior del intervalo de búsqueda.
    :type b: float
    :return: El punto donde la primera derivada de `f` es cero.
    :rtype: float

    """
    
    def primera_derivada(x, f):
        delta = 0.0001
        return (f(x + delta) - f(x - delta)) / (2 * delta)
    
    a = np.random.uniform(a, b)
    b = np.random.uniform(a, b)
    
    while(primera_derivada(a,f) > 0):
        a = np.random.uniform(a, b)
    
    while (primera_derivada(b,f) < 0): 
        b = np.random.uniform(a, b)
    
    x1=a
    x2=b
    
    while True:
        z = (x1 + x2) / 2
        f_primaz = primera_derivada(z, f)
    
        if abs(f_primaz) < e:  
            break
        elif f_primaz < 0:
            x1 = z
        elif f_primaz > 0:
            x2 = z

    return x1+x2/2

**Ejemplo**

.. code-block:: python

    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesUnivariables import f1
    from optimizacionpaquete_meliyork_1.metodosUnivariables.biseccion import biseccion
    e=0.1
    a=0.1
    b=10
    result = biseccion(f1, e, a, b)
    print(result)


Método de secante
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python 
    
    def secante(f, e, a, b):
    """
    Encuentra una raíz de una función unidimensional utilizando el método de la secante.

    El método de la secante es una técnica iterativa para encontrar soluciones de ecuaciones no lineales. A diferencia del método de Newton-Raphson, la secante no requiere el cálculo de la derivada, sino que utiliza una aproximación basada en dos puntos previos.

    :param f: La función para la cual se busca una raíz.
    :type f: function
    :param e: La tolerancia para el criterio de convergencia.
    :type e: float
    :param a: El límite inferior del intervalo de búsqueda.
    :type a: float
    :param b: El límite superior del intervalo de búsqueda.
    :type b: float
    :return: El valor de \(x\) que aproxima una raíz de la función.
    :rtype: float

    """
    
    def primera_derivada(x, f):
        delta = 0.0001
        return (f(x + delta) - f(x - delta)) / (2 * delta)
       
    a = np.random.uniform(a, b)
    b = np.random.uniform(a, b)
    x1 = a
    x2 = b
    
    while True:
        z= x2- ( (primera_derivada(x2, f))  / (    ( (primera_derivada(x2, f)) - (primera_derivada(x1,f)) ) /   (x2-x1)   )     )
        f_primaz = primera_derivada(z, f)
    
        if abs(x2 - x1) < e: 
            break
        elif f_primaz < 0:
            x1 = z
        elif f_primaz > 0:
            x2 = z

    return x1+x2/2

**Ejemplo**

.. code-block:: python
 
    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesUnivariables import f1
    from optimizacionpaquete_meliyork_1.metodosUnivariables.secante import secante
    e=0.1
    a=0.1
    b=10

    resul = secante(f1, 1e-5, 1.0, 2.0)
    print(f"Resultado: x = {resul}, f(x) = {f1(raiz)}")



Métodos para funciones multivariadas
======================================

Métodos directos
----------------------------------

Caminata aleatoria 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python 
    
    def caminata_aleatoria(f, x0, step, iter_max):
        
        """
        Este método intenta encontrar un mínimo local de la función `f` realizando 
        pasos aleatorios desde el punto inicial `x0`.

        :param f: La función objetivo que se va a minimizar.
        :type f: function
        :param x0: El punto inicial desde donde se empieza la caminata aleatoria.
        :type x0: numpy.ndarray
        :param step: La magnitud máxima del paso aleatorio.
        :type step: float
        :param iter_max: El número máximo de iteraciones a realizar.
        :type iter_max: int
        :return: El punto donde se encontró el mínimo local.
        :rtype: numpy.ndarray
        :raises ValueError: Si `x0` no es un numpy.ndarray.
        
        """
        x = x0
        
        for i in range(iter_max):
            x_nuevo = x + np.random.uniform(-step, step, size=x.shape)
            if f(x_nuevo) < f(x):
                x = x_nuevo
        return x
        

**Ejemplo**

.. code-block:: python
    
    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesMultivariables import himmelblau
    from optimizacionpaquete_meliyork_1.metodosMultivariables.caminataAleatoria import caminata_aleatoria

    x0 = np.array([1.0, 1.0])
    step = 0.1
    iter_max = 1000
    result = caminata_aleatoria(himmelblau, x0, step, iter_max)
    print(result)


Método de Nelder y Mead (Simplex) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python 
    
    def nelder_mead(funcion, inicio):
    
    """
    Este método intenta encontrar un mínimo local de la función `funcion` utilizando un algoritmo de búsqueda directa conocido como el método simplex de Nelder-Mead.

    :param funcion: La función objetivo que se va a minimizar.
    :type funcion: function
    :param inicio: El punto inicial desde donde comienza la optimización.
    :type inicio: list or numpy.ndarray
    :return: El punto donde se encontró el mínimo local.
    :rtype: numpy.ndarray
    
    """
    dimensiones = len(inicio)
    alfa = 1.0
    gamma = 2.0
    beta = 0.5
    tolerancia = 1e-5
    iter_max = 1000
    
    delta1 = (np.sqrt(dimensiones + 1) + dimensiones - 1) / (dimensiones * np.sqrt(2)) * alfa
    delta2 = (np.sqrt(dimensiones + 1) - 1) / (dimensiones * np.sqrt(2)) * alfa
    
    simplex = np.zeros((dimensiones + 1, dimensiones))
    simplex[0] = inicio
    
    for i in range(1, dimensiones + 1):
        punto = inicio.copy()
        punto[i - 1] += delta1
        for j in range(dimensiones):
            if j != i - 1:
                punto[j] += delta2
        simplex[i] = punto
    
    for iteracion in range(iter_max):
        simplex = sorted(simplex, key=funcion)
        simplex = np.array(simplex)
        
        centroide = np.mean(simplex[:-1], axis=0)
        reflexion = 2 * centroide - simplex[-1]
        
        if funcion(reflexion) < funcion(simplex[0]):
            expansion = centroide + gamma * (centroide - simplex[-1])
            nuevo_punto = expansion if funcion(expansion) < funcion(reflexion) else reflexion
        elif funcion(reflexion) >= funcion(simplex[-2]):
            if funcion(reflexion) < funcion(simplex[-1]):
                contraccion_fuera = centroide + beta * (reflexion - centroide)
                nuevo_punto = contraccion_fuera
            else:
                contraccion_dentro = centroide - beta * (centroide - simplex[-1])
                nuevo_punto = contraccion_dentro
        else:
            nuevo_punto = reflexion
        
        simplex[-1] = nuevo_punto
        
        if np.sqrt(np.mean([(funcion(x) - funcion(centroide))**2 for x in simplex])) <= tolerancia:
            break

    simplex = sorted(simplex, key=funcion)
    simplex = np.array(simplex)
    
    return simplex[0]


**Ejemplo**

.. code-block:: python
    
    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesMultivariables import himmelblau
    from optimizacionpaquete_meliyork_1.metodosMultivariables.nelderMeadSimplex import nelder_mead
    
    inicio = np.array([-1.2, 1.0])
    >result = nelder_mead(himmelblau, inicio)
    print(result)



Método de Hooke-Jeeves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    def hooke_jeeves(f, x_initial, delta, alpha, epsilon):
        
        """
        Este método intenta encontrar un mínimo local de la función `f` utilizando un algoritmo de búsqueda directa.

        :param f: La función objetivo que se va a minimizar.
        :type f: function
        :param x_initial: El punto inicial desde donde comienza la optimización.
        :type x_initial: list or numpy.ndarray
        :param delta: El tamaño del paso para la búsqueda exploratoria.
        :type delta: list or numpy.ndarray
        :param alpha: El factor de reducción para el tamaño del paso.
        :type alpha: float
        :param epsilon: El umbral para determinar la convergencia.
        :type epsilon: float
        :return: El punto donde se encontró el mínimo local.
        :rtype: numpy.ndarray
        :raises ValueError: Si `x_initial` o `delta` no son listas o numpy.ndarrays.
        
        """
        def movimiento_exploratorio(xc, delta, func):
            x = np.copy(xc)
            for i in range(len(x)):
                f = func(x)
                x[i] += delta[i]
                f_mas = func(x)
                if f_mas < f:
                    f = f_mas
                else:
                    x[i] -= 2*delta[i]
                    f_menos = func(x)
                    if f_menos < f:
                        f = f_menos
                    else:
                        x[i] += delta[i]
            return x
        
        x = np.array(x_initial)
        delta = np.array(delta)
        while True:
            x_nuevo = movimiento_exploratorio(x, delta, f)
            
            if np.array_equal(x, x_nuevo):
                if np.linalg.norm(delta) < epsilon:
                    break
                else:
                    delta /= alpha
                    continue
            
            x_p = x_nuevo + (x_nuevo - x)
            x_p_nuevo = movimiento_exploratorio(x_p, delta, f)
            
            if f(x_p_nuevo) < f(x_nuevo):
                x = x_p_nuevo
            else:
                x = x_nuevo
        
        return x 

**Ejemplo**

.. code-block:: python

    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesMultivariables import himmelblau
    from optimizacionpaquete_meliyork_1.metodosMultivariables.hookeJeeves import hooke_jeeves

    x_initial = [-5, -2.5]
    delta = [0.5, 0.25]
    alpha = 2
    epsilon = 0.1
    result = hooke_jeeves(himmelblau, x_initial, delta, alpha, epsilon)
    print(result)


Métodos de gradiente
----------------------------------

Método de Cauchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python 
    
    def cauchy(f, x0, epsilon1, epsilon2,  maxiter, metodo):
    
    """
    Este método intenta encontrar un mínimo local de la función `f` usando el gradiente descendente y una búsqueda de línea
    con el método especificado.

    :param f: La función objetivo que se va a minimizar.
    :type f: function
    :param x0: El punto inicial desde donde comienza la optimización.
    :type x0: numpy.ndarray
    :param epsilon1: El umbral para la norma del gradiente bajo el cual se considera que la solución ha convergido.
    :type epsilon1: float
    :param epsilon2: El umbral para la norma del cambio relativo en `xk` bajo el cual se considera que la solución ha convergido.
    :type epsilon2: float
    :param maxiter: El número máximo de iteraciones.
    :type maxiter: int
    :param metodo: El método de búsqueda de línea a utilizar.
    :type metodo: function
    :return: El punto donde se encontró el mínimo local.
    :rtype: numpy.ndarray
    :raises ValueError: Si `x0` no es un numpy.ndarray.
    
    :Ejemplo:

    >>> import numpy as np
    >>> def f(x):
    >>>     return np.sum(x**2)
    >>> def fibonacci_search(f, e, a, b):
    >>>     L = b - a
    >>>     fib = [0, 1]
    >>>     while len(fib) <= e + 2:
    >>>         fib.append(fib[-1] + fib[-2])
    >>>     k = 2
    >>>     while k < e:
    >>>         Lk = (fib[e - k + 2] / fib[e + 2]) * L
    >>>         x1 = a + Lk
    >>>         x2 = b - Lk
    >>>         fx1 = f(x1)
    >>>         fx2 = f(x2)
    >>>         if fx1 < fx2:
    >>>             b = x2
    >>>         elif fx1 > fx2:
    >>>             a = x1
    >>>         elif fx1 == fx2:
    >>>             a = x1
    >>>             b = x2
    >>>         k += 1
    >>>     return (a + b) / 2
    
    """
    def gradiente(f, x, deltaX=0.001):
        grad=[]
        for i in range(0, len(x)):
            xp=x.copy()
            xn=x.copy()
            xp[i]=xp[i]+deltaX
            xn[i]=xn[i]-deltaX
            grad.append((f(xp)-f(xn))/(2*deltaX))
        return grad
    
    terminar=False
    xk=x0
    k=0

    while not terminar:
        grad=np.array(gradiente(f, xk))

        if np.linalg.norm(grad)<epsilon1 or k>=maxiter:
            terminar=True
        else:
            def alpha_funcion(alpha):
                return f(xk-alpha*grad)
            
            alpha=metodo(alpha_funcion, e=epsilon2, a=0.0, b=1.0) 
            x_k1=xk-alpha*grad

            if np.linalg.norm(x_k1-xk)/(np.linalg.norm(xk)+0.00001) <= epsilon2:
                terminar=True
            else:
                k=k+1
                xk=x_k1
    return xk

**Ejemplo**

.. code-block:: python

  
    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesMultivariables import himmelblau
    from optimizacionpaquete_meliyork_1.metodosUnivariables.fibonacci import fibonacci_search
    from optimizacionpaquete_meliyork_1.metodosMultivariables.cauchy import cauchy

    x0=np.array([0.0, 0.0])
    epsilon1=0.001
    epsilon2=0.001
    max_iter=100
    alpha=0.2
    result = print(cauchy(himmelblau, x0, epsilon1, epsilon2, max_iter, fibonacci_search))
    print(result)
 


Método de Fletcher-Reeves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python 

    def fletcherReeves(f, x0, epsilon1, epsilon2, epsilon3, metodo):

    """
    Este método intenta encontrar un mínimo local de la función `f` utilizando gradiente conjugado con la actualización
    de Fletcher-Reeves.

    :param f: La función objetivo que se va a minimizar.
    :type f: function
    :param x0: El punto inicial desde donde comienza la optimización.
    :type x0: numpy.ndarray
    :param epsilon1: El umbral para la búsqueda de línea.
    :type epsilon1: float
    :param epsilon2: El umbral para el cambio relativo en `x`.
    :type epsilon2: float
    :param epsilon3: El umbral para la norma del gradiente bajo el cual se considera que la solución ha convergido.
    :type epsilon3: float
    :param metodo: El método de búsqueda de línea a utilizar.
    :type metodo: function
    :return: El punto donde se encontró el mínimo local.
    :rtype: numpy.ndarray
    :raises ValueError: Si `x0` no es un numpy.ndarray.

    """

    def gradiente(f, x, deltaX=0.001):
        grad = []
        for i in range(len(x)):
            xp = x.copy()
            xn = x.copy()
            xp[i] = xp[i] + deltaX
            xn[i] = xn[i] - deltaX
            grad.append((f(xp) - f(xn)) / (2 * deltaX))
        return np.array(grad)

    x = x0
    grad = gradiente(f, x)
    s = -grad
    k = 0

    while True:
        alpha = metodo(lambda alpha: f(x + alpha * s), e=epsilon1, a=0.0, b=1.0)
        x_next = x + alpha * s
        grad_next = gradiente(f, x_next)

        if np.linalg.norm(x_next - x) / np.linalg.norm(x) <= epsilon2 or np.linalg.norm(grad_next) <= epsilon3:
            break

        beta = np.linalg.norm(grad_next) ** 2 / np.linalg.norm(grad) ** 2
        s = -grad_next + beta * s

        x = x_next
        grad = grad_next
        k += 1

    return x

**Ejemplo**

.. code-block:: python

    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesMultivariables import himmelblau
    from optimizacionpaquete_meliyork_1.metodosUnivariables.fibonacci import fibonacci_search
    from optimizacionpaquete_meliyork_1.metodosMultivariables.cauchy import cauchy
  
    x0 = np.array([2.0, 3.0])
    epsilon1 = 0.001
    epsilon2 = 0.001
    epsilon3 = 0.001
    result = fletcherReeves(himmelblau, x0, epsilon1, epsilon2, epsilon3, fibonacci_search)
    print(result)



Método de Newton
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python 
    
    def newton(f, x0, epsilon1, epsilon2, maxiter, metodo):

        """
        Este método intenta encontrar un mínimo local de la función `f` utilizando el método de Newton, que emplea tanto el gradiente como la matriz Hessiana de la función objetivo.

        :param f: La función objetivo.
        :type f: function
        :param x0: Punto inicial.
        :type x0: list or numpy.ndarray
        :param epsilon1: Criterio de convergencia basado en el gradiente.
        :type epsilon1: float
        :param epsilon2: Criterio de convergencia basado en el cambio en las variables.
        :type epsilon2: float
        :param maxiter: Número máximo de iteraciones permitidas.
        :type maxiter: int
        :param metodo: Método de búsqueda de línea para determinar el paso óptimo.
        :type metodo: function
        :return: El punto donde se encontró el mínimo local.
        :rtype: numpy.ndarray

        """
        terminar = False
        xk = x0
        k = 0

        def gradiente(f, x, deltaX=0.001):
            grad = []
            for i in range(len(x)):
                xp = x.copy()
                xn = x.copy()
                xp[i] = xp[i] + deltaX
                xn[i] = xn[i] - deltaX
                grad.append((f(xp) - f(xn)) / (2 * deltaX))
            return np.array(grad)
        
        def hessian_matrix(f, x, deltaX):
            fx = f(x)
            N = len(x)
            H = []
            for i in range(N):
                hi = []
                for j in range(N):
                    if i == j:
                        xp = x.copy()
                        xn = x.copy()
                        xp[i] = xp[i] + deltaX
                        xn[i] = xn[i] - deltaX
                        hi.append((f(xp) - 2 * fx + f(xn)) / (deltaX ** 2))
                    else:
                        xpp = x.copy()
                        xpn = x.copy()
                        xnp = x.copy()
                        xnn = x.copy()
                        xpp[i] = xpp[i] + deltaX
                        xpp[j] = xpp[j] + deltaX
                        xpn[i] = xpn[i] + deltaX
                        xpn[j] = xpn[j] - deltaX
                        xnp[i] = xnp[i] - deltaX
                        xnp[j] = xnp[j] + deltaX
                        xnn[i] = xnn[i] - deltaX
                        xnn[j] = xnn[j] - deltaX
                        hi.append((f(xpp) - f(xpn) - f(xnp) + f(xnn)) / (4 * deltaX ** 2))
                H.append(hi)
            return np.array(H)

        while not terminar:
            grad = np.array(gradiente(f, xk))
            hessian = hessian_matrix(f, xk, deltaX=0.001)
            hessian_inv = np.linalg.inv(hessian)

            if np.linalg.norm(grad) < epsilon1 or k >= maxiter:
                terminar = True
            else:
                def alpha_funcion(alpha):
                    return f(xk - alpha * np.dot(hessian_inv, grad))

                alpha = metodo(alpha_funcion, e=epsilon2, a=0.0, b=1.0)
                x_k1 = xk - alpha * np.dot(hessian_inv, grad)

                if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 0.00001) <= epsilon2:
                    terminar = True
                else:
                    k += 1
                    xk = x_k1
        return xk

**Ejemplo**

.. code-block:: python

    from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesMultivariables import himmelblau
    from optimizacionpaquete_meliyork_1.metodosUnivariables.busquedaDorada import busquedaDorada
    from optimizacionpaquete_meliyork_1.metodosMultivariables.newton import newton

    x0=np.array([0.0, 0.0])
    epsilon1=0.001
    epsilon2=0.001
    max_iter=100
    result = newton(himmelblau, x0, epsilon1, epsilon2, 1000, fibonacci_search)
    print(f"Resultado: x = {result}, f(x) = {rosenbrock(result)}")