# optimizacionpaquete_meliyork_1

Un paquete de Python para algoritmos de optimización univariables y multivariables.

## Funciones

Este paquete incluye varias funciones para resolver problemas de optimización. Las métodos están divididos en dos categorías principales:

### Metodos Multivariables
##### Métodos Directos

- **Caminata Aleatoria (Random Walk):**
  Un método de optimización que explora el espacio de búsqueda realizando movimientos aleatorios para encontrar el mínimo de una función.

- **Método de Nelder y Mead (Simplex):**
  Un algoritmo de optimización basado en un método geométrico que utiliza un simplex (un polígono en n+1 dimensiones) para buscar el mínimo de una función.

- **Método de Hooke-Jeeves:**
  Un algoritmo de búsqueda directa que explora el espacio de búsqueda a través de movimientos de exploración y patrones de búsqueda para encontrar el mínimo de una función.

##### Métodos de Gradiente

- **Método de Cauchy:**
  Un algoritmo de optimización basado en el descenso del gradiente que ajusta la dirección de búsqueda utilizando el gradiente de la función objetivo.

- **Método de Fletcher-Reeves:**
  Un método de optimización basado en el descenso del gradiente conjugado que utiliza información del gradiente para mejorar la búsqueda del mínimo de una función.

- **Método de Newton:**
  Un método de optimización que utiliza la segunda derivada (matriz Hessiana) para encontrar el mínimo de una función mediante un método iterativo que mejora las estimaciones de los puntos de mínima.


### Metodos Univariables
##### Metodos de eliminacion de regiones
- **Fibonacci**: Método de búsqueda basado en la secuencia de Fibonacci.

- **Interval Halving**: Método de búsqueda que reduce el intervalo de búsqueda.

- **Búsqueda Dorada (Golden Search)**: Método de búsqueda que utiliza la relación áurea para encontrar el mínimo de una función unimodal.

##### Metodos basados en la derivada
- **Bisección**: Método de búsqueda que divide el intervalo en dos partes para encontrar la raíz de una función.

- **Secante**: Método para encontrar raíces de funciones basado en una aproximación de la derivada.

- **Newton-Raphson**: Método para encontrar raíces de funciones.


