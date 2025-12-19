# 🇪🇸 Visualizando _Flow Matching_

- __Autor__: Víctor González García

![animation](https://github.com/user-attachments/assets/f7bf6da9-1ee9-42fc-9ff0-e244854a4b5a)

<img width="1346" height="763" alt="program" src="https://github.com/user-attachments/assets/7e2be1d2-1124-4ec0-b8bf-5b8bf3e5f795" />

Esta es una pequeña aplicación que nos permite visualizar cómo se generar datos a través de _Flow Matching_La aplicación nos permite escoger entre dos distribuciones "de juguete" y otra opción en la que podemos dibujar nuestro dataset con el ratón. Una vez escogidos los datos de los que queremos aprender podremos ver una gráfica de la función de pérdida en cada época de entrenamiento y una animación de como se transforma el ruido inicial gaussiano en la distribución que hayamos elegido. Con la opción de dibujar uno puede darse cuenta de la flexibilidad de estos métodos para aprender distribuciones arbitrarias.

## Instalación.
1. Descarga el repositorio.
2. Crea un entorno virtual en el repositorio con el comando `python3 -m venv .venv`. Asegúrate de estar en la carpeta principal del proyecto (donde está `app.py`).
3. Activa el entorno `source .venv/bin/activate`.
4. Instala las librerías: `pip install -r requirements.txt`.
5. Ejecuta la aplicación `streamlit run app.py`.
   
## Teoría
El _Flow Matching_ es una técnica de __IA generativa__ basada en ecuaciones diferenciales ordinarias que transportan distribuciones, desde una gaussiana normalmente --aunque no exclusivamente-- hasta la distribución teórica que subyace a una muestra de datos (o _dataset_). 

Rigurosamente el problema de generación se puede plantear de la siguiente manera:

> Dada una muestra de datos $X\rightsquigarrow p_{data}$ (texto, imagen o cualquier dato que pueda representarse de forma vectorial) queremos generar un nuevo punto que provenga de esta distribución.

La principal limitación es que no conocemos de qué distribución provienen los datos. En _Flow Matching_ el objetivo es descubrir __qué campo vectorial transporta la distribución inicial a la distribución de los datos__. Este campo se conoce en general:

> Se puede demostrar que si $X_0\rightsquigarrow p_{init}$ es una nube de puntos distribuida como $p_{init}$, entonces la nube de puntos solución del problema
> 
> 
> $\frac{dX_t}{dt}=u_t(X_t)$
>
> donde $\displaystyle \int u_t(x|z)\frac{p_t(x|z)p_{data}(z)}{p_t(x)} dz$, verifica que $X_1\rightsquigarrow p_{data}$. Entonces al final de la simulación ($t=1$) obtenemos una nube de puntos con la misma distribución de los datos.


Los términos que aparecen en la integral son técnicos y no los explicaré aquí, pero observamos que para poder evaluar el campo necesitamos conocer la distribución $p_{data}$ (!!!). Ese era nuestro objetivo desde el principio. Es por eso que no podemos usar esta fórmula tal cual esta así (incluso si la conociéramos, esa integral es intratable).

En la práctica lo que hacemos es sustituir el campo vectorial (la integral intratable) por una red neuronal $u_t^\theta(x)$ y entrenarla para que reproduzca de forma fiel el valor de la integral. Es por eso que este es un método de ___Deep Learning___.

Para entrenar esta red lo que debemos de hacer es encontrar $\theta\in\mathbb{R}^N$ que minimice el error cuadrático medio:

$\mathbb{E}[\Vert u_t^\theta(x)-u_t(x)\Vert^2]$, donde $x \rightsquigarrow p_{data}$

De nuevo, por no conocer el valor exacto de $u_t(x)$ recurrimos a otra función de pérdida: la _función de pérdida condicional_.

> $\mathbb{E}[\Vert u_t^\theta(x)-u_t(x|z)\Vert^2]$, donde $t \rightsquigarrow U[0,1],\quad z \rightsquigarrow p_{data},\quad x\rightsquigarrow p_t(·|z)$

Esta función sí es calculable (aunque no hayamos explicado qué es $u_t(x|z)$ y además se demuestra que los parámetros $\theta$ que minimizan la función de pérdida condicional minimizan también la función de pérdida original que nos interesaba. Así que a todos los efectos el entrenamiento se puede producir usando la función de pérdida original. A esto se le conoce como

Una buena noticia es que la expresión de la función de pérdida condicional en el caso en el que $p_{init}=\mathcal{N}(0,I)$ es una normal de media 0 y matriz de covarianza $I$, la identidad, es muy sencilla:

$\mathcal{L}(\theta)=\mathbb{E}[\Vert u_t^\theta(tz+(1-t)\varepsilon)-(z-\varepsilon)\Vert^2]$, donde $t\rightsquigarrow U[0,1],\quad z\rightsquigarrow p_{data}, \quad \varepsilon \rightsquigarrow \mathcal{N}(0,I)$

En resumen lo que debemos hacer es:

1. Tomar al azar $z$ del _dataset_.
2. Generar un número aleatorio $t \rightsquigarrow U[0,1]$.
3. Generar ruido $\varepsilon \rightsquigarrow \mathcal{N}(0,I)$.
4. Calcular $x=tz+(1-t)\varepsilon$.
5. Calcular la función de pérdida condicional: $\mathcal{L}(\theta)=\mathbb{E}[\Vert u_t^\theta(tz+(1-t)\varepsilon)-(z-\varepsilon)\Vert^2]$
6. Actualizar los parámetros $\theta$ vía descenso de gradiente aplicado a $\mathcal{L}(\theta)$.


Una vez entrenado el modelo debemos integrar la ecuación con algún método (Euler, RK4, etc.) y obtendremos en $t=1$ el dato generado.
