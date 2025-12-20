![animation](https://github.com/user-attachments/assets/f7bf6da9-1ee9-42fc-9ff0-e244854a4b5a)

<img width="1346" height="763" alt="program" src="https://github.com/user-attachments/assets/7e2be1d2-1124-4ec0-b8bf-5b8bf3e5f795" />

# 🇬🇧 Visualizing Flow Matching
- __Author__: Víctor González García

This is a small application that allows us to visualize how data is generated through Flow Matching. The application allows us to choose between two "toy" distributions and another option where we can draw our dataset with the mouse. Once the data is chosen, we can see a graph of the loss function at each training epoch and an animation of how the initial Gaussian noise transforms into the distribution. With the drawing option, one can realize the capability of these methods to learn arbitrary distributions.

A Jupyter Notebook is also included to guide us through the process in case we want to replicate the behavior. The network architecture consists of an input layer with 3 neurons (x, y, t), two hidden layers of 128 neurons each, and a final output layer with two neurons (vx, vy). All layers except the output layer use the SiLU (Sigmoid Linear Unit) activation function.

## Installation
1. Download the repository.
2. Create a virtual environment in the repository with the command `python3 -m venv .venv`. Make sure you are in the main project folder (where `app.py` is located).
3. Activate the environment `source .venv/bin/activate`.
4. Install the libraries: `pip install -r requirements.txt`.
5. Run the app with streamlit run `app.py`.

## Theory
Flow Matching is a Generative AI technique based on ordinary differential equations (ODEs) that transport distributions, usually—though not exclusively—from a Gaussian to the theoretical distribution underlying a data sample (or dataset).

Rigorously, the generation problem can be posed as follows:

> Given a data sample $X\rightsquigarrow p_{data}$ (text, image, or any data that can be represented vectorially), we want to generate a new point originating from this distribution.

The main limitation is that we do not know which distribution the data comes from. In Flow Matching, the objective is to discover __which vector field transports the initial distribution to the data distribution__. This field is generally known:

> It can be shown that if $X_0\rightsquigarrow p_{init}$ eis a point cloud distributed as $p_{init}$, then the point cloud solution to the problem
> 
> 
> $\frac{dX_t}{dt}=u_t(X_t)$
>
> where $\displaystyle \int u_t(x|z)\frac{p_t(x|z)p_{data}(z)}{p_t(x)} dz$, satisfies $X_1\rightsquigarrow p_{data}$. Thus, ath the end of the simulation ($t=1$) we obtain a point cloud with the same distribution as the data.

The terms appearing in the integral are technical, and I will not explain them here, but we observe that to evaluate the field we need to know the distribution $p_{data}$ (!!!). That was our goal from the beginning. That is why we cannot use this formula as is (even if we knew it, that integral is intractable). 

In practice, what we do is replace the vector field (the intractable integral) with a neural network $u_t^\theta(x)$ and train it to faithfully reproduce the value of the integral. That is why this is a __Deep Learning__ method.

To train this network, we must find $\theta\in\mathbb{R}^N$ that minimizes the mean squared error:

$\mathbb{E}[\Vert u_t^\theta(x)-u_t(x)\Vert^2]$, where $x \rightsquigarrow p_{data}$
 
Again, because we do not know the exact value of $u_t(x)$, we resort to another loss function: the conditional loss function.

> $\mathbb{E}[\Vert u_t^\theta(x)-u_t(x|z)\Vert^2]$, where $t \rightsquigarrow U[0,1],\quad z \rightsquigarrow p_{data},\quad x\rightsquigarrow p_t(·|z)$

This function is computable (even though we haven't explained what $u_t(x|z)$ is), and furthermore, it is proven that the parameters $\theta$ that minimize the conditional loss function also minimize the original loss function we were interested in. This is known as __Conditional Flow Matching__.

Conveniently, the expression for the conditional loss function in the case where $p_{init}=\mathcal{N}(0,I)$(a normal distribution with mean $0$ and covariance matrix $I$, the identity) is very simple:

$\mathcal{L}(\theta)=\mathbb{E}[\Vert u_t^\theta(tz+(1-t)\varepsilon)-(z-\varepsilon)\Vert^2]$, where $t\rightsquigarrow U[0,1],\quad z\rightsquigarrow p_{data}, \quad \varepsilon \rightsquigarrow \mathcal{N}(0,I)$

In summary, what we must do is:

1. Sample z randomly from the dataset.
2. Sample a random number $t \rightsquigarrow U[0,1]$.
3. Sample noise $\varepsilon \rightsquigarrow \mathcal{N}(0,I)$.
4. Calculate $x=tz+(1−t)\varepsilon$.
5. Calculate the conditional loss function: $\mathcal{L}(\theta)=\mathbb{E}[\Vert u_t^\theta(tz+(1-t)\varepsilon)-(z-\varepsilon)\Vert^2]$
6. Update parameters θ via gradient descent applied to $\mathcal{L}(\theta)$.

Once the model is trained, what we must do to obtained the generated data is simply use an EDO solver (Euler, RK4, etc.) and evaluate the solution at $t=1$.

# 🇪🇸 Visualizando _Flow Matching_

- __Autor__: Víctor González García

Esta es una pequeña aplicación que nos permite visualizar cómo se generar datos a través de _Flow Matching_. La aplicación nos permite escoger entre dos distribuciones "de juguete" y otra opción en la que podemos dibujar nuestro dataset con el ratón. Una vez escogidos los datos podremos ver una gráfica de la función de pérdida en cada época de entrenamiento y una animación de como se transforma el ruido inicial gaussiano en la distribución. Con la opción de dibujar uno puede darse cuenta de la flexibilidad de estos métodos para aprender distribuciones arbitrarias. 

También se incluye una Jupyter Notebook que nos guía en el proceso por si quisiéramos replicar el comportamiento. La arquitectura de la red consiste de una capa de entrada con 3 neuronas (x,y,t), dos capas ocultas de 128 neuronas cada una y una última de salida con dos neuronas (vx,vy). Todas las capas menos la de salida usan la función de activación SiLU (Sigmoid Linear Unit).

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

Una vez entrenado el modelo debemos integrar la ecuación con algún método (_Euler_, _RK4_, etc.) y obtendremos en $t=1$ el dato generado.


Once the model is trained, we must integrate the equation using some method (_Euler_, _RK4_, etc.) and we will obtain the generated datum at $t=1$.
