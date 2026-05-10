# Tarea pendiente: dataset sintético (TP1)

## Contexto

En los experimentos se comparan optimizadores (GD vs Adam). Aun así hace falta un **dataset sintético de prueba**, con cierta libertad en los valores elegidos, pero cumpliendo la estructura y restricciones que indica la consigna.

## Requisitos (consigna — `Deep_Learning_TP1_Cohorte_24_2B2026.ipynb`)

### Variable `x`

- Definir `x` como array **unidimensional** con **`n`** valores (`n >= 200`).
- Usar:

  ```text
  x = np.linspace(ini, fin, n)
  ```

- El rango va de **`ini`** a **`fin`**. Se recomienda que sea **simétrico** (mismo valor absoluto, signo opuesto), por ejemplo `ini=-3, fin=3`, `ini=-5, fin=5`, `ini=-10, fin=10`, etc. La elección queda a criterio propio.

### Variable `y` (target)

- `y` es un vector **unidimensional** de tamaño **`n`**.
- Debe seguir la forma:

  ```text
  y = funcion_no_lineal(x) + ruido
  ```

  donde el **patrón no lineal** respecto de `x` lo eligen ustedes y el **ruido** puede generarse con funciones del paquete **`np.random`**.

### Elección de la función no lineal

El patrón puede ser, entre otros:

- trigonométrica,
- exponencial,
- logarítmica,
- sigmoidal,
- **polinómica de grado ≥ 3**,
- o **combinación** de varias de estas (recomendado).

**Importante:** ser prudentes: es una red de **una sola conexión sináptica**; no elegir algo tan complejo que impida que el modelo converja de manera razonable.

## Entregable sugerido en el notebook

- Una celda (o bloque) que defina `n`, `ini`, `fin`, `x`, la función no lineal elegida, el ruido y `y`.
- Breve comentario en español justificando el rango y la forma de `y` (opcional pero útil para la corrección).

---

## Propuesta de solución (combinación no lineal)

### Por qué importa la forma del modelo

El modelo solo puede producir \(\hat{y} = \tanh(wx + b)\): una curva **suave** y **monótona** en \(x\) (crece o decrece, con forma en S). Conviene que el \(y\) sintético **no** pida varias subidas y bajadas fuertes: eso no lo puede ajustar bien una sola \(\tanh\), el error se queda alto y el paisaje de \(J(w,b)\) puede volverse más difícil (más sensibilidad a \(\eta\), mínimos malos), aunque el optimizador “converja” en el sentido de dejar de cambiar mucho.

### Opción recomendada (enriquece sin pasarse)

**Combinación dominante tipo sigmoide + perturbación trigonométrica pequeña:**

\[
y = A\,\tanh(kx) + B\sin(\omega x) + \varepsilon
\]

Parámetros orientativos:

- \(x = \mathrm{linspace}(-4,\,4,\,n)\), con \(n \ge 200\).
- \(A = 1\), \(k\) entre **0.5 y 1**.
- \(B\) **claramente menor** que \(A\) (orden de **0.15–0.25** de \(|A|\)).
- \(\omega\) entre **1 y 2** (evitar \(\omega\) muy grande si el intervalo es ancho).
- Ruido gaussiano moderado: por ejemplo `np.random.randn(n) * sigma` con \(\sigma\) en **0.02–0.08**.

**Por qué sirve para el ejercicio:** mezcla **dos familias** no lineales (acotada tipo sigmoide + oscilación suave), el problema sigue **numéricamente estable**, el MSE suele ser razonablemente suave y se pueden ver diferencias entre GD y Adam sin forzar un target **imposible** para una sola conexión.

### Alternativa simple y defendible

\[
y = a_1 x^3 + a_2 \tanh(c x) + \text{ruido}
\]

Con \(x\) en un rango simétrico acotado (por ejemplo \([-3, 3]\)) y coeficientes no enormes: **cúbica + término acotado**; es fácil de explicar y menos agresiva que un polinomio de alto grado suelto.

### Qué conviene evitar

- \(\sin(\omega x)\) con **\(\omega\) muy grande** o intervalo **muy ancho** (muchas oscilaciones).
- **Exponenciales / log** sin acotar bien \(x\) o con escalas que exploten.
- Combinaciones con **muchas oscilaciones** o **picos**: empujan a un target que una sola \(\tanh\) no puede aproximar y el experimento pasa a ser más “imposible de ajustar” que “comparar optimizadores”.

### Siguiente paso práctico

Fijar números concretos (`ini`, `fin`, `n`, \(A\), \(B\), \(k\), \(\omega\), \(\sigma\)) al implementar la celda en el notebook de entrega.
