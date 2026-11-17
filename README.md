# Predicción de impago de crédito (Give Me Some Credit)

Trabajo de evaluación continua de **Machine Learning [P375]**, CUNEF Universidad. Entrega inicial: descripción del problema, fuente de datos, formulación científica y análisis exploratorio (EDA).

**Repositorio:** https://github.com/CarlosCUNEF/machine_learning_assignment

## Autores

| Nombre completo | Email CUNEF |
|---|---|
| Alejandro Alcázar *(completar apellidos si procede)* | alejandro.alcazar@cunef.edu |
| Grego *(completar nombre completo)* | *(completar email CUNEF)* |
| Carlos *(completar nombre completo)* | *(completar email CUNEF)* |

## 1. Descripción del problema de negocio

### Por qué importa

Conceder crédito es el negocio central de un banco, y también su principal riesgo. Cada préstamo que no se devuelve es una pérdida directa, y además obliga a la entidad a inmovilizar capital en provisiones. Cada decisión de concesión es, en el fondo, una apuesta: si el banco es demasiado permisivo, aumentan los impagos; si es demasiado restrictivo, rechaza a clientes que sí habrían pagado y pierde ingresos.

El problema también tiene una cara social. Un buen sistema de scoring permite dar crédito a personas con poco historial pero buen perfil de pago, que con criterios muy rígidos quedarían excluidas. Mejorar la predicción del impago mejora a la vez la rentabilidad de la entidad y el acceso al crédito.

### Carencias de las soluciones actuales

- **Reglas fijas** (por ejemplo, "se rechaza si la deuda supera X veces el ingreso"): son fáciles de explicar, pero tratan cada variable por separado y no capturan interacciones entre ellas. Los umbrales se fijan a mano y envejecen mal.
- **Scorecards tradicionales** (regresión logística sobre variables discretizadas): son el estándar del sector y muy interpretables, pero asumen relaciones aproximadamente lineales y aditivas, y tienen dificultades con relaciones no lineales y con valores extremos o ausentes.
- En ambos casos, la mejora de la capacidad predictiva suele quedar limitada por el diseño manual de variables y reglas.

Los modelos de Machine Learning pueden aprender relaciones no lineales e interacciones directamente de los datos. Su reto es mantener un nivel de interpretabilidad aceptable, algo que el regulador exige en decisiones de crédito.

**Decisión de negocio que apoya el modelo:** a quién conceder crédito y en qué condiciones, estimando la probabilidad de que un cliente tenga un impago grave en los próximos dos años.

## 2. Fuente del dataset

- **Dataset:** *Give Me Some Credit*, competición de Kaggle de 2011.
- **Enlace:** https://www.kaggle.com/competitions/GiveMeSomeCredit
- **Fichero utilizado:** `cs-training.csv` (unas 150.000 filas). El fichero `cs-test.csv` no incluye etiquetas, por lo que no sirve para evaluar y el reparto en entrenamiento y test lo haremos nosotros.

### Ventajas

- Son datos reales, con errores, nulos y valores extremos reales, lo que da contenido a la EDA.
- El tamaño es manejable.
- Todas las variables son numéricas, así que cualquier algoritmo del temario se aplica sin fricción.
- El problema de negocio es claro y encaja con el perfil de CUNEF.
- El desbalance de clases (en torno al 7% de positivos) permite justificar bien la elección de métricas.

### Inconvenientes

- Solo tiene 10 variables, lo que limita el feature engineering de la entrega final.
- No hay variables categóricas.
- El origen es opaco: no se conoce la entidad, el país ni las fechas.
- Los datos son de 2011, así que el comportamiento de los clientes puede haber cambiado.
- Es un dataset muy trabajado en Kaggle, por lo que hay poca novedad.
- El fichero de test no tiene etiquetas.

## 3. Estructura del repositorio

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/            # vacío en el repo: aquí se coloca cs-training.csv
└── notebooks/
    └── 01_EDA.ipynb
```

Para reproducir el análisis, descargar `cs-training.csv` de Kaggle y guardarlo en `data/`. Esa carpeta está en `.gitignore`, por lo que los datos no se suben al repositorio.

<!-- Secciones pendientes de los siguientes turnos:
     - Formulación científica (Grego)
     - Resumen de la EDA con conclusiones y acciones (Carlos) -->
