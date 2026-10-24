
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# PUNTO 1. Exploración inicial del dataset
df = pd.read_csv("datos_para_limpiar.csv")

print("=== Primeras 5 filas ===")
print(df.head())

print("\n=== Información general ===")
df.info()

print("\n=== Valores nulos por columna ===")
print(df.isnull().sum())


# PUNTO 2. Eliminación de columnas innecesarias
df = df.drop(columns=["ID", "Notas"])


# PUNTO 3. Normalización de nombres de columnas
df.columns = (
    df.columns.str.strip().str.lower().str.replace(" ", "_")
    .str.replace("á", "a").str.replace("é", "e")
    .str.replace("í", "i").str.replace("ó", "o")
    .str.replace("ú", "u")
)


# PUNTO 4. Corrección de errores de escritura en ciudad
df.replace(["?", "N/A", "n/a", "--", "none"], pd.NA, inplace=True)

df["ciudad"] = df["ciudad"].str.strip().str.lower()
df["ciudad"] = df["ciudad"].replace({
    "bogota": "Bogotá",
    "bogotá": "Bogotá",
    "medellin": "Medellín",
    "medellín": "Medellín",
    "cali": "Cali"
})

df["genero"] = df["genero"].str.strip().str.lower()


# PUNTO 5. Tratamiento de valores nulos
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

edades_validas = df[df["edad"].between(0, 120)]["edad"]
df["edad"] = df["edad"].fillna(edades_validas.mean())

df["salario"] = pd.to_numeric(df["salario"], errors="coerce")
df["salario"] = df["salario"].fillna(df["salario"].median())

df["ciudad"] = df["ciudad"].fillna(df["ciudad"].mode()[0])
df["genero"] = df["genero"].fillna(df["genero"].mode()[0])


# PUNTO 6. Asegurar los tipos de datos correctos
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df["ingresos"] = pd.to_numeric(df["ingresos"], errors="coerce")
df["gastos"] = pd.to_numeric(df["gastos"], errors="coerce")

print("\n=== Tipos de datos ===")
print(df[["edad", "fecha", "salario"]].dtypes)


# PUNTO 7. Filtrado de datos fuera de rango
filas_antes = len(df)

df = df[df["edad"].between(0, 120)]
df = df[df["salario"] <= 1_000_000_000].copy()

print("\n=== Registros eliminados ===")
print(filas_antes - len(df))
print("Registros restantes:", len(df))


# PUNTO 8. Conversión de la variable ordinal nivel
orden = {"bajo": 1, "medio": 2, "alto": 3}
df["nivel"] = df["nivel"].str.strip().str.lower().map(orden)

print("\n=== Valores de nivel ===")
print(df["nivel"].value_counts(dropna=False))


# PUNTO 9. Codificación de las variables categóricas
df = pd.get_dummies(
    df,
    columns=["genero", "ciudad"],
    drop_first=True,
    dtype=int
)

print("\n=== Columnas después de la codificación ===")
print(df.columns.tolist())


# PUNTO 10. Escalado de las variables edad e ingresos
ingresos_originales = df["ingresos"].copy()

scaler = MinMaxScaler()
df[["edad", "ingresos"]] = scaler.fit_transform(
    df[["edad", "ingresos"]]
)


# PUNTO 11. Eliminación de registros duplicados
filas_antes = len(df)
df = df.drop_duplicates()

print("\n=== Duplicados eliminados ===")
print(filas_antes - len(df))


# PUNTO 12. Creación de la columna utilidad
df["utilidad"] = df["ingresos"] - df["gastos"]


# PUNTO 13. Verificación y guardado del dataset limpio
print("\n=== Información del dataset final ===")
df.info()

print("\n=== Resumen de datos numéricos ===")
print(df.describe())

print("\n=== Primeras filas del dataset limpio ===")
print(df.head())

df.to_csv("datos_limpios.csv", index=False)

print("\nArchivo datos_limpios.csv guardado correctamente")
