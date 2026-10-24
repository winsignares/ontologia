
# ============================================================
# UNIDAD 3: REPRESENTACIÓN SEMÁNTICA DEL LENGUAJE
# Proyecto: Ontología para soporte universitario
# Herramientas: Python, RDFLib, Pandas y Scikit-learn
# ============================================================

# ------------------------------------------------------------
# 1. IMPORTAR LAS BIBLIOTECAS
# ------------------------------------------------------------
rdflib
pandas
scikit-learn

import re
import unicodedata

import pandas as pd

from rdflib import Graph, Namespace, RDF, RDFS, Literal
from rdflib.namespace import XSD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# 2. CREAR EL CORPUS
# ------------------------------------------------------------
# Un corpus es un conjunto de textos que utilizamos para
# estudiar o procesar el lenguaje natural.
#
# Cada texto tiene:
# - Un identificador.
# - Una frase escrita por un estudiante.
# - Una intención que representa lo que necesita.
#
# Las intenciones son etiquetas que asignamos manualmente
# para este ejercicio didáctico.

corpus = [
    {
        "id": 1,
        "texto": "Olvidé mi contraseña y no puedo ingresar al campus virtual",
        "intencion": "recuperar_contrasena"
    },
    {
        "id": 2,
        "texto": "Necesito recuperar mi clave para entrar al aula virtual",
        "intencion": "recuperar_contrasena"
    },
    {
        "id": 3,
        "texto": "No recuerdo mi password para iniciar sesión",
        "intencion": "recuperar_contrasena"
    },
    {
        "id": 4,
        "texto": "La plataforma presenta errores y no me permite ingresar",
        "intencion": "reportar_incidente"
    },
    {
        "id": 5,
        "texto": "El campus virtual está fallando",
        "intencion": "reportar_incidente"
    },
    {
        "id": 6,
        "texto": "Quiero reportar un problema técnico del aula virtual",
        "intencion": "reportar_incidente"
    },
    {
        "id": 7,
        "texto": "¿Cuál es la fecha de entrega de mi actividad?",
        "intencion": "consultar_fecha"
    },
    {
        "id": 8,
        "texto": "Necesito conocer el plazo para entregar el trabajo",
        "intencion": "consultar_fecha"
    },
    {
        "id": 9,
        "texto": "¿Cuándo vence la entrega de la tarea?",
        "intencion": "consultar_fecha"
    },
    {
        "id": 10,
        "texto": "Quiero actualizar mi correo institucional",
        "intencion": "actualizar_correo"
    },
    {
        "id": 11,
        "texto": "Necesito cambiar mi correo electrónico",
        "intencion": "actualizar_correo"
    },
    {
        "id": 12,
        "texto": "No necesito recuperar mi contraseña, quiero cambiar mi correo",
        "intencion": "actualizar_correo"
    },
    {
        "id": 13,
        "texto": "Solicito un certificado de estudios",
        "intencion": "solicitar_certificado"
    },
    {
        "id": 14,
        "texto": "Necesito descargar una constancia académica",
        "intencion": "solicitar_certificado"
    },
    {
        "id": 15,
        "texto": "¿Cómo puedo obtener mi certificado universitario?",
        "intencion": "solicitar_certificado"
    }
]


# ------------------------------------------------------------
# 3. VISUALIZAR EL CORPUS
# ------------------------------------------------------------
# Pandas convierte la lista de diccionarios en una tabla.
# Esto facilita revisar los datos antes de procesarlos.

df = pd.DataFrame(corpus)

print("\n========== CORPUS DE SOLICITUDES ==========")
print(df.to_string(index=False))

print("\nCantidad de documentos:", len(df))

print("\nCantidad de ejemplos por intención:")
print(df["intencion"].value_counts())


# ------------------------------------------------------------
# 4. NORMALIZACIÓN Y TOKENIZACIÓN
# ------------------------------------------------------------
# Normalizar: convertir el texto a una forma consistente.
# Tokenizar: separar el texto en unidades o palabras.
#
# Esta función conserva las letras españolas, pero elimina
# signos de puntuación y convierte el texto a minúsculas.

def normalizar(texto):
    texto = texto.lower().strip()

    # Normalizar caracteres Unicode.
    # NFD separa las letras de sus marcas diacríticas.
    texto_unicode = unicodedata.normalize("NFD", texto)

    # Retirar las marcas diacríticas para facilitar
    # algunas comparaciones léxicas.
    texto = "".join(
        caracter
        for caracter in texto_unicode
        if unicodedata.category(caracter) != "Mn"
    )

    # Conservar letras, números y espacios.
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)

    # Eliminar espacios repetidos.
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def tokenizar(texto):
    # Primero normalizamos y después separamos por espacios.
    return normalizar(texto).split()


print("\n========== EJEMPLO DE PLN ==========")

frase = "¡Olvidé mi Contraseña del Campus Virtual!"

print("Texto original:", frase)
print("Texto normalizado:", normalizar(frase))
print("Tokens:", tokenizar(frase))


# ------------------------------------------------------------
# 5. REPRESENTACIÓN SEMÁNTICA BASADA EN REGLAS
# ------------------------------------------------------------
# Para el ejercicio definimos equivalencias léxicas.
# Estas reglas ayudan a relacionar palabras diferentes
# que pueden expresar una necesidad similar.
#
# No son un sistema completo de comprensión del lenguaje.

sinonimos = {
    "clave": "contrasena",
    "password": "contrasena",
    "aula": "campus",
    "plataforma": "campus",
    "fallando": "error",
    "errores": "error",
    "problema": "error",
    "plazo": "fecha",
    "vence": "fecha",
    "tarea": "actividad",
    "trabajo": "actividad",
    "constancia": "certificado"
}


def normalizar_semantico(texto):
    tokens = tokenizar(texto)

    # Reemplazar las palabras por su concepto equivalente.
    tokens_normalizados = [
        sinonimos.get(token, token)
        for token in tokens
    ]

    return tokens_normalizados


print("\n========== NORMALIZACIÓN SEMÁNTICA ==========")

frase = "Necesito recuperar mi clave"

print("Frase original:", frase)
print("Conceptos normalizados:", normalizar_semantico(frase))


# ------------------------------------------------------------
# 6. DEFINIR LAS INTENCIONES Y LOS SERVICIOS
# ------------------------------------------------------------
# Este diccionario representa conocimiento del dominio.
# Cada intención se relaciona con un servicio universitario.

servicios = {
    "recuperar_contrasena": "PortalAutoservicio",
    "reportar_incidente": "MesaDeAyuda",
    "consultar_fecha": "CalendarioAcademico",
    "actualizar_correo": "GestionDeCuenta",
    "solicitar_certificado": "SecretariaAcademica"
}

# Descripciones legibles para explicar las respuestas.
descripciones_servicios = {
    "PortalAutoservicio":
        "Utiliza el portal de autoservicio para restablecer tu contraseña.",
    "MesaDeAyuda":
        "Contacta a la mesa de ayuda para reportar el inconveniente.",
    "CalendarioAcademico":
        "Consulta el calendario académico y las fechas de entrega.",
    "GestionDeCuenta":
        "Utiliza el servicio de gestión de cuenta para actualizar tu correo.",
    "SecretariaAcademica":
        "Consulta con la secretaría académica cómo solicitar tu certificado."
}


# ------------------------------------------------------------
# 7. CREAR LA ONTOLOGÍA RDF
# ------------------------------------------------------------
# RDF representa el conocimiento mediante tripletas:
#
# SUJETO - PREDICADO - OBJETO
#
# Ejemplo:
# RecuperarContrasena - requiereServicio - PortalAutoservicio
#
# El grafo g almacenará todas estas relaciones.

g = Graph()

# Definir el espacio de nombres de la ontología.
# Es un identificador para nuestros conceptos; no hace
# falta que exista una página web en esta dirección.
UNI = Namespace("http://ejemplo.org/universidad#")

# Registrar el prefijo uni para facilitar la lectura.
g.bind("uni", UNI)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)


# ------------------------------------------------------------
# 8. DEFINIR LAS CLASES DE LA ONTOLOGÍA
# ------------------------------------------------------------
# Una clase representa una categoría de entidades.
#
# Solicitud: petición realizada por un estudiante.
# Intencion: necesidad que expresa la solicitud.
# Servicio: recurso que puede atender esa necesidad.
# Usuario: persona que utiliza el sistema.
# Estudiante: tipo específico de usuario.

clases = [
    UNI.Usuario,
    UNI.Estudiante,
    UNI.Solicitud,
    UNI.Intencion,
    UNI.Servicio
]

for clase in clases:
    g.add((clase, RDF.type, RDFS.Class))

# Definir la relación de herencia:
# todo Estudiante pertenece a la clase Usuario.
g.add((
    UNI.Estudiante,
    RDFS.subClassOf,
    UNI.Usuario
))


# ------------------------------------------------------------
# 9. DEFINIR LAS PROPIEDADES SEMÁNTICAS
# ------------------------------------------------------------
# Las propiedades describen relaciones entre entidades.
#
# tieneTexto: una solicitud contiene un texto.
# tieneIntencion: una solicitud expresa una intención.
# requiereServicio: una intención necesita un servicio.

propiedades = [
    UNI.tieneTexto,
    UNI.tieneIntencion,
    UNI.requiereServicio
]

for propiedad in propiedades:
    g.add((propiedad, RDF.type, RDF.Property))


# Documentar el significado de las propiedades.
g.add((
    UNI.tieneTexto,
    RDFS.comment,
    Literal("Texto original de una solicitud.", lang="es")
))

g.add((
    UNI.tieneIntencion,
    RDFS.comment,
    Literal("Intención identificada para una solicitud.", lang="es")
))

g.add((
    UNI.requiereServicio,
    RDFS.comment,
    Literal("Servicio asociado a una intención.", lang="es")
))


# ------------------------------------------------------------
# 10. CREAR LAS INTENCIONES COMO INSTANCIAS
# ------------------------------------------------------------
# Es importante diferenciar:
#
# Clase: Intencion.
# Instancia: recuperar_contrasena.
#
# Las instancias representan conceptos concretos dentro
# de las categorías definidas en la ontología.

for intencion in servicios.keys():
    nombre_concepto = "".join(
        palabra.capitalize()
        for palabra in intencion.split("_")
    )

    concepto = UNI[nombre_concepto]

    g.add((concepto, RDF.type, UNI.Intencion))

    # Registrar el nombre legible de la intención.
    g.add((
        concepto,
        RDFS.label,
        Literal(intencion.replace("_", " "), lang="es")
    ))


# Crear las instancias de servicios.
for nombre_servicio in servicios.values():
    servicio = UNI[nombre_servicio]

    g.add((servicio, RDF.type, UNI.Servicio))

    g.add((
        servicio,
        RDFS.label,
        Literal(nombre_servicio, lang="es")
    ))


# ------------------------------------------------------------
# 11. RELACIONAR LAS INTENCIONES CON LOS SERVICIOS
# ------------------------------------------------------------
# Esta sección introduce conocimiento explícito.
#
# Por ejemplo:
# recuperar_contrasena -> PortalAutoservicio.
#
# La ontología almacena esta relación para consultarla
# posteriormente.

for intencion, nombre_servicio in servicios.items():

    nombre_concepto = "".join(
        palabra.capitalize()
        for palabra in intencion.split("_")
    )

    concepto_intencion = UNI[nombre_concepto]
    concepto_servicio = UNI[nombre_servicio]

    g.add((
        concepto_intencion,
        UNI.requiereServicio,
        concepto_servicio
    ))


# ------------------------------------------------------------
# 12. INCORPORAR LAS SOLICITUDES DEL CORPUS
# ------------------------------------------------------------
# Cada frase será una instancia de Solicitud.
#
# También almacenamos su texto y su intención.
# Esto nos permitirá consultar el conocimiento con SPARQL.

for documento in corpus:

    solicitud = UNI[f"Solicitud_{documento['id']}"]

    nombre_concepto = "".join(
        palabra.capitalize()
        for palabra in documento["intencion"].split("_")
    )

    concepto_intencion = UNI[nombre_concepto]

    # Declarar que el recurso es una solicitud.
    g.add((solicitud, RDF.type, UNI.Solicitud))

    # Almacenar la frase original.
    g.add((
        solicitud,
        UNI.tieneTexto,
        Literal(documento["texto"], lang="es")
    ))

    # Asociar la solicitud con su intención.
    g.add((
        solicitud,
        UNI.tieneIntencion,
        concepto_intencion
    ))


print("\n========== ONTOLOGÍA CONSTRUIDA ==========")
print("Número de tripletas RDF:", len(g))


# ------------------------------------------------------------
# 13. MOSTRAR ALGUNAS TRIPLETAS
# ------------------------------------------------------------
# Las tripletas permiten observar el conocimiento
# representado dentro del grafo.

print("\n========== TRIPLETAS DE LA ONTOLOGÍA ==========")

for sujeto, predicado, objeto in list(g)[:15]:
    print(
        sujeto.n3(g.namespace_manager),
        "->",
        predicado.n3(g.namespace_manager),
        "->",
        objeto.n3(g.namespace_manager)
    )


# ------------------------------------------------------------
# 14. CONSULTAR LA ONTOLOGÍA CON SPARQL
# ------------------------------------------------------------
# SPARQL es un lenguaje de consulta para grafos RDF.
#
# Esta consulta obtiene:
# - El texto de la solicitud.
# - La intención asociada.
# - El servicio requerido por esa intención.

consulta = """
PREFIX uni: <http://ejemplo.org/universidad#>

SELECT ?texto ?intencion ?servicio
WHERE {
    ?solicitud uni:tieneTexto ?texto .
    ?solicitud uni:tieneIntencion ?intencion .
    ?intencion uni:requiereServicio ?servicio .
}
ORDER BY ?solicitud
"""

resultados = g.query(consulta)

print("\n========== CONSULTA SPARQL ==========")

for fila in resultados:
    print("Solicitud:", fila.texto)
    print("Intención:", fila.intencion.split("#")[-1])
    print("Servicio:", fila.servicio.split("#")[-1])
    print("-" * 60)


# ------------------------------------------------------------
# 15. CONSTRUIR REPRESENTACIONES VECTORIALES CON TF-IDF
# ------------------------------------------------------------
# TF-IDF convierte los textos en vectores numéricos.
#
# Esto no es un embedding contextual como BERT.
# Es una representación basada en la importancia de
# las palabras dentro del corpus.
#
# La utilizaremos para comparar frases nuevas con
# las solicitudes conocidas.

textos = [documento["texto"] for documento in corpus]

vectorizador = TfidfVectorizer(
    preprocessor=normalizar,
    ngram_range=(1, 2)
)

matriz_vectores = vectorizador.fit_transform(textos)

print("\n========== REPRESENTACIÓN VECTORIAL ==========")
print("Cantidad de frases:", matriz_vectores.shape[0])
print("Cantidad de características:", matriz_vectores.shape[1])


# ------------------------------------------------------------
# 16. BUSCAR SOLICITUDES SEMÁNTICAMENTE SIMILARES
# ------------------------------------------------------------
# Calculamos la similitud coseno entre una frase nueva
# y las frases que ya tenemos en el corpus.
#
# Un valor más alto indica mayor similitud entre
# los vectores, pero no garantiza que la intención
# sea correcta.

def buscar_similares(frase_nueva, cantidad=3):

    vector_nuevo = vectorizador.transform([frase_nueva])

    similitudes = cosine_similarity(
        vector_nuevo,
        matriz_vectores
    )[0]

    indices = similitudes.argsort()[::-1][:cantidad]

    resultados = []

    for indice in indices:
        resultados.append({
            "texto": corpus[indice]["texto"],
            "intencion": corpus[indice]["intencion"],
            "similitud": float(similitudes[indice])
        })

    return resultados


print("\n========== BÚSQUEDA POR SIMILITUD ==========")

frase_nueva = "No puedo entrar al campus porque olvidé mi clave"

print("Frase nueva:", frase_nueva)

for resultado in buscar_similares(frase_nueva):
    print("\nTexto:", resultado["texto"])
    print("Intención:", resultado["intencion"])
    print("Similitud:", round(resultado["similitud"], 3))


# ------------------------------------------------------------
# 17. REGLAS DE INFERENCIA BÁSICA
# ------------------------------------------------------------
# La inferencia utiliza conocimiento explícito para
# obtener una conclusión.
#
# Aquí consultamos la ontología para averiguar qué
# servicio corresponde a una intención.
#
# No es un motor de inferencia OWL completo: es una
# regla de consulta sobre las relaciones almacenadas.

def inferir_servicio(intencion):

    nombre_concepto = "".join(
        palabra.capitalize()
        for palabra in intencion.split("_")
    )

    concepto_intencion = UNI[nombre_concepto]

    consulta_servicio = """
    PREFIX uni: <http://ejemplo.org/universidad#>

    SELECT ?servicio
    WHERE {
        ?intencion uni:requiereServicio ?servicio .
    }
    """

    resultados = g.query(consulta_servicio)

    for fila in resultados:
        # Verificar que el servicio consultado pertenezca
        # a la intención solicitada.
        if fila.servicio in [
            objeto
            for objeto in g.objects(
                concepto_intencion,
                UNI.requiereServicio
            )
        ]:
            return fila.servicio.split("#")[-1]

    return None


print("\n========== PRUEBA DE INFERENCIA ==========")

intencion_prueba = "recuperar_contrasena"

servicio_inferido = inferir_servicio(intencion_prueba)

print("Intención:", intencion_prueba)
print("Servicio inferido:", servicio_inferido)


# ------------------------------------------------------------
# 18. INTEGRAR LA BÚSQUEDA Y LA ONTOLOGÍA
# ------------------------------------------------------------
# El sistema:
# 1. Recibe una frase.
# 2. Busca ejemplos similares.
# 3. Propone la intención del ejemplo más cercano.
# 4. Consulta la ontología para obtener el servicio.
#
# ADVERTENCIA:
# La similitud por sí sola no comprende bien la negación.
# Por eso verificamos explícitamente el caso didáctico
# en el que se niega la recuperación de contraseña.

def responder_solicitud(frase):

    texto_normalizado = normalizar(frase)

    # Regla específica para demostrar la negación.
    # No sustituye a un analizador lingüístico completo.
    niega_recuperacion = bool(
        re.search(
            r"\bno\s+(?:quiero|necesito)\s+"
            r"(?:recuperar|restablecer)\b",
            texto_normalizado
        )
    )

    if niega_recuperacion:
        # La regla anterior evita interpretar como afirmada
        # una acción que el usuario ha negado.
        # Se continúa buscando la necesidad alternativa.
        if any(
            palabra in texto_normalizado.split()
            for palabra in ["correo", "email", "electronico"]
        ):
            intencion = "actualizar_correo"
        else:
            return (
                "La solicitud contiene una negación. "
                "Se necesita analizar mejor lo que el usuario desea."
            )

    else:
        # Buscar el ejemplo más parecido del corpus.
        candidatos = buscar_similares(frase, cantidad=1)

        if not candidatos or candidatos[0]["similitud"] < 0.10:
            return (
                "No encuentro una solicitud suficientemente similar. "
                "Se recomienda solicitar una aclaración."
            )

        intencion = candidatos[0]["intencion"]

    # Consultar la ontología.
    servicio = inferir_servicio(intencion)

    if servicio is None:
        return (
            "Se identificó una intención, pero no existe "
            "un servicio asociado en la ontología."
        )

    respuesta = descripciones_servicios.get(
        servicio,
        "Consulta con el servicio de atención de la universidad."
    )

    return (
        f"Intención identificada: {intencion}\n"
        f"Servicio asociado: {servicio}\n"
        f"Respuesta sugerida: {respuesta}"
    )


# ------------------------------------------------------------
# 19. PROBAR EL SISTEMA INTEGRADO
# ------------------------------------------------------------

pruebas = [
    "Olvidé mi clave y no puedo entrar al campus",
    "El aula virtual presenta un error",
    "¿Cuándo vence mi tarea?",
    "Quiero cambiar mi correo institucional",
    "Necesito descargar un certificado de estudios",
    "No necesito recuperar mi contraseña, quiero cambiar mi correo"
]

print("\n========== SISTEMA INTEGRADO ==========")

for frase in pruebas:
    print("\nUsuario:", frase)
    print(responder_solicitud(frase))
    print("-" * 60)


# ------------------------------------------------------------
# 20. EXPORTAR LA ONTOLOGÍA A UN ARCHIVO RDF
# ------------------------------------------------------------
# Esta sección es opcional.
#
# Guarda el grafo en formato Turtle para que puedas
# inspeccionarlo o reutilizarlo posteriormente.
# El archivo se crea en la carpeta desde la que ejecutas
# main.py, no es necesario descargarlo previamente.

g.serialize(destination="ontologia_universidad.ttl", format="turtle")

print("\nOntología exportada a ontologia_universidad.ttl")
print("Ejecución finalizada.")
