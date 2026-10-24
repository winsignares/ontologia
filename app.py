#importamos librerias
import re
import unicodedata
import pandas as pd

from rdflib import Graph, Namespace, RDF, RDFS, Literal
from rdflib.namespace import XSD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#creamos nuestro corpus

corpus = [
    {
        "id": "Q1",
        "texto": "No puedo ingresar al campus virtual porque olvidé mi contraseña",
        "intencion": "restablecer_contraseña",
    },
    {
        "id": "Q2",
        "texto": "No puedo ingresar al campus virtual porque mi usuario aparece bloqueado",
        "intencion": "desbloquear_usuario",
    },
    {
        "id": "Q3",
        "texto": "Necesito cambiar mi contraseña para proteger mi cuenta",
        "intencion": "cambiar_contraseña",
    },
    {
        "id": "Q4",
        "texto": "No puedo ver las materias que tengo matriculadas en el campus virtual",
        "intencion": "consultar_materias",
    },
    {
        "id": "Q5",
        "texto": "La plataforma no me permite subir las actividades de una asignatura",
        "intencion": "solucionar_error_entrega",
    },
    {
        "id": "Q6",
        "texto": "Olvidé mi nombre de usuario y no puedo iniciar sesión",
        "intencion": "recuperar_usuario",
    },
    {
        "id": "Q7",
        "texto": "No puedo acceder a las calificaciones de mis asignaturas",
        "intencion": "consultar_calificaciones",
    },
    {
        "id": "Q8",
        "texto": "El campus virtual presenta errores y no carga los contenidos de mis cursos",
        "intencion": "reportar_falla_plataforma",
    }
]

#3. Visualizamos el corpus

df = pd.DataFrame(corpus)

print(df.to_string(index=False))

print(f"Cantidad de datos {len(df)}")

print(f"Cantidad de intenciones {df['intencion'].value_counts()}")

#4. Tokenización y Normalización

def normalizar(texto):
    texto = texto.lower().strip()
    #noralizamos los caracteres con Unicode y NFD
    texto_unicode = unicodedata.normalize("NFD",texto)
    texto = "".join(
        caracter
        for caracter in texto_unicode
        if unicodedata.category(caracter)!="Mn"
    )

    #conservamos las letras, numeros y espacios
    texto = re.sub(r"[^a-z0-9\s]"," ",texto)
    #eliminamos espacio repetidos
    texto = re.sub(r"\s+"," ",texto).strip()

    return texto

def tokenizar(texto):
    #primero normalizamos el texto y despues lo separamos por espacio 
    return normalizar(texto).split()

print("Ejemplo de PLN")

frase = "Olvidé mi contraseña de usuario y no puedo iniciar sesión"

print("Texto original ", frase)
print("Texto normalizado ", normalizar(frase))
print("tokens ",tokenizar(frase))

#5. Creamos nuestra Ontologia con el RDF

g = Graph()

UNI = Namespace("http://ejmplo.org/universidad#")

#creamos un prefijo uni para facilitar la lectura

g.bind("uni",UNI)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)

#creamos una clase para representar las diferentes entidades 
#Solicitud, Intencio, Servicio, Usuario, Estudiante,

clases = [
    UNI.Usuario,
    UNI.Estudiante,
    UNI.Solicitud,
    UNI.Intencion,
    UNI.Servicio
]

for clase in clases:
    g.add((clase, RDF.type, RDFS.Class))

#Defineme las relaciones entre herencia
#todo Estudiantes pertenece a la clase Usuario

g.add((UNI.Estudiante,RDFS.subClassOf, UNI.Usuario))

#Definimos las propiedades Semanticas

propiedades = [
        UNI.tieneTexto,
        UNI.tieneIntencion,
        UNI.requiereServicio
]

for propiedad in propiedades:
    g.add((propiedad, RDF.type, RDF.Property))

#Documentamos el significado de la parte ontologica con sus propiedades

g.add((
    UNI.tieneTexto,
    RDFS.comment,
    Literal("Texto original de una solicitud.", lang="es")
))

g.add((
    UNI.tieneIntencion,
    RDFS.comment,
    Literal("Intencion identifica para una solicitud.", lang="es")
))

g.add((
    UNI.requiereServicio,
    RDFS.comment,
    Literal("Servicio asociado a una intencion.", lang="es")
))

#Relacionar las intenciones con el servicio

servicios = {
    "recuperar_contraseña":"PortalAutoservicio",
    "reportar_incidente": "MesaDeAyuda",
    "consultar_fecha": "CalendarioAcademico",
    "actualizar_correo":"GestionDeCuenta",
    "solicitar_certificado": "SecretariaAcademica"
}
descripcion_servicios = {
    "PortalAutoservicio": "Utiliza el portal autoservicio para restablecer tu contraseña",
    "MesaDeAyuda": "Contacta a la mesa de ayuda para reportar el incidente",
    "CalendarioAcademico": "Consulta calendario academico y saber las fechas de entrega",
    "GestionDeCuenta": "Utiliza el servicio de gestion de cuenta para actualizar tus datos o correo",
    "SecretariaAcademica": "Consulta con la secretaria academica como solicitar certificado"
}


for intencion in servicios.keys():
    nombre_concepto = "".join(
        palabra.capitalize()
        for palabra in intencion.split("_")
    )
    concepto = UNI[nombre_concepto]
    g.add((concepto,RDF.type, UNI.Intencion))

    g.add((
        concepto, 
        RDFS.label, 
        Literal(intencion.replace("_"," "), lang="es")
        ))

#Creamos la instancia con la parte de servicio

for nombre_servicio in servicios.values():
    servicio = UNI[nombre_servicio]
    g.add((servicio,RDF.type, UNI.Servicio))

    g.add((
            servicio, 
            RDFS.label, 
            Literal(nombre_servicio, lang="es")
        ))

#Relacionar las intenciones con el Servicio

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

#incorporamos las solicitudes al corpus


for documento in corpus:
    solicitud = UNI[f"Solicitud_{documento['id']}"]

    nombre_concepto = "".join(
        palabra.capitalize()
        for palabra in documento["intencion"].split("_")
    )

    concepto_intencion = UNI[nombre_concepto]

    #Declara que el recurso es una solicitud

    g.add((solicitud,RDF.type, UNI.Solicitud ))

    #almacenar la frase original 
    g.add((
        solicitud, 
        UNI.tieneIntencion,
        concepto_intencion
    ))

print("******Ontologia Construida*******")
print("Numero de tripletas RDF:",len(g))