#importamos librerias
import re
import unicadedata
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