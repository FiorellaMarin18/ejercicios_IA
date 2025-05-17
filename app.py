from transformers import pipeline

print("#####################PRIMER PROCESO##########################")
# Creamos el pipeline de pregunta-respuesta
nlp_qa = pipeline("question-answering" )

# Pedimos al usuario que escriba el contexto
contexto = input("Escribe el contexto del texto (lo que has leído):\n")

# Pregunta predefinida
pregunta = "¿De quién se está hablando en el texto?"

# Usamos el pipeline con la pregunta y el contexto proporcionado
respuesta = nlp_qa(question=pregunta, context=contexto)

# Mostramos la respuesta
print("\nRespuesta encontrada:")
print(respuesta['answer']) 

print("#####################FINALIZA PROCESO##########################")

#SEGUNDO PROCESO 

print("#####################SEGUNDO PROCESO##########################")

nlp_generacion = pipeline(
    "text-generation",
    model="gpt2-medium",
    device_map="auto",
    trust_remote_code=True
)

texto_generado = nlp_generacion(contexto, max_length=500, do_sample=True, temperature=0.7)

print("\nTexto generado:")
print(texto_generado[0]['generated_text'])

print("#####################FINALIZA PROCESO##########################")


# TERCER PROCESO

print("#####################TERCER PROCESO##########################")

from pysentimiento import create_analyzer

# Cargamos el analizador de sentimiento
analyzer = create_analyzer(task="sentiment", lang="es")

# Analizamos el contexto
resultado = analyzer.predict(contexto)

print("\nAnálisis de sentimiento:")
print(f"Etiqueta: {resultado.output}")
print(f"Probabilidades: {resultado.probas}")


print("#####################FINALIZA PROCESO##########################")

#CUARTO PROCESO

print("#####################CUARTO PROCESO##########################")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

modelo_resumen = "csebuetnlp/mT5_multilingual_XLSum"

tokenizer = AutoTokenizer.from_pretrained(modelo_resumen, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(modelo_resumen)
model.to("cpu")  # Mover explícitamente a CPU

nlp_resumen = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=-1  # -1 para CPU, 0 para GPU
)

resumen = nlp_resumen(contexto, max_length=50, min_length=10, do_sample=False)
print(resumen)

print("#####################FINALIZA PROCESO##########################")