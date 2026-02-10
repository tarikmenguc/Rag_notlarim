from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
#ilk olarak yükledim
loader=WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
# ikinci olarak yüklenen belgeyi parçalara bölüyoruz 
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitted_docs=text_splitter.split_documents(docs)

#üçüncü embedding yapılacak ve kaydedilecek 
vectorestore=Chroma.from_documents(documents=splitted_docs,embedding=OpenAIEmbeddings())

retriever=vectorestore.as_retriever()

# retriver and generation (geri getirme ve cevap üretme için)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#prompt şabolunu

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt=ChatPromptTemplate.from_template(template)

# propmt girmek için model tanımlama

llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)

# rag zincirini kur LangChain Expression Language - LCEL

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What is task decomposition?")
print(result)


# çoklu sorgu (gelişmiş teknik)

from langchain.retrievers.multi_query import MultiQueryRetriever

# Mevcut LLM'i kullanarak soruyu çeşitlendiren bir retriever oluştur
advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorestore.as_retriever(), 
    llm=llm
)

# Artık bu retriever'ı rag_chain içinde 'retriever' yerine kullanabilirsin.

#self rag / adaptive rag  (kendi kendine düzeltme yapan rag)
# burada land graph kullanılır bu en gelişmiş yöntemdir
#süreç şöyle işliyor:
#Arama Yap: İlgili dökümanları getir.
#Dökümanları Puanla: LLM'e sor: "Bu döküman soruyla ilgili mi?"
#Eğer HAYIR ise: İnternetten (Web Search) yeni bilgi ara.
#Cevap Üret: Bilgilerle cevap yaz.
#Cevabı Kontrol Et: "Cevapta yanlış bilgi (halüsinasyon) var mı?"
#Eğer VAR ise: Tekrar başa dön, soruyu düzelt.


# döküman puanlama için kod mantığı 
# ama ondan önce pydantic tanımlanacak bunun sebebi belirli bir
#formatta cevap alamk için 
#bu kodun evet ve hayırı anlaması için şart 

from langchain_core.pydantic_v1 import BaseModel, Field

# LLM'in dökümanı puanlarken uyması gereken şablon
class GradeDocuments(BaseModel):
    """Geri getirilen dökümanın soruyla alakalı olup olmadığını belirleyen ikili puan."""
    binary_score: str = Field(description="Döküman soruyla alakali mi? 'yes' veya 'no'")

# Modeli bu yapıya zorla
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# dokuman puanlama skoru 
# bu fonksiyon dökümanlaro tek tek kontrol ediyor 

def grade_documents(state):
    question = state["question"]
    docs = state["docs"]
    
    filtered_docs=[]
    for d in docs:
        #llm her döküman için yes veya no der 
        score = structured_llm_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        
        if grade == "yes":
            print("---KARAR: DÖKÜMAN ALAKALI---")
            filtered_docs.append(d)
        else:
            print("---KARAR: DÖKÜMAN ALAKASIZ---")
            search_needed = "yes" # Eğer tek bir döküman bile alakasızsa web aramasına işaret koyar
            continue
            
    return {"documents": filtered_docs, "question": question, "web_search": search_needed}

   
   
   
    #karar mekanizması
   
def decide_to_generate(state):
    """
    Üretim mi yapacağız yoksa web araması mı?
    """
    search_needed = state["web_search"]

    if search_needed == "yes":
        # Eğer dökümanlar kötüyse web_search düğümüne git
        print("---KARAR: WEB ARAMASI YAPILACAK---")
        return "transform_query" 
    else:
        # Dökümanlar iyiyse cevap üretme düğümüne git
        print("---KARAR: CEVAP ÜRETİLİYOR---")
        return "generate"

        # LangGraph ile Grafı Kurmak tüm parçaları birleşitrmek için

from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Düğmeleri Tanımla
workflow.add_node("retrieve", retrieve)        # Bilgiyi getir
workflow.add_node("grade_docs", grade_documents) # Bilgiyi puanla
workflow.add_node("generate", generate)        # Cevap yaz
workflow.add_node("transform_query", transform_query) # Soruyu web için iyileştir
workflow.add_node("web_search", web_search)    # İnternette ara

# Akışı (Kenarları) Bağla
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_docs")

# Kritik Nokta: Şartlı Kenar (Conditional Edge)
workflow.add_conditional_edges(
    "grade_docs", # Puanlamadan sonra...
    decide_to_generate, # ...bu karar fonksiyonuna bak
    {
        "transform_query": "transform_query", # Eğer fonksiyon 'transform_query' dönerse oraya git
        "generate": "generate",               # Eğer 'generate' dönerse oraya git
    },
)

workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Grafı Derle
app = workflow.compile()