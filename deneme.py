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

# sadece bu kadar değil 

# sorgu dönüştürme (query translation) "soruyu daha iyi sormak"
#bazen kullanıcının sorusu veritabanındaki terimlerle eşleşmez 
# bunun için 3 teknik kullanılır

# 1 Rag fusion: soruyu 3-4 farklı şekilde sorar hepsinden sonuçları toplar RRF adlı bir algo 
# kullanarak en ortak ve kaliteli olanı seçer 

#2 step-back prompting : eğer soru çok zorsa llm önce  bu sorunun temelindeki genel kavram nedir diye sorar
# önce genel kavramı sonra  

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Soruyu çoğaltan Prompt
multi_query_prompt = ChatPromptTemplate.from_template("""You are an AI language assistant. 
Your task is to generate 3 different versions of the given user question to retrieve relevant documents.
Original question: {question}""")

llm = ChatOpenAI(temperature=0)

# 2. Zincir: Soru -> 3 Yeni Soru
generate_queries = (
    multi_query_prompt 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n")) # Satırlara bölerek liste yap
)

#3. HyDE
#MANTIK: Soru -> LLM "Hayali bir cevap" yazar -> Bu hayali cevapla arama yapılır.
# Neden? Çünkü hayali cevap, veritabanındaki dökümanlara soru cümlesinden daha çok benzer.

# MANTIK: Soru -> LLM (Uydurma Döküman) -> Vektör Arama
hyde_prompt = ChatPromptTemplate.from_template("""Please write a scientific paper passage 
to answer this question: {question}""")

generate_docs = hyde_prompt | llm | StrOutputParser()

# Örnek Kullanım:
question = "What is task decomposition?"
hypothetical_doc = generate_docs.invoke({"question": question})

# Şimdi soruyu değil, LLM'in yazdığı bu uzun metni aratıyoruz:
docs = retriever.get_relevant_documents(hypothetical_doc)

#------------------------

# 2 yönlendirme (routing)
# her soru veritabanında olmayabilir mantıksal ve semantik(anlamsal) yönlendirme yapılabilir 
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field

# LLM'in seçeceği seçenekler
class RouteQuery(BaseModel):
    """Sorguyu en alakalı veri deposuna yönlendir."""
    datasource: Literal["python_docs", "js_docs", "general_web"] = Field(
        ..., description="Kullanıcı sorusuna göre gidilecek depo."
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

# Karar anı:
question = "How to use decorators in Python?"
source = structured_llm_router.invoke(question)

if source.datasource == "python_docs":
    print("Python veritabanına gidiliyor...")
# Mantıksal Yönlendirme Örneği:
def router(state):
    question = state["question"]
    # LLM karar verir: Bu bir kod sorusu mu yoksa dökümantasyon sorusu mu?
    if "python" in question or "js" in question:
        return "code_database"
    else:
        return "vectorstore"


# 3 sorgu yapılandırma (qoery structuring)
#Kullanıcı "2023'ten sonra yayınlanan makaleleri getir" dediğinde, standart RAG bunu yapamaz çünkü "2023" bir kelime değil bir filtredir.
#Self-Querying :
#LLM, soruyu ikiye böler:
#Arama terimi: "Makaleler"
#Filtre: {"year": {"$gt": 2023}}


# gelişmiş indexing (veriyi daha akıllı saklamak )
#RAG'ın performansını %50 artıran iki yöntemden bahsediliyor:
#Multi-representation (Çoklu Temsil):
#Arama yapmak için küçük parçalar (chunks) kullanılır (çünkü daha isabetli sonuç verir).
#Cevap üretirken LLM'e bu küçük parçaların ait olduğu ana döküman (parent document) gönderilir (çünkü daha fazla bağlam sağlar).
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore

# 1. Ana dökümanları tutan depo
store = InMemoryStore()
id_key = "doc_id"

# 2. Retriever'ı kur
retriever = MultiVectorRetriever(
    vectorstore=vectorstore, # Küçük parçalar (çocuklar) burada
    docstore=store,          # Büyük dökümanlar (babalar) burada
    id_key=id_key,
)
 
# Bir dökümanın özetini çıkarıyorsun (summary), 
# özeti vektör deposuna, dökümanın kendisini döküman deposuna atıyor.

#RAPTOR (Hiyerarşik Dizinleme):
#Dökümanlar önce özetlenir, sonra o özetlerin de özeti çıkarılır (bir ağaç yapısı gibi).
#Böylece "Bu şirketin genel stratejisi nedir?" gibi çok geniş sorulara cevap verebilir.
# MANTIK:
# Dökümanlar -> Kümele (K-Means) -> Her kümeyi özetle (Summary)
# Özetleri de kümele -> Tekrar özetle (Üst seviye özet)

# Kodda bu genellikle bir döngü ile yapılır:
def recursive_summarize(docs, level=0):
    # Dökümanları gruplandır ve her gruba bir "özet" yazdır.
    # Bu özetleri yeni dökümanlar olarak sisteme ekle.
    pass



# ColBERT (Embedding'leri Arama için Kullanmak)
#Normalde biz soruyu vektöre çeviririz, veritabanındaki vektörlerle karşılaştırırız.
#ColBERT ise soruyu ve veritabanındaki her kelimeyi ayrı ayrı vektöre çevirir.
#Sonra "Matris Çarpımı" yaparak en uyumlu kelimeleri bulur.
#Avantajı: Çok daha hassas eşleşme sağlar


#LangSmith: İzleme ve Debugging
# LangSmith, LangChain uygulamalarını test etmek, izlemek ve analiz etmek için kullanılan bir platformdur.
# LangGraph akışlarını görselleştirmenizi ve her adımda neler olduğunu görmenizi sağlar.
# Ayrıca, LLM çağrılarını, retrieved dokümanları ve zincir performansını analiz edebilirsiniz.
#Neden Önemli? 
#Yazdığın RAG kodunda "Neden yanlış cevap verdi?" sorusunu anlamak için her adımın (retrieve ne getirdi?,
# LLM ne düşündü?) kaydını tutar.
#kodun çalışmasını izlediği o ekran için kod yazmıyorsun, sadece ortam değişkenlerini ayarlıyorsun:
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "senin_anahtarın"
os.environ["LANGCHAIN_PROJECT"] = "RAG_Sifirdan"



# yeniden sıralama (re-ranking)
#ektör araması (semantic search) bazen ilk 10 sonuç içine alakasız şeyler karıştırabilir. 
#En doğru döküman 7. sırada olabilir. Onu nasıl 1. sıraya taşırız?
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# 1. Önce normal arama yap (Retriever)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 2. Yeniden sıralayıcıyı (Re-ranker) ekle
# Bu model dökümanları anlamca sorguya göre tekrar dizer
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

# Kullanım
compressed_docs = compression_retriever.get_relevant_documents("Soru buraya")


#Multi-Vector Retrieval (Parent Document Retriever)
#dökümanları parçalara bölmenin (chunking) bir paradoks yaratabilir:
#Küçük parçalar: Arama için iyidir (vektörler daha net olur).
#Büyük parçalar: LLM'in cevabı anlaması için iyidir (bağlam kaybolmaz).
#Çözüm: Küçük parçalarla arama yap, ama LLM'e o parçanın ait olduğu ana dökümanı gönder.

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# 1. Küçük (child) ve Büyük (parent) parçalayıcıları tanımla
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# 2. Vektör deposu (küçükler için) ve döküman deposu (büyükler için)
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)