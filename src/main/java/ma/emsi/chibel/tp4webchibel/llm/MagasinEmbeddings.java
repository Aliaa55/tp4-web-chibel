package ma.emsi.chibel.tp4webchibel.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;

/**
 * Phase 1 du RAG (ingestion des documents), effectuée une seule fois pour toute l'application :
 * les embeddings des 2 fichiers PDF sont enregistrés dans 2 EmbeddingStores différents.
 * Fournit les ContentRetrievers utilisés pendant la phase 2 du RAG.
 */
@ApplicationScoped
public class MagasinEmbeddings {

    private ContentRetriever contentRetrieverRag;
    private ContentRetriever contentRetrieverCouscous;

    @PostConstruct
    public void init() {
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        EmbeddingStore<TextSegment> embeddingStoreRag = creerEmbeddingStore("rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> embeddingStoreCouscous = creerEmbeddingStore("couscous.pdf", embeddingModel);
        this.contentRetrieverRag = creerContentRetriever(embeddingStoreRag, embeddingModel);
        this.contentRetrieverCouscous = creerContentRetriever(embeddingStoreCouscous, embeddingModel);
    }

    public ContentRetriever getContentRetrieverRag() {
        return contentRetrieverRag;
    }

    public ContentRetriever getContentRetrieverCouscous() {
        return contentRetrieverCouscous;
    }

    // Chargement d'un document, découpage en segments, calcul des embeddings
    // et enregistrement dans un magasin d'embeddings en mémoire
    private EmbeddingStore<TextSegment> creerEmbeddingStore(String nomFichier, EmbeddingModel embeddingModel) {
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, documentParser);
        List<TextSegment> segments = DocumentSplitters.recursive(300, 30).split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }

    // ContentRetriever qui retrouve les 2 segments les plus pertinents, avec un score d'au moins 0,5
    private ContentRetriever creerContentRetriever(EmbeddingStore<TextSegment> embeddingStore,
                                                   EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }
}
