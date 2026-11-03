package ma.emsi.chibel.tp4webchibel.llm;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.service.AiServices;
import jakarta.enterprise.context.Dependent;

import java.io.Serializable;

/**
 * Classe "métier" liée au LLM choisi (Gemini), qui s'appuie sur LangChain4j.
 * Injectée dans le backing bean (portée view), elle doit donc implémenter Serializable.
 */
@Dependent
public class LlmClient implements Serializable {

    /**
     * Rôle système choisi par l'utilisateur pour l'assistant IA.
     */
    private String systemRole;

    /**
     * Assistant (service IA) dont l'implémentation est fournie par LangChain4j.
     */
    private Assistant assistant;

    /**
     * Mémoire utilisée par l'assistant pour garder l'historique de la conversation.
     */
    private ChatMemory chatMemory;

    public LlmClient() {
        // Récupère la clé secrète de l'API du LLM
        String cle = System.getenv("GEMINI_KEY");
        // Création du modèle
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite") // gemini-2.5-flash n'est plus disponible
                .build();
        // Création de l'assistant, avec une mémoire
        this.chatMemory = MessageWindowChatMemory.withMaxMessages(20);
        this.assistant = AiServices.builder(Assistant.class)
                                   .chatModel(model)
                                   .chatMemory(chatMemory)
                                   .build();
    }

    /**
     * Met le rôle système. La mémoire est vidée car le contexte change,
     * puis le rôle est ajouté à la mémoire pour que le LLM en tienne compte.
     *
     * @param systemRole le rôle système.
     */
    public void setSystemRole(String systemRole) {
        this.systemRole = systemRole;
        this.chatMemory.clear();
        this.chatMemory.add(SystemMessage.from(systemRole));
    }

    /**
     * Envoie une requête au LLM et retourne sa réponse.
     *
     * @param question la question de l'utilisateur.
     * @return la réponse du LLM.
     */
    public String envoyerRequete(String question) {
        return this.assistant.chat(question);
    }
}
