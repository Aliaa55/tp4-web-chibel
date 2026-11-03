package ma.emsi.chibel.tp4webchibel.llm;

/**
 * Service IA : interactions de l'application avec le LLM.
 * L'implémentation est fournie par LangChain4j (AiServices).
 */
public interface Assistant {
  String chat(String prompt);
}
