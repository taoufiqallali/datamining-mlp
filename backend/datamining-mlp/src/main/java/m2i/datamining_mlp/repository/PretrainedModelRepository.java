package m2i.datamining_mlp.repository;

// Importation du modèle PretrainedModel qui représente un modèle entraîné à sauvegarder ou charger depuis MongoDB
import m2i.datamining_mlp.model.PretrainedModel;

// Importation de l'interface MongoRepository fournie par Spring Data MongoDB
import org.springframework.data.mongodb.repository.MongoRepository;

// Interface de dépôt (repository) pour accéder aux données du modèle pré-entraîné dans MongoDB
// Elle hérite de MongoRepository, ce qui fournit automatiquement des méthodes CRUD (Create, Read, Update, Delete)
public interface PretrainedModelRepository extends MongoRepository<PretrainedModel, String> {
    // Aucune méthode personnalisée ici, mais toutes les méthodes standard (findById, save, delete, etc.) sont disponibles
}
