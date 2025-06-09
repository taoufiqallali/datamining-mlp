package m2i.datamining_mlp.controller;

import m2i.datamining_mlp.DTO.EmailRequest;
import m2i.datamining_mlp.DTO.TrainingRequest;
import m2i.datamining_mlp.DTO.TrainingResponse;
import m2i.datamining_mlp.service.ClassifierService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api") // Définit la racine des URLs pour ce contrôleur
@CrossOrigin(origins = "*") // Permet les requêtes CORS depuis n'importe quelle origine
public class ClassifierController {

    @Autowired // Injection automatique du service ClassifierService
    private ClassifierService classifierService;

    @PostMapping("/train") // Point d’accès POST pour entraîner le modèle
    public ResponseEntity<TrainingResponse> trainModel(@RequestBody TrainingRequest request) {
        TrainingResponse response = classifierService.trainModel(request); // Appelle le service pour entraîner
        return ResponseEntity.ok(response); // Retourne la réponse HTTP 200 avec le résultat
    }


    @PostMapping("/predict") // Point d’accès POST pour prédire à partir d’un email
    public ResponseEntity<Map<String, Object>> predictEmail(@RequestBody EmailRequest request) {
        double[] features = classifierService.textToFeatureVector(request.getEmail()); // Convertit le texte en vecteur de caractéristiques
        Map<String, Object> result = classifierService.predictEmail(features); // Prédit si spam ou pas
        return ResponseEntity.ok(result); // Retourne le résultat en JSON
    }

    @PostMapping("/pretrained-predict") // Point d’accès POST pour prédiction avec modèle pré-entraîné
    public ResponseEntity<Map<String, Object>> predictPretrainedEmail(@RequestBody EmailRequest request) {
        double[] features = classifierService.textToFeatureVector(request.getEmail()); // Convertit le texte en vecteur
        Map<String, Object> result = classifierService.predictPretrainedEmail(features); // Prédit avec modèle pré-entraîné
        return ResponseEntity.ok(result); // Retourne le résultat
    }

    @GetMapping("/metrics") // Point d’accès GET pour obtenir les métriques du dernier entraînement
    public ResponseEntity<TrainingResponse.TrainingMetrics> getMetrics() {
        TrainingResponse.TrainingMetrics metrics = classifierService.getLastTrainingMetrics(); // Récupère les métriques
        if (metrics == null) {
            return ResponseEntity.notFound().build(); // Si aucune métrique, retourne 404
        }
        return ResponseEntity.ok(metrics); // Sinon retourne les métriques
    }

    @GetMapping("/pretrained-metrics") // Point d’accès GET pour métriques du modèle pré-entraîné
    public ResponseEntity<TrainingResponse.TrainingMetrics> getPretrainedMetrics() {
        TrainingResponse.TrainingMetrics metrics = classifierService.getPretrainedMetrics(); // Récupère métriques pré-entraînées
        if (metrics == null) {
            return ResponseEntity.notFound().build(); // Retourne 404 si pas trouvées
        }
        return ResponseEntity.ok(metrics); // Sinon retourne les métriques
    }

    @GetMapping("/model-info") // Point d’accès GET pour infos du modèle actuel
    public ResponseEntity<Map<String, Object>> getModelInfo() {
        Map<String, Object> info = classifierService.getModelInfo(); // Récupère les infos
        return ResponseEntity.ok(info); // Retourne infos en JSON
    }

    @GetMapping("/pretrained-model-info") // Point d’accès GET pour infos du modèle pré-entraîné
    public ResponseEntity<Map<String, Object>> getPretrainedModelInfo() {
        Map<String, Object> info = classifierService.getPretrainedModelInfo(); // Récupère infos pré-entraînées
        return ResponseEntity.ok(info); // Retourne infos
    }

    @GetMapping("/activation-functions") // Point d’accès GET pour récupérer les fonctions d’activation supportées
    public ResponseEntity<String[]> getActivationFunctions() {
        String[] functions = {"SIGMOID", "TANH", "RELU", "LEAKY_RELU"}; // Liste des fonctions d’activation
        return ResponseEntity.ok(functions); // Retourne la liste
    }

    @PostMapping("/save-pretrained") // Point d’accès POST pour sauvegarder le modèle pré-entraîné
    public ResponseEntity<Map<String, String>> savePretrainedModel() {
        try {
            classifierService.savePretrainedModel(); // Sauvegarde le modèle
            return ResponseEntity.ok(Map.of("status", "success", "message", "Pretrained model saved successfully")); // Succès
        } catch (IllegalStateException e) {
            return ResponseEntity.badRequest().body(Map.of("status", "error", "message", e.getMessage())); // Erreur en cas d’exception
        }
    }
}
