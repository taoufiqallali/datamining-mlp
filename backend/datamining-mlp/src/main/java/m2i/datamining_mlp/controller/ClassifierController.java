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
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class ClassifierController {

    @Autowired
    private ClassifierService classifierService;

    @PostMapping("/train")
    public ResponseEntity<TrainingResponse> trainModel(@RequestBody TrainingRequest request) {
        TrainingResponse response = classifierService.trainModel(request);
        return ResponseEntity.ok(response);
    }


    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predictEmail(@RequestBody EmailRequest request) {
        double[] features = classifierService.textToFeatureVector(request.getEmail());
        Map<String, Object> result = classifierService.predictEmail(features);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/pretrained-predict")
    public ResponseEntity<Map<String, Object>> predictPretrainedEmail(@RequestBody EmailRequest request) {
        double[] features = classifierService.textToFeatureVector(request.getEmail());
        Map<String, Object> result = classifierService.predictPretrainedEmail(features);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/metrics")
    public ResponseEntity<TrainingResponse.TrainingMetrics> getMetrics() {
        TrainingResponse.TrainingMetrics metrics = classifierService.getLastTrainingMetrics();
        if (metrics == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(metrics);
    }

    @GetMapping("/pretrained-metrics")
    public ResponseEntity<TrainingResponse.TrainingMetrics> getPretrainedMetrics() {
        TrainingResponse.TrainingMetrics metrics = classifierService.getPretrainedMetrics();
        if (metrics == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(metrics);
    }

    @GetMapping("/model-info")
    public ResponseEntity<Map<String, Object>> getModelInfo() {
        Map<String, Object> info = classifierService.getModelInfo();
        return ResponseEntity.ok(info);
    }

    @GetMapping("/pretrained-model-info")
    public ResponseEntity<Map<String, Object>> getPretrainedModelInfo() {
        Map<String, Object> info = classifierService.getPretrainedModelInfo();
        return ResponseEntity.ok(info);
    }

    @GetMapping("/activation-functions")
    public ResponseEntity<String[]> getActivationFunctions() {
        String[] functions = {"SIGMOID", "TANH", "RELU", "LEAKY_RELU"};
        return ResponseEntity.ok(functions);
    }

    @PostMapping("/save-pretrained")
    public ResponseEntity<Map<String, String>> savePretrainedModel() {
        try {
            classifierService.savePretrainedModel();
            return ResponseEntity.ok(Map.of("status", "success", "message", "Pretrained model saved successfully"));
        } catch (IllegalStateException e) {
            return ResponseEntity.badRequest().body(Map.of("status", "error", "message", e.getMessage()));
        }
    }
}
