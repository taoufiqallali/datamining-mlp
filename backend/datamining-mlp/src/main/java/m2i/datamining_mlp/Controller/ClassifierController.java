package m2i.datamining_mlp.Controller;

import m2i.datamining_mlp.DTO.TrainingRequest;
import m2i.datamining_mlp.DTO.TrainingResponse;
import m2i.datamining_mlp.service.ClassifierService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class ClassifierController {

    @Autowired
    private ClassifierService classifierService;

    @PostMapping("/train")
    public ResponseEntity<TrainingResponse> trainModel(
            @RequestParam("hiddenSize") int hiddenSize,
            @RequestParam("learningRate") double learningRate,
            @RequestParam("epochs") int epochs) {

        TrainingRequest request = new TrainingRequest(hiddenSize, learningRate, epochs);
        TrainingResponse response = classifierService.trainModel(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predictEmail(@RequestBody double[] features) {
        Map<String, Object> result = classifierService.predictEmail(features);
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
}
