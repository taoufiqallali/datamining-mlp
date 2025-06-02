package m2i.datamining_mlp.Controller;

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

    /**
     * Train model with configurable hidden layers and activation function
     * @param request Training configuration including hidden layer sizes and activation function
     * @return Training response with metrics
     */
    @PostMapping("/train")
    public ResponseEntity<TrainingResponse> trainModel(@RequestBody TrainingRequest request) {
        TrainingResponse response = classifierService.trainModel(request);
        return ResponseEntity.ok(response);
    }

    /**
     * Alternative training endpoint with URL parameters (for simple configurations)
     * @param hiddenSizes Comma-separated list of hidden layer sizes (e.g., "10,5" for two layers)
     * @param activationFunction Activation function name (SIGMOID, TANH, RELU, LEAKY_RELU)
     * @param learningRate Learning rate
     * @param epochs Number of training epochs
     * @return Training response
     */
    @PostMapping("/train-simple")
    public ResponseEntity<TrainingResponse> trainModelSimple(
            @RequestParam("hiddenSizes") String hiddenSizes,
            @RequestParam("activationFunction") String activationFunction,
            @RequestParam("learningRate") double learningRate,
            @RequestParam("epochs") int epochs) {

        try {
            // Parse hidden layer sizes from comma-separated string
            String[] sizeStrings = hiddenSizes.split(",");
            int[] hiddenLayerSizes = new int[sizeStrings.length];

            for (int i = 0; i < sizeStrings.length; i++) {
                hiddenLayerSizes[i] = Integer.parseInt(sizeStrings[i].trim());
                if (hiddenLayerSizes[i] <= 0) {
                    TrainingResponse errorResponse = new TrainingResponse();
                    errorResponse.setStatus("error");
                    errorResponse.setMessage("All hidden layer sizes must be positive integers");
                    return ResponseEntity.badRequest().body(errorResponse);
                }
            }

            TrainingRequest request = new TrainingRequest(hiddenLayerSizes, activationFunction, learningRate, epochs);
            TrainingResponse response = classifierService.trainModel(request);
            return ResponseEntity.ok(response);

        } catch (NumberFormatException e) {
            TrainingResponse errorResponse = new TrainingResponse();
            errorResponse.setStatus("error");
            errorResponse.setMessage("Invalid number format in hidden layer sizes: " + e.getMessage());
            return ResponseEntity.badRequest().body(errorResponse);
        }
    }

    /**
     * Predict email classification
     * @param features Email feature vector
     * @return Prediction result
     */
    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predictEmail(@RequestBody double[] features) {
        Map<String, Object> result = classifierService.predictEmail(features);
        return ResponseEntity.ok(result);
    }

    /**
     * Get last training metrics
     * @return Training metrics from the last training session
     */
    @GetMapping("/metrics")
    public ResponseEntity<TrainingResponse.TrainingMetrics> getMetrics() {
        TrainingResponse.TrainingMetrics metrics = classifierService.getLastTrainingMetrics();
        if (metrics == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(metrics);
    }

    /**
     * Get current model information
     * @return Model architecture and configuration details
     */
    @GetMapping("/model-info")
    public ResponseEntity<Map<String, Object>> getModelInfo() {
        Map<String, Object> info = classifierService.getModelInfo();
        return ResponseEntity.ok(info);
    }

    /**
     * Get available activation functions
     * @return List of supported activation functions
     */
    @GetMapping("/activation-functions")
    public ResponseEntity<String[]> getActivationFunctions() {
        String[] functions = {"SIGMOID", "TANH", "RELU", "LEAKY_RELU"};
        return ResponseEntity.ok(functions);
    }
}