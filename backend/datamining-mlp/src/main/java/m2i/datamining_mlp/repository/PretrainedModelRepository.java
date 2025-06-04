package m2i.datamining_mlp.repository;

import m2i.datamining_mlp.model.PretrainedModel;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface PretrainedModelRepository extends MongoRepository<PretrainedModel, String> {
}