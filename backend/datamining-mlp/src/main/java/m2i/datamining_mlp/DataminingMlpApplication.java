package m2i.datamining_mlp;

// Importation de la classe SpringApplication pour lancer l'application Spring Boot
import org.springframework.boot.SpringApplication;
// Importation de l'annotation @SpringBootApplication qui configure automatiquement l'application
import org.springframework.boot.autoconfigure.SpringBootApplication;

// Annotation qui indique que cette classe est le point de départ d'une application Spring Boot
@SpringBootApplication
public class DataminingMlpApplication {

	// Méthode principale qui lance l'application Spring Boot
	public static void main(String[] args) {
		SpringApplication.run(DataminingMlpApplication.class, args);
	}

}
