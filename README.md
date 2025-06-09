#  Classificateur d'e-mails avec réseau de neurones (Spring Boot + MongoDB)

Ce projet est une application Java Spring Boot qui implémente un **réseau de neurones multicouche (MLP)** pour classer les e-mails en **SPAM** ou **NON SPAM**.

##  Fonctionnalités

- Entraînement d'un modèle de classification à partir de données CSV.
- Prédiction en temps réel si un e-mail est du spam ou non.
- Sauvegarde et chargement de modèles entraînés depuis MongoDB.
- Interface simple via un fichier HTML (`index.html`).

---

## ⚙ Prérequis

- **Java 17** ou version ultérieure.
- **Maven** (`mvn`) installé.
- **MongoDB** en local, accessible via : mongodb://localhost:27017/

- Base de données MongoDB : `spam_classifier`  
- Collection MongoDB : `pretrained_models`

---

##  Dépendances (automatiquement gérées par Maven)

Le projet utilise les bibliothèques suivantes :

- Spring Boot
- Spring Data MongoDB
- Jackson pour le traitement JSON
- Java standard (`java.util`, `java.io`, etc.)

Toutes les dépendances sont spécifiées dans le fichier `pom.xml`.  
Aucune installation manuelle n'est requise si vous utilisez Maven.

---

##  Étapes d'exécution

### 1. Lancer MongoDB

Assurez-vous que votre serveur MongoDB fonctionne sur : mongodb://localhost:27017/

Créer manuellement une base de données appelée :

spam_classifier

Et une collection :

pretrained_models

### 2.  Importer les données 
Vous pouvez importer un fichier  spam_classifier.pretrained_models.json  contenant un modèle pré-entraîné dans la collection.

### 3.  Lancer l'application Spring Boot
executer la commande suivante a partire du dossier datamining-mlp\backend\datamining-mlp
mvn spring-boot:run

### 4. Ouvrir l'interface utilisateur
Ouvrez le fichier index.html dans votre navigateur : le  fichier est situee dans datamining-mlp\frontend

## Interface Web

- **Train Model** : permet d'entraîner un modèle en fournissant des paramètres personnalisés.  
- **Predict** : permet de prédire si un e-mail est un spam ou non en utilisant le modèle entraîné.  
- **Metrics** : affiche les métriques du dernier modèle entraîné.  
- **Pretrained Models** : permet de tester un modèle préentraîné enregistré dans la base de données.  

## Structure du projet

├── backend/datamining-mlp/
│   ├── mvnw / mvnw.cmd
│       → Scripts pour exécuter Maven sans l’installer globalement (auto-générés par Spring Initializr).
│
│   ├── pom.xml  
│       → Fichier de configuration Maven. Définit les dépendances, les plugins, le Java version, etc.
│
│   ├── src/main/java/m2i/datamining_mlp/
│   │   ├── DataminingMlpApplication.java  
│   │       → Point d’entrée principal de l’application Spring Boot.
│   │
│   │   ├── controller/ClassifierController.java  
│   │       → Contrôleur REST : expose les routes pour entraîner, tester et utiliser le modèle.
│   │
│   │   ├── service/ClassifierService.java  
│   │       → Contient la logique métier pour entraîner et prédire avec le modèle de classification.
│   │
│   │   ├── model/Classifier.java  
│   │       → Implémentation d’un réseau de neurones simple (perceptron multicouche).
│   │
│   │   ├── model/PretrainedModel.java  
│   │       → Modèle de données représentant un modèle entraîné, stocké en base MongoDB.
│   │
│   │   ├── repository/PretrainedModelRepository.java  
│   │       → Interface Spring Data MongoDB pour accéder aux modèles pré-entraînés.
│   │
│   │   ├── DTO/
│   │       ├── EmailRequest.java  
│   │           → Représente les données envoyées pour faire une prédiction (texte d’email).
│   │       ├── TrainingRequest.java  
│   │           → Contient les paramètres pour l'entraînement du modèle.
│   │       ├── TrainingResponse.java  
│   │           → Réponse contenant les métriques et les pertes par époque après entraînement.
│
│   ├── src/main/resources/
│   │   ├── application.properties  
│   │       → Configuration de l’application : MongoDB URI, nom de la base, etc.
│   │   ├── dataset/emails.csv  
│   │       → Jeu de données utilisé pour entraîner un modèle.
│
│   ├── src/test/java/m2i/datamining_mlp/DataminingMlpApplicationTests.java  
│       → Classe de test générée automatiquement pour vérifier le démarrage de l’application.
│
├── frontend/index.html  
│   → Interface utilisateur simple pour interagir avec le backend (à ouvrir dans un navigateur).

├── spam_classifier.pretrained_models.json  
│   → Fichier JSON contenant des modèles pré-entraînés à importer dans la base MongoDB.
│
├── spam_classifier.pretrained_models.zip  
│   → Jeu de données utilisé pour entraîner un modèle.
