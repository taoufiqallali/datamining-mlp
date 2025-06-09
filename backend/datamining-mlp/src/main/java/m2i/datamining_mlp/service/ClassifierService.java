package m2i.datamining_mlp.service;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import m2i.datamining_mlp.DTO.TrainingRequest;
import m2i.datamining_mlp.DTO.TrainingResponse;
import m2i.datamining_mlp.model.Classifier;
import m2i.datamining_mlp.model.PretrainedModel;
import m2i.datamining_mlp.repository.PretrainedModelRepository;

/**
 * Classe de service pour l'entraînement et l'utilisation d'un classifieur à
 * réseau de neurones pour la détection de spam dans les e-mails. Gère le
 * chargement du jeu de données, l'entraînement du modèle, la prédiction et la
 * récupération des métriques du modèle.
 */
@Service
public class ClassifierService {

    /**
     * L'instance actuelle du classifieur entraîné.
     */
    private Classifier currentClassifier;

    /**
     * Stocke les métriques de la dernière session d'entraînement.
     */
    private TrainingResponse.TrainingMetrics lastTrainingMetrics;

    private final PretrainedModelRepository pretrainedModelRepository;

    @Autowired
    public ClassifierService(PretrainedModelRepository pretrainedModelRepository) {
        this.pretrainedModelRepository = pretrainedModelRepository;
    }

    /**
     * Entraîne un classifieur réseau de neurones en utilisant la configuration
     * et le jeu de données fournis.
     *
     * @param request La requête d'entraînement contenant les tailles des
     * couches cachées, la fonction d'activation, le taux d'apprentissage, et le
     * nombre d'époques.
     * @return Un objet TrainingResponse contenant le statut de l'entraînement,
     * un message, les métriques, et l'historique des pertes par époque.
     */
    public TrainingResponse trainModel(TrainingRequest request) {
        TrainingResponse response = new TrainingResponse();
        List<TrainingResponse.EpochLoss> epochLosses = new ArrayList<>();

        try {
            // Valider les tailles des couches cachées
            if (request.getHiddenSizes() == null || request.getHiddenSizes().length == 0) {
                response.setStatus("error");
                response.setMessage("Les tailles des couches cachées doivent être spécifiées");
                return response;
            }

            // Vérifier que toutes les tailles des couches cachées sont positives
            for (int size : request.getHiddenSizes()) {
                if (size <= 0) {
                    response.setStatus("error");
                    response.setMessage("Toutes les tailles des couches cachées doivent être positives");
                    return response;
                }
            }

            // Parser et valider la fonction d'activation
            Classifier.ActivationFunction activationFunction;
            try {
                activationFunction = Classifier.ActivationFunction.valueOf(request.getActivationFunction().toUpperCase());
            } catch (IllegalArgumentException e) {
                response.setStatus("error");
                response.setMessage("Fonction d'activation invalide. Options valides : SIGMOID, TANH, RELU, LEAKY_RELU");
                return response;
            }

            // Charger le jeu de données depuis un fichier CSV
            List<String[]> dataset = loadDataset();

            // Vérifier si le jeu de données est valide
            if (dataset.isEmpty()) {
                response.setStatus("error");
                response.setMessage("Le jeu de données est vide ou invalide");
                return response;
            }

            // Déterminer le nombre de caractéristiques (excluant la première colonne ID et la dernière colonne cible)
            String[] header = dataset.get(0);
            int featureCount = header.length - 2; // -2 pour exclure les colonnes ID et cible

            // Valider le nombre de caractéristiques
            if (featureCount <= 0) {
                response.setStatus("error");
                response.setMessage("Aucune caractéristique valide trouvée dans le jeu de données");
                return response;
            }

            // Initialiser les tableaux pour les caractéristiques et cibles
            double[][] features = new double[dataset.size() - 1][featureCount]; // -1 pour ignorer l'entête
            int[] target = new int[dataset.size() - 1];

            // Parser le jeu de données, ignorer la première colonne (ID) et extraire la dernière colonne comme cible
            for (int i = 1; i < dataset.size(); i++) { // Commencer à 1 pour ignorer l'entête
                String[] row = dataset.get(i);

                // Parser les caractéristiques, commencer à l'index 1 pour ignorer ID
                for (int j = 0; j < featureCount; j++) {
                    try {
                        features[i - 1][j] = Double.parseDouble(row[j + 1]);
                    } catch (NumberFormatException e) {
                        features[i - 1][j] = 0.0; // Par défaut 0 pour les nombres invalides
                    }
                }

                // Parser la cible depuis la dernière colonne
                try {
                    target[i - 1] = Integer.parseInt(row[header.length - 1]);
                } catch (NumberFormatException e) {
                    target[i - 1] = 0; // Par défaut 0 pour cible invalide
                }
            }

            // Calculer les statistiques du jeu de données
            int totalEmails = features.length;
            int spamCount = 0;
            for (int t : target) {
                if (t == 1) {
                    spamCount++;
                }
            }

            // Diviser le jeu de données en ensembles d'entraînement (80%) et de test (20%)
            int trainSize = (int) (totalEmails * 0.8);
            int testSize = totalEmails - trainSize;

            // Mélanger les indices pour un découpage aléatoire train-test
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < totalEmails; i++) {
                indices.add(i);
            }
            Collections.shuffle(indices, new Random(42));

            // Initialiser les ensembles d'entraînement et de test
            double[][] xTrain = new double[trainSize][featureCount];
            double[][] xTest = new double[testSize][featureCount];
            int[] yTrain = new int[trainSize];
            int[] yTest = new int[testSize];

            // Remplir l'ensemble d'entraînement
            for (int i = 0; i < trainSize; i++) {
                xTrain[i] = features[indices.get(i)].clone();
                yTrain[i] = target[indices.get(i)];
            }

            // Remplir l'ensemble de test
            for (int i = 0; i < testSize; i++) {
                xTest[i] = features[indices.get(i + trainSize)].clone();
                yTest[i] = target[indices.get(i + trainSize)];
            }

            // Initialiser le classifieur avec l'architecture spécifiée
            currentClassifier = new Classifier(featureCount, request.getHiddenSizes(),
                    request.getLearningRate(), activationFunction);

            // Entraîner le modèle en suivant la perte par époque
            for (int epoch = 0; epoch < request.getEpochs(); epoch++) {
                double totalLoss = 0.0;

                // Mélanger les données d'entraînement à chaque époque
                List<Integer> trainIndices = new ArrayList<>();
                for (int i = 0; i < trainSize; i++) {
                    trainIndices.add(i);
                }
                Collections.shuffle(trainIndices);

                // Entraîner sur chaque échantillon et calculer la perte
                for (int idx : trainIndices) {
                    currentClassifier.trainSample(xTrain[idx], yTrain[idx]);

                    double prediction = currentClassifier.predict(xTrain[idx]);
                    double loss = Math.pow(yTrain[idx] - prediction, 2);
                    totalLoss += loss;
                }

                // Enregistrer la perte moyenne pour chaque 5ème époque ou la dernière époque
                double avgLoss = totalLoss / xTrain.length;
                if (epoch % 5 == 0 || epoch == request.getEpochs() - 1) {
                    epochLosses.add(new TrainingResponse.EpochLoss(epoch, avgLoss));
                }
            }

            // Évaluer le modèle sur l'ensemble de test
            int correct = 0;
            int totalSpam = 0;
            int correctSpam = 0;
            int totalNotSpam = 0;
            int correctNotSpam = 0;

            for (int i = 0; i < xTest.length; i++) {
                double prediction = currentClassifier.predict(xTest[i]);
                int predictedClass = prediction > 0.5 ? 1 : 0;

                if (predictedClass == yTest[i]) {
                    correct++;
                }

                if (yTest[i] == 1) {
                    totalSpam++;
                    if (predictedClass == 1) {
                        correctSpam++;
                    }
                } else {
                    totalNotSpam++;
                    if (predictedClass == 0) {
                        correctNotSpam++;
                    }
                }
            }

            // Construire les métriques de la réponse
            TrainingResponse.TrainingMetrics metrics = new TrainingResponse.TrainingMetrics();
            metrics.setTotalEmails(totalEmails);
            metrics.setSpamEmails(spamCount);
            metrics.setNonSpamEmails(totalEmails - spamCount);
            metrics.setFeatureDimensions(featureCount);
            metrics.setTrainSize(trainSize);
            metrics.setTestSize(testSize);
            metrics.setAccuracy((double) correct / testSize);
            metrics.setSpamDetectionRate(totalSpam > 0 ? (double) correctSpam / totalSpam : 0);
            metrics.setNonSpamDetectionRate(totalNotSpam > 0 ? (double) correctNotSpam / totalNotSpam : 0);

            // Définir les informations sur l'architecture du réseau neuronal
            metrics.setHiddenLayerSizes(request.getHiddenSizes());
            metrics.setNumHiddenLayers(request.getHiddenSizes().length);
            metrics.setActivationFunction(request.getActivationFunction());

            // Stocker les métriques pour récupération ultérieure
            lastTrainingMetrics = metrics;

            System.out.println(totalSpam);
            System.out.println(correctSpam);
            System.out.println(totalNotSpam);
            System.out.println(correctNotSpam);

            // Définir la réponse de succès
            response.setStatus("success");
            response.setMessage(String.format("Modèle entraîné avec succès avec %d couches cachées en utilisant %s comme activation",
                    request.getHiddenSizes().length, request.getActivationFunction()));
            response.setMetrics(metrics);
            response.setEpochLosses(epochLosses);

        } catch (Exception e) {
            response.setStatus("error");
            response.setMessage("Échec de l'entraînement : " + e.getMessage());
            e.printStackTrace();
        }

        return response;
    }

    /**
     * Prédit si un email est un spam à partir de ses caractéristiques.
     *
     * @param features Le vecteur de caractéristiques de l'email (sans l'ID).
     * @return Une map contenant le résultat de la prédiction, incluant le score
     * brut, la classification spam, la confiance et les informations sur le
     * modèle.
     */
    public Map<String, Object> predictEmail(double[] features) {
        Map<String, Object> result = new HashMap<>();

        // Vérifie si un modèle entraîné est disponible
        if (currentClassifier == null) {
            result.put("error", "Aucun modèle entraîné disponible");
            return result;
        }

        try {
            // Valide la taille du vecteur de caractéristiques
            if (features.length != currentClassifier.getInputSize()) {
                result.put("error", String.format("Taille du vecteur de caractéristiques incorrecte. Attendu %d, reçu %d",
                        currentClassifier.getInputSize(), features.length));
                return result;
            }

            // Effectue la prédiction
            double prediction = currentClassifier.predict(features);
            boolean isSpam = prediction > 0.5;
            double confidence = isSpam ? prediction : (1 - prediction);

            // Remplit le résultat
            result.put("prediction", prediction);
            result.put("isSpam", isSpam);
            result.put("classification", isSpam ? "SPAM" : "NOT SPAM");
            result.put("confidence", confidence);
            result.put("modelInfo", String.format("Réseau : %d couches, activation %s",
                    currentClassifier.getNumHiddenLayers(),
                    currentClassifier.getActivationFunction()));

        } catch (Exception e) {
            result.put("error", "Échec de la prédiction : " + e.getMessage());
        }

        return result;
    }

    /**
     * Prédit si un email est un spam en utilisant un modèle pré-entraîné stocké
     * en base.
     *
     * @param features Le vecteur de caractéristiques de l'email (sans l'ID).
     * @return Une map contenant le résultat de la prédiction, incluant le score
     * brut, la classification spam, la confiance et les informations sur le
     * modèle.
     */
    public Map<String, Object> predictPretrainedEmail(double[] features) {
        Map<String, Object> result = new HashMap<>();

        // Recherche le modèle pré-entraîné dans la base de données avec l'ID "pretrained_model"
        Optional<PretrainedModel> pretrainedModelOpt = pretrainedModelRepository.findById("pretrained_model");

        // Si aucun modèle trouvé, renvoie une erreur
        if (pretrainedModelOpt.isEmpty()) {
            result.put("error", "Aucun modèle pré-entraîné disponible");
            return result;
        }

        // Récupère le modèle pré-entraîné et le convertit en classifieur
        PretrainedModel pretrainedModel = pretrainedModelOpt.get();
        Classifier pretrainedClassifier = pretrainedModel.toClassifier();

        try {
            // Vérifie la taille du vecteur de caractéristiques
            if (features.length != pretrainedClassifier.getInputSize()) {
                result.put("error", String.format("Taille du vecteur de caractéristiques incorrecte. Attendu %d, reçu %d",
                        pretrainedClassifier.getInputSize(), features.length));
                return result;
            }

            // Fait la prédiction avec le classifieur pré-entraîné
            double prediction = pretrainedClassifier.predict(features);
            boolean isSpam = prediction > 0.5;
            double confidence = isSpam ? prediction : (1 - prediction);

            // Remplit le résultat
            result.put("prediction", prediction);
            result.put("isSpam", isSpam);
            result.put("classification", isSpam ? "SPAM" : "NOT SPAM");
            result.put("confidence", confidence);
            result.put("modelInfo", String.format("Réseau pré-entraîné : %d couches, activation %s",
                    pretrainedClassifier.getNumHiddenLayers(),
                    pretrainedClassifier.getActivationFunction()));

        } catch (Exception e) {
            // En cas d'erreur pendant la prédiction
            result.put("error", "Échec de la prédiction : " + e.getMessage());
        }

        return result;
    }

    /**
     * Récupère les métriques de la dernière session d'entraînement.
     *
     * @return L'objet TrainingMetrics contenant les détails du dernier
     * entraînement, ou null si aucun entraînement n'a eu lieu.
     */
    public TrainingResponse.TrainingMetrics getLastTrainingMetrics() {
        return lastTrainingMetrics;
    }

    /**
     * Récupère les métriques du modèle pré-entraîné stocké en base.
     *
     * @return L'objet TrainingMetrics du modèle pré-entraîné, ou null si le
     * modèle n'existe pas en base.
     */
    public TrainingResponse.TrainingMetrics getPretrainedMetrics() {
        Optional<PretrainedModel> pretrainedModelOpt = pretrainedModelRepository.findById("pretrained_model");
        return pretrainedModelOpt.map(PretrainedModel::getMetrics).orElse(null);
    }

    /**
     * Récupère les informations sur le modèle actuellement entraîné.
     *
     * @return Une map contenant les détails du modèle comme la taille d'entrée,
     * les tailles des couches cachées, le nombre de couches cachées, la
     * fonction d'activation, et le taux d'apprentissage.
     */
    public Map<String, Object> getModelInfo() {
        Map<String, Object> info = new HashMap<>();

        // Vérifie si un modèle entraîné existe
        if (currentClassifier == null) {
            info.put("error", "No trained model available");
            return info;
        }

        // Remplit les informations du modèle
        info.put("inputSize", currentClassifier.getInputSize());
        info.put("hiddenLayerSizes", currentClassifier.getHiddenSizes());
        info.put("numHiddenLayers", currentClassifier.getNumHiddenLayers());
        info.put("activationFunction", currentClassifier.getActivationFunction().toString());
        info.put("learningRate", currentClassifier.getLearningRate());

        return info;
    }

    /**
     * Récupère les informations du modèle pré-entraîné stocké dans la base de
     * données.
     *
     * @return Une map contenant les détails du modèle comme la taille d'entrée,
     * les tailles des couches cachées, le nombre de couches cachées, la
     * fonction d'activation, et le taux d'apprentissage, ou une erreur si aucun
     * modèle pré-entraîné n'est disponible.
     */
    public Map<String, Object> getPretrainedModelInfo() {
        Map<String, Object> info = new HashMap<>();
        Optional<PretrainedModel> pretrainedModelOpt = pretrainedModelRepository.findById("pretrained_model");
        if (pretrainedModelOpt.isEmpty()) {
            info.put("error", "No pretrained model available");
            return info;
        }

        PretrainedModel pretrainedModel = pretrainedModelOpt.get();
        info.put("inputSize", pretrainedModel.getInputSize());
        info.put("hiddenLayerSizes", pretrainedModel.getHiddenSizes());
        info.put("numHiddenLayers", pretrainedModel.getHiddenSizes().length);
        info.put("activationFunction", pretrainedModel.getActivationFunction());
        info.put("learningRate", pretrainedModel.getLearningRate());

        return info;
    }

    /**
     * Sauvegarde le modèle entraîné actuel et ses métriques comme modèle
     * pré-entraîné dans la base de données.
     *
     * @throws IllegalStateException si aucun modèle entraîné ou métriques ne
     * sont disponibles.
     */
    public void savePretrainedModel() {
        if (currentClassifier == null || lastTrainingMetrics == null) {
            throw new IllegalStateException("No trained model or metrics available to save");
        }
        PretrainedModel pretrainedModel = new PretrainedModel(currentClassifier, lastTrainingMetrics);
        pretrainedModelRepository.save(pretrainedModel);
    }

    /**
     * Charge le dataset d'emails depuis un fichier CSV.
     *
     * @return Une liste de tableaux de chaînes de caractères, chaque tableau
     * représentant une ligne dans le fichier CSV.
     * @throws Exception Si une erreur survient lors de la lecture du fichier.
     */
    private List<String[]> loadDataset() throws Exception {
        List<String[]> dataset = new ArrayList<>();
        String datasetPath = "src/main/resources/dataset/emails.csv";

        try (BufferedReader br = new BufferedReader(new FileReader(datasetPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Analyse une ligne CSV, en gérant les champs entre guillemets
                String[] row = parseCsvLine(line);
                dataset.add(row);
            }
        }

        return dataset;
    }

    /**
     * Analyse une ligne CSV en gérant les champs entre guillemets qui peuvent
     * contenir des virgules.
     *
     * @param line La ligne CSV à analyser.
     * @return Un tableau de chaînes de caractères représentant les champs
     * extraits de la ligne.
     */
    private String[] parseCsvLine(String line) {
        List<String> result = new ArrayList<>();
        boolean inQuotes = false;
        StringBuilder field = new StringBuilder();

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);

            if (c == '"') {
                inQuotes = !inQuotes; // Basculer l'état "dans les guillemets"
            } else if (c == ',' && !inQuotes) {
                result.add(field.toString().trim()); // Fin du champ si virgule hors guillemets
                field = new StringBuilder();
            } else {
                field.append(c); // Ajouter le caractère au champ en cours
            }
        }

        // Ajouter le dernier champ après la fin de la ligne
        result.add(field.toString().trim());
        return result.toArray(new String[0]);
    }

    private static String features = "the,to,ect,and,for,of,a,you,hou,in,on,is,this,enron,i,be,that,will,have,with,your,at,we,s,are,it,by,com,as,from,gas,or,not,me,deal,if,meter,hpl,please,re,e,any,our,corp,can,d,all,has,was,know,need,an,forwarded,new,t,may,up,j,mmbtu,should,do,am,get,out,see,no,there,price,daren,but,been,company,l,these,let,so,would,m,into,xls,farmer,attached,us,information,they,message,day,time,my,one,what,only,http,th,volume,mail,contract,which,month,more,robert,sitara,about,texas,nom,energy,pec,questions,www,deals,volumes,pm,ena,now,their,file,some,email,just,also,call,change,other,here,like,b,flow,net,following,p,production,when,over,back,want,original,them,below,o,ticket,c,he,could,make,inc,report,march,contact,were,days,list,nomination,system,who,april,number,sale,don,its,first,thanks,business,help,per,through,july,forward,font,free,daily,use,order,today,r,had,fw,set,plant,statements,go,gary,oil,line,sales,w,effective,well,tenaska,take,june,x,within,nbsp,she,how,north,america,being,under,next,week,than,january,last,two,service,purchase,name,less,height,off,agreement,k,work,tap,group,year,based,transport,after,think,made,each,available,changes,due,f,h,services,smith,send,management,stock,sent,ll,co,office,needs,cotten,did,actuals,u,money,before,looking,then,pills,online,request,look,desk,ami,his,same,george,chokshi,point,delivery,friday,does,size,august,product,pat,width,iv,noms,address,above,sure,give,october,future,find,market,n,mary,vance,melissa,said,internet,still,account,those,down,link,hsc,rate,people,pipeline,best,actual,very,end,home,houston,tu,high,her,team,products,many,currently,spot,receive,good,such,going,process,feb,monday,info,david,lloyd,again,both,click,subject,jackie,december,total,na,lisa,ve,september,hours,until,resources,because,aol,february,where,g,investment,issue,duke,since,pay,show,way,global,computron,further,most,place,offer,natural,activity,eastrans,graves,right,prices,date,john,utilities,november,clynes,jan,securities,meeting,susan,hplc,julie,able,received,align,term,id,revised,thursday,pg,fee,hplno,trading,additional,site,txu,data,wellhead,reply,taylor,news,unify,michael,provide,note,much,access,lannou,every,between,keep,tuesday,review,great,tom,put,done,long,save,section,must,v,part,nd,million,check,trade,bob,created,steve,prior,copy,continue,numbers,via,world,demand,hanks,contracts,phone,transaction,customer,possible,pefs,meyers,months,special,without,used,regarding,software,howard,support,buy,young,meters,thru,believe,gcs,cec,entered,control,dec,face,create,weissman,st,color,come,supply,brian,hplo,own,correct,customers,web,allocation,soon,using,development,mark,low,power,problem,once,however,tickets,border,performance,manager,rates,center,companies,risk,details,needed,international,field,even,someone,doc,fuel,lee,paid,while,start,index,include,nominations,act,pricing,scheduled,gathering,type,href,during,aimee,anything,feel,fuels,getting,advice,why,increase,path,sell,works,issues,three,enronxgate,camp,either,form,security,interest,financial,family,xp,plan,current,top,another,src,spreadsheet,allen,wednesday,read,him,working,wynne,add,deliveries,buyback,allocated,firm,james,marketing,tx,results,got,stocks,calpine,might,operations,position,logistics,fax,cost,party,zero,pops,old,pt,scheduling,flowed,dollars,update,gco,katy,including,follow,yahoo,already,suite,error,past,page,stop,changed,book,program,few,better,operating,equistar,move,cotton,aep,y,state,ees,rita,provided,employees,period,morning,cd,hotmail,entex,swing,real,exchange,tomorrow,lst,counterparty,parker,person,follows,valid,visit,little,professional,quality,confirm,something,megan,brenda,around,windows,im,storage,accounting,called,ranch,tax,problems,case,teco,fact,always,too,unsubscribe,amount,coastal,never,rodriguez,love,acton,shut,pipe,project,hope,limited,invoice,credit,full,survey,ray,carlos,anyone,wanted,yet,ic,scott,years,charlie,soft,notice,advise,addition,donald,lsk,wish,katherine,website,hplnl,schumack,prescription,cover,shares,cash,imbalance,united,handle,big,everyone,style,clear,producer,weekend,city,requested,stone,left,payment,mobil,shows,small,confirmed,technology,meet,extend,life,intended,sherlyn,schedule,else,letter,box,bill,richard,lamphier,complete,ever,release,newsletter,anita,clem,having,herod,beginning,papayoti,try,mike,enter,estimates,location,cut,question,things,personal,feedback,cialis,found,area,dow,terms,central,necessary,man,run,reason,third,midcon,charge,president,de,listed,meds,thomas,thought,capital,added,ask,weeks,investing,commercial,star,several,easy,view,cannot,extended,lauri,beaumont,union,times,open,cause,monthly,action,offers,industry,states,side,mailto,probably,neal,second,stephanie,download,flash,agree,mcf,transfer,doing,important,basis,different,final,koch,exxon,remove,microsoft,interested,application,sept,mg,write,lp,east,requirements,code,value,thank,together,exploration,mid,dfarmer,everything,receipt,thu,afternoon,late,enserch,coming,bank,response,tell,shipping,night,events,cynthia,lsp,close,legal,country,direct,expected,ces,corporation,options,really,voip,nominated,etc,latest,potential,priced,edward,valero,material,stack,victor,redeliveries,loss,remember,baumbach,option,private,longer,aware,included,drugs,public,reinhardt,version,hesse,discuss,related,asked,say,viagra,revision,bgcolor,kind,pro,completed,health,ready,plans,registered,regards,carthage,zone,fill,away,computer,systems,industrial,mentioned,told,therefore,growth,sold,track,reports,south,rd,jim,costs,image,expect,return,physical,el,browser,donna,stacey,begin,china,duty,approximately,showing,unit,jones,hard,verify,updated,eol,cs,orders,talk,trying,base,given,server,source,pathed,strong,bryan,directly,risks,whole,major,users,purchases,oo,karen,luong,level,required,delivered,portfolio,riley,ali,easttexas,poorman,bellamy,assistance,nothing,gif,thing,retail,didn,valley,department,cleburne,allow,gpgfin,answer,items,paste,avila,taken,mm,nguyen,ensure,reference,hall,later,lone,user,methanol,facility,network,spoke,though,tabs,taking,status,considered,purchased,says,yourself,paliourg,dy,jeff,businesses,fred,transportation,apache,morris,nov,ltd,brand,federal,statement,oasis,reflect,assets,lamadrid,general,bridge,ability,oct,play,enrononline,compliance,spam,availability,king,understanding,chance,quick,effort,points,reliantenergy,fixed,short,hill,cheryl,aepin,key,understand,valign,capacity,game,took,bring,guys,god,green,care,withers,property,hub,johnson,employee,wants,albrecht,meaning,expectations,mx,moved,cernosek,matter,devon,calls,worldwide,records,removed,lose,large,referenced,walker,iferc,enw,ponton,eileen,ship,upon,enerfin,jennifer,looks,staff,pc,target,waha,making,cp,impact,partner,immediately,shall,channel,takes,sat,others,hear,went,travel,listing,approved,processing,early,enough,sally,starting,distribution,tejas,transactions,stay,earl,superty,doesn,reserves,includes,choose,adobe,publisher,paso,cornhusker,training,markets,content,solution,shell,jpg,print,drive,pain,password,half,herrera,saturday,moopid,hotlist,balance,super,vacation,sex,happy,excess,existing,fund,stella,share,sign,wells,won,four,text,card,tisdale,fwd,appreciate,non,experience,savings,settlements,draft,couple,informed,biz,watch,plus,sun,expense,images,land,occur,flowing,mar,terry,darren,cheap,weight,dynegy,activities,become,mr,format,attention,entire,photoshop,williams,instructions,neon,janet,contains,ago,friends,against,boas,music,certain,liz,svcs,record,fast,dave,held,mind,ua,publication,differ,comments,fun,rest,instant,agent,communications,director,partners,investors,expedia,kevin,assist,safe,approval,allocate,black,none,intrastate,document,eric,hakemack,expired,lower,active,secure,cc,five,determine,press,colspan,missing,jill,discussion,relief,respect,specific,technologies,al,holmes,white,yesterday,medical,pinion,sorry,men,leave,pass,video,gomes,doctor,projects,limit,air,knle,pharmacy,confirmation,opportunity,involve,notify,gtc,class,ken,started,outage,confidential,room,blue,estimated,officer,reach,messages,database,words,prc,tracked,transition,light,national,hot,offering,gulf,provides,iit,demokritos,mckay,average,wide,heard,files,dan,billed,mccoy,rc,exactly,middle,select,bruce,louisiana,receiving,california,event,roll,mops,william,appear,perfect,html,features,join,greater,sunday,pick,featured,cdnow,prize,reveffo,olsen,expects,estimate,near,common,package,title,whether,bought,evergreen,difference,elizabeth,history,monitor,advised,result,sources,school,unaccounted,paragraph,turn,kimberly,increased,communication,members,concerns,uncertainties,associated,reduce,committed,wi,asap,goes,trader,waiting,canada,worth,representative,claim,ceo,london,discussions,php,brazos,trevino,calling,involved,la,gift,southern,groups,hour,tufco,previously,voice,normally,resolve,efforts,nor,recent,purchasing,county,ok,express,generic,according,respond,situation,hold,lot,interconnect,word,came,west,role,opportunities,corporate,remain,similar,readers,suggestions,subscribers,projections,lead,learn,resolved,agreed,sec,head,enjoy,img,rnd,responsible,outstanding,member,panenergy,american,cass,register,promotions,parties,winfree,selling,usage,appropriate,assignment,media,believes,require,submit,model,spinnaker,copano,facilities,opinion,factors,identified,beverly,ews,gdp,deliver,job,profile,across,neuweiler,suggest,girls,manage,usa,local,bad,greg,vs,fees,digital,cf,strangers,registration,delta,rolex,goliad,hesco,success,primary,quarter,course,chairman,petroleum,notes,medications,ei,instead,fine,lake,pre,force,seek,recipient,gain,placed,age,least,body,asking,discussed,hanson,emails,nominate,ext,known,ones,ed,assigned,htmlimg,means,present,various,invoices,gd,agency,along,located,reflects,solutions,ex,house,cds,br,owner,apr,sullivan,basin,linda,worked,car,seen,properties,booked,higher,store,est,revenue,wait,women,far,met,wholesale,range,kcs,recorded,brown,lots,match,input,grant,providing,huge,investor,kelly,apply,paths,handling,pipes,advantage,analysis,focus,draw,red,origination,connection,planning,wilson,golf,summary,item,bankruptcy,expenses,pgev,encina,beaty,memo,initial,thousand,mills,penis,friend,conversation,multiple,martin,names,bit,dth,talked,behalf,preliminary,button,herein,gisb,coupon,sa,oi,appears,door,texaco,csikos,arrangements,cpr,expires,popular,sending,research,conditions,gb,board,ca,applications,tried,paying,acquisition,reporting,normal,maintenance,resume,announced,attachment,buyer,objectives,prod,represent,sandi,hplnol,government,committee,running,tetco,discount,jo,holding,earlier,positions,happen,mailing,decided,recently,chris,xanax,valium,broadband,individual,station,td,financing,somehow,pena,critical,attend,kristen,inform,highly,hl,phillips,minutes,titles,affiliate,wife,lonestar,charlotte,quickly,paper,test,comes,mobile,internal,privacy,ideas,live,gotten,floor,benefit,percent,ms,dr,ebs,msn,gave,dallas,enterprise,rx,spring,ftar,ooking,hawkins,exclusive,selected,baxter,actually,single,shop,nominates,guarantee,minute,correctly,unique,bid,building,stated,accept,assumptions,centana,senior,pill,kinsey,sap,immediate,goals,category,mitchell,acceptance,termination,sweeney,facts,amazon,arrangement,josey,funds,among,accuracy,mean,rather,kim,egmnom,indicate,updates,extra,adjustment,accounts,lowest,gold,purposes,remaining,talking,entry,road,load,simply,europe,lindley,understood,logos,hi,speed,profit,notified,jackson,z,vols,serve,additionally,shipped,connor,fontfont,q,kept,dollar,jr,almost,fri,paul,documents,analyst,crude,cap,shopping,aug,clearance,schneider,ftworth,father,anticipated,resellers,congress,counterparties,epgt,buying,san,invest,cartwheel,brandywine,wrong,mtbe,split,submitted,hull,gra,children,leader,true,baseload,mb,letters,billion,rights,mtr,heidi,clean,historical,asset,foreign,gr,entity,developed,maybe,jeffrey,transmission,outside,lost,membership,invitation,ocean,legislation,hernandez,pep,payments,wallis,rev,kenneth,seaman,annual,guess,bammel,lines,guadalupe,zivley,exception,example,pathing,revisions,pipelines,equity,budget,wed,dealers,window,juno,claims,bottom,standard,alternative,merchant,braband,topica,telephone,reliant,speculative,yes,en,morgan,cable,edmondson,participate,usb,throughout,checked,myself,contents,fat,investments,six,build,giving,calendar,inherent,edition,darial,hr,trip,pull,moving,concern,proposed,rm,deer,enquiries,alt,tammy,front,reduction,evening,concerning,gets,effect,isn,haven,cowboy,sea,dvd,launch,minimum,changing,built,avoid,chief,stephen,chad,manual,finally,strategy,executive,thousands,conflict,resulting,policy,commission,stand,positive,quantity,programs,airmail,texoma,prepared,austin,matt,intent,uae,citibank,jaquet,hol,harris,min,hplr,advance,weather,terminated,whom,sheet,venturatos,cellpadding,hotel,leading,guaranteed,idea,announce,pleased,award,operational,prepare,schedulers,child,sum,quote,adjusted,warning,issued,ga,cross,detail,pertaining,tess,owe,crow,availabilities,griffin,christy,crosstex,eel,itoy,heart,licensed,overnight,cal,otherwise,luck,stretch,generation,broker,construed,except,traders,carry,column,approx,main,alert,charges,step,revenues,games,gottlob,looked,individuals,beck,stuff,welcome,port,glover,description,daniel,quantities,park,managing,town,seller,summer,tina,dates,eff,dudley,ferc,robin,charles,customerservice,zonedubai,emirates,aeor,clickathome,materia,island,vaughn,sexual,eiben,forms,delete,realize,tailgate,behind,villarreal,lon,benoit,simple,tech,ahead,double,ordering,se,miss,law,eb,post,outlook,equipment,leslie,reeves,org,tools,cold,adjustments,contained,saw,edit,deciding,finance,patti,listbot,river,kathryn,holiday,successful,unable,advisor,pool,bryce,outages,adjust,screen,otc,brent,helps,auto,foot,region,links,contain,knowledge,yvette,dial,pressure,detailed,indicated,charged,sites,makes,female,mcmills,cook,mazowita,meredith,allocations,meetings,particular,environment,drug,search,mailings,designed,rock,measurement,art,corrected,kids,benefits,tv,seems,husband,fix,grow,decision,wireless,mo,conference,interview,levels,copies,cindy,urgent,regular,payroll,shown,consumers,reliable,tr,indicating,coast,greif,severson,tri,vicodin,liquids,significant,intend,usd,pager,avails,spencer,ce,charset,verdana,fully,flynn,da,personnel,multi,closed,vice,administration,gmt,midstream,eye,speckels,studio,cilco,likely,managers,structure,sit,parent,preparation,mix,mmbtus,timing,happening,lottery,killing,acquire,mack,pcx,fares,internationa,notification,swift,identify,areas,separate,unless,producers,allows,pretty,waste,joanie,drop,taxes,premium,teams,choice,largest,addressed,dolphin,ngo,self,davis,htm,ad,graphics,hit,competitive,thus,incorrect,ti,acts,previous,edu,proven,electric,pictures,charlene,benedict,chevron,treatment,lesson,player,sds,wc,intraday,assurance,sdsnom,rebecca,quit,netco,intra,whatever,lyondell,reviewed,solicitation,filings,log,noon,locations,joe,completely,rivers,language,street,automatically,ft,powerful,specials,alone,fyi,properly,proper,explode,decrease,medication,desks,impacted,anywhere,completion,banking,consider,certificate,exercise,zeroed,websites,tonight,diligence,education,club,vegas,affordable,sports,predictions,billing,diamond,posted,prayer,actions,nomad,resuits,jason,purpose,deposit,entertainment,materially,blank,resolution,anderson,nat,rom,soma,organization,aquila,solid,affected,transco,spend,responsibilities,assume,header,accountant,functionality,meant,killed,analysts,rick,rolled,noted,discovered,offices,torch,often,york,joint,briley,competition,guide,intercompany,son,settlement,presently,cart,tim,entries,russ,valadez,rules,molly,apple,atleast,scheduler,pi,hector,dell,opm,hottlist,yap,gone,heal,llc,setting,reached,proposal,hundred,trust,official,table,mcgee,written,operation,cellspacing,laptop,feature,ram,victoria,larry,units,requests,continued,external,pack,couldn,lateral,strictly,resource,although,sr,commodity,pulled,protocol,bed,generated,redmond,girl,apparently,tool,reviews,released,movies,inside,shareholder,rr,compensation,beliefs,foresee,lease,rule,marta,chemical,hillary,hp,tongue,adonis,advises,master,eight,wasn,itself,documentation,xl,humble,elsa,pics,hughes,brokered,distribute,consultation,sheri,lists,cannon,treated,factor,putting,verified,releases,enhanced,controls,craig,worksheet,conversion,max,hrs,helpful,hand,producing,dl,developing,design,woman,understands,standards,promotion,sarco,hospital,ffffff,respective,richmond,conoco,driver,easily,sean,den,gateway,holdings,brad,college,gains,adult,dated,em,mcloughlin,anticipates,henderson,julia,negotiations,sofftwaares,garrick,comstock,trochta,imceanotes,ecom,larger,nommensen,coordinate,partnership,otcbb,announces,louis,dealer,reliance,season,agua,dulce,offshore,gathered,forever,function,happened,sample,easier,aim,pa,expensive,thinks,maximum,war,mining,drilling,owned,todd,advanced,provider,pending,providers,silver,cherry,hundreds,thoughts,addresses,beach,baby,requires,caused,variance,extension,carbide,anytime,adding,triple,dawn,martinez,entering,login,bretz,ls,writeoff,locker,wiil,block,blood,romeo,responsibility,brennan,btu,venture,connected,nascar,opinions,executed,cell,flag,doctors,invoiced,marlin,coffey,nice,amazing,ii,determined,handled,keeping,touch,upgrade,shipment,brought,forwarding,confidence,hesitate,seem,electronic,appreciated,deadline,franklin,heather,reasons,passed,safety,procedures,payback,networks,utility,count,africa,exact,creating,loading,processed,court,tier,sender,att,mailbox,glad,buddy,profiles,portion,protection,compressor,okay,oba,finding,heads,bar,turned,remote,illustrator,oem,noticed,mails,darron,nick,urbanek,jerry,barrett,ehronline,und,abdv,egm,couid,technoiogies,owns,improved,eat,moment,owners,develop,installed,videos,frank,hearing,inches,busy,ref,valuable,et,un,url,shawna,iso,capture,extremely,ya,causing,consent,anyway,round,discrepancies,cheapest,confidentiality,disclosure,prohibited,vol,correction,communicate,processes,spain,shareholders,supported,smoking,mine,biggest,erections,platform,miles,exciting,association,die,restricted,ma,income,goal,bane,collection,nathan,wind,piece,familiar,gore,experiencing,pico,mai,dewpoint,tessie,hair,bussell,diane,delivering,originally,accurate,began,seven,tracking,randall,gay,emerging,prescriptions,story,arial,florida,space,ownership,european,sutton,concerned,male,spent,agreements,industries,picture,filled,continues,death,choate,majeure,device,hence,ten,campaign,massive,eyes,requesting,lives,reminder,eliminate,copied,consemiu,died,sound,offered,expressed,anti,duplicate,steps,books,improve,implementation,gives,ac,peggy,proprietary,ways,advertisement,published,earnings,mortgage,consumer,ct,tape,fl,cia,organizational,agenda,rental,carriere,moshou,church,trouble,medium,aggressive,smart,zajac,ail,participants,gap,earthlink,wire,trades,messaging,ut,wil,richardson,blvd,glo,seneca,pubiisher,imited,isc,contacts,sleep,kyle,cooperation,possibly,leaving,motor,hopefully,tie,speak,mi,suggested,canadian,uses,connect,pvr,rich,places,auction,po,spacer,client,recommended,royalty,amended,default,living,regardless,human,bringing,focused,stores,variety,netherlands,leaders,bowen,salary,signed,penny,loan,desktop,chase,pleasure,compare,session,overall,stranger,length,planned,sp,darrel,raise,palestinian,expiration,serial,premiere,suzanne,reduced,players,applicable,impotence,buckley,wayne,hansen,indicative,sabrae,dating,winners,marshall,highest,ea,presentation,allowed,square,danny,gepl,hydrocarbon,alpine,christmas,muscle,souza,relating,begins,ecf,forth,answers,audit,approve,lunch,types,starts,difficult,le,lasts,series,till,edge,growing,covered,shipper,sometime,republic,filter,sooner,increasing,nelson,percentage,returned,pop,interface,kin,experienced,prime,merger,obtain,ryan,servers,attachments,achieve,effects,gov,examples,procedure,explore,caribbean,rally,amounts,comfort,attempt,greatly,amelia,engel,delay,fare,der,cove,filing,fletcher,leth,undervalued,cents,esther,hlavaty,reid,lls,troy,palmer,metals,las,carter,luis,migration,brief,hess,therein,ur,pond,joanne,community,tglo,eogi,ml,wysak,felipe,errors,affect,convenient,minimal,boost,incremental,decide,reserve,superior,kerr,willing,quite,wild,unlimited,sans,mother,computers,unfortunately,ordered,satisfaction,priority,traded,testing,portal,ward,lets,aren,knows,refer,shot,fda,tue,saying,cancel,forecast,cousino,bass,permanent,phones,technical,whose,objective,cards,distributed,learning,fire,drill,towards,forget,explosion,gloria,formula,redelivery,audio,visual,encoding,approach,doubt,staffing,excite,corel,tm,enronavailso,contacting,alland,heavy,economic,nigeria,milwaukee,phillip,curve,returns,padre,kathy,buttons,sir,vary,sounds,disclose,authority,flw,straight,worldnet,beemer,ooo,defs,thorough,officers,flight,prefer,awesome,macintosh,feet,constitutes,formosa,porn,armstrong,driscoll,watches,newsietter,twenty,tommy,fields,method,setup,allocating,initially,missed,clarification,especially,dorcheus,del,millions,insurance,pooling,trial,tennessee,ellis,direction,bold,catch,performing,accepted,matters,batch,continuing,winning,symbol,offsystem,decisions,produced,ended,greatest,degree,solmonson,imbalances,fall,fear,hate,fight,reallocated,debt,reform,australia,plain,prompt,remains,ifhsc,enhancements,connevey,jay,valued,lay,infrastructure,military,allowing,ff,dry";

    // Liste prédéfinie des caractéristiques (mots) dans l'ordre spécifié
    private static final String[] FEATURES = features.split(",");

    public double[] textToFeatureVector(String text) {
        // Initialise le vecteur de caractéristiques avec des zéros
        double[] featureVector = new double[FEATURES.length];

        // Convertit le texte en minuscules pour une correspondance insensible à la casse
        text = text.toLowerCase();

        // Divise le texte en mots (tokenisation simple par espaces et ponctuation)
        String[] words = text.split("[\\s\\p{Punct}]+");

        // Compte les occurrences de chaque mot caractéristique
        for (String word : words) {
            for (int i = 0; i < FEATURES.length; i++) {
                if (word.equals(FEATURES[i])) {
                    featureVector[i]++;
                }
            }
        }

        return featureVector;
    }

}
