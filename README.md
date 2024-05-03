# Utilizing Machine Learning to Predict Mass Shootings: An Evaluation of Isolation Forest and Random Forest Classifier Models
 
 ***Abstract***

*Mass shootings have become a distressing and recurrent phenomenon in the United States, raising urgent concerns about public safety and the identification of individuals at risk of perpetrating such violent acts. This paper presents an approach to address this challenge by proposing two predictive model based on Isolation Forest and Random Forest Classifier to assess the likelihood of an individual becoming a mass shooter based on the similarity between their behavioral attributes and those of known mass shooters. By leveraging the above cited machine learning techniques and specific datasets, we aim to uncover potential markers that might indicate an individual's predisposition to engage in mass violence. The findings underscore the validity of these techniques in their predictive capabilities. Nevertheless, due to the rarity of the mass shooting occurrences and the complex nature of the phenomenon, an objective evalutation is currently unfeasible.*

***Keywords**: mass shootings, gun violence, predictive model, behavioral attributes, machine learning, risk factors, public safety, sociocultural factors, psychological factors, environmental factors.*

**Introduction**

Gun violence, particularly in the form of mass shootings, has become a pervasive and deeply concerning issue in the United States. The alarming frequency and devastating impact of these incidents have prompted researchers, policymakers, and the general public to seek comprehensive solutions that can potentially prevent future occurrences. As the nation grapples with the complex challenge of identifying individuals who might be at risk of committing mass shootings, the integration of advanced technology and data analysis has emerged as a promising avenue for proactive intervention.

This paper aims to develop a predictive model that assesses the likelihood of an individual becoming a mass shooter by analyzing the similarities between their behavioral patterns and those of known mass shooters. By analyzing a broad range of attributes, including psychological profiles, social interactions, and personal histories, we endeavor to uncover potential markers that might indicate an individual's predisposition to engage in such violent acts. The rise of data-driven approaches in various fields, coupled with the increasing availability of comprehensive datasets, provides an unprecedented opportunity to delve into the intricate dynamics of gun violence. In the following sections, we will delve into the existing literature on gun violence and mass shootings, highlighting the sociocultural, psychological, and environmental factors that have been associated with these incidents. We will then outline our proposed methodology, which combines data analysis techniques with carefully curated datasets, to construct a predictive model. It is our hope that the insights gained from this research will contribute to the development of proactive strategies for preventing mass shootings and enhancing public safety.


**Literature Review**

The first study we will refer to titled "*Public Mass Shootings: Database Amasses Details of a Half Century of U.S. Mass Shootings with Firearms, Generating Psychosocial Histories*" [1]. This paper revealed that mass shootings are alarmingly frequent in the United States, occurring approximately every 12.5 days. This research made significant strides by constructing psychosocial histories of mass shootings within the U.S., deepening our understanding of these tragic events. Moreover the researchers created a comprehensive dataset on the psychological and sociological histories of mass shooting perpetrators from 1960 until today. Another relevant study, "*Study: Two-Thirds of Mass Shootings Linked to Domestic Violence*" [2], analyzed data from the Gun Violence Archive spanning 2014 to 2019. It unveiled a troubling trend where over two-thirds of mass shootings were connected to domestic violence, either through the killing of family or intimate partners or a documented history of domestic violence by the perpetrators. On the same path but with different results from the previous study, "*Domestic Violence and Mass Shootings*" [3] examined the complex relationship between domestic abuse history and the risk of mass shootings. However, despite numerous efforts to address this question, the study concluded that results remain inconclusive, underscoring the intricate nature of the issue. In the paper "*Mental Illness, Mass Shootings, and the Future*" [4], researchers explored the connection between mental illness and mass shootings. While acknowledging that some high-profile mass shooters exhibited clinical symptoms, the study highlighted the ambiguity surrounding the role of psychopathology in mass shootings. These works, establish a starting point by identifying some key characteristics of the average mass shooting perpetrator. This progresses opened the doors for a new approach to this problem by allowing the modeling of an ML-based risk assessment system that can predict a person's tendency to become a mass shooter.  In “An ML-Powered Risk Assessment System for Predicting Prospective Mass Shooting“ [5] the researchers build a system that uses two ML models, local outlier factor and K-means clustering, to learn both the psychological factors and social media behavior of potential mass shooters. However, this study is not without controversy, as the use of clustering to predict mass shootings has been criticized by some researchers and advocates. While it is not uncommon for mass shootings to cluster in time, earlier studies did not find evidence of clustering of rampage murders. Overall, while this study have explored the use of clustering and machine learning to predict mass shootings, the effectiveness and accuracy of these methods remain a subject of debate and further research is needed.

Our paper aims to contribute to this body of research by proposing a machine learning-based approach for predicting the likelihood of an individual becoming a mass shooter. We will leverage “the Violence Project Database”, to develop a tool that can identify individuals at potential risk of committing mass shootings, enabling early intervention to hopefully prevent such tragic events.

**But, are ML methods a valid choice for predicting future mass shooters?**

Machine learning methods offers the potential for early identification of individuals exhibiting concerning patterns in their behavior or sociological and environmental conditions. This allows for timely intervention or further investigation, taking a holistic approach to threat assessment. Moreover, ML allow us to objectively analyze datasets, revealing patterns that may elude human observation due to bias or cognitive limitations across a wide array of attributes making them well-suited for analyzing a broad spectrum of potential threats, considering behavioral, sociological, and environmental dimensions. However, the quality of training data and potential biases within it can significantly impact the accuracy and fairness of the models, especially when dealing with multiple attribute types. ML models, particularly those using anomaly detection, can produce false positives, potentially subjecting innocent individuals to unwarranted scrutiny or intervention. This risk is heightened when multiple attribute types are considered and as in our case. Another issue is that profiling individuals based on a wide range of attributes raises ethical considerations concerning privacy, discrimination, and stigmatization. Therefore it's essential to address these concerns, ensuring that individual rights are respected throughout the process. Employing ML predictions for law enforcement or surveillance purposes may also entail legal challenges, including issues related to due process and civil liberties, particularly when an extensive range of attributes is considered. Moreover, predicting mass shooters based on a comprehensive set of attributes is challenging due to the rarity of such events and the inherent limitations in predictive power, especially when multiple dimensions are involved.

As we highlighted above, a substantial body of research highlights the role of psychological factors in shaping an individual's propensity for violent behavior, including mass shootings. Psychological traits such as a history of mental illness, a propensity for aggression, and a fascination with violence have been examined in the context of mass shooters. The interplay between these factors and an individual's emotional state, cognitive processes, and decision-making patterns can provide valuable insights into the potential emergence of violent tendencies. Although, the intrinsic strong influence that sociological factors have on people’s psyche seem to overlap the importance of considering psychological factors alone.

The societal context in which an individual lives and interacts can significantly influence their behavior. Sociological factors encompass a wide range of elements, including socialization, peer influence, family dynamics, economic disparities, and exposure to violent media. Examining how these factors contribute to the development of a person's values, beliefs, and coping mechanisms can help uncover the pathways that lead some individuals down a violent trajectory.

While psychological and sociological factors are often treated as separate aspects of human behavior, they are intricately intertwined. An individual's psychological makeup can interact with their social environment, leading to a complex interplay that may influence their likelihood of engaging in violent acts. By considering both dimensions simultaneously, we aim to develop a more holistic understanding of the factors that contribute to mass shootings. In summary, our decision to focus on psychological and sociological factors stems from the recognition that a deeper exploration of these aspects holds the key to building a predictive model that can identify individuals at risk of becoming mass shooters.


**Methodology**


***Dataset***

We will refer to a dataset called the "Violence Project Database" for our research. This dataset contains sociocultural and psychological histories about mass shootings perpetrators in the United States. When we talk about "mass shootings," it can mean different things. In this database, the authors use the definition from the Congressional Research Service:

“a multiple homicide incident in which four or more victims are murdered with firearms—not including the offender(s)—within one event, and at least some of the murders occurred in a public location or locations in close geographical proximity (e.g., a workplace, school, restaurant, or other public settings), and the murders are not attributable to any other underlying criminal activity or commonplace circumstance (armed robbery, criminal competition, insurance fraud, argument, or romantic triangle).”


The main purpose of this work is to help us understand mass shooters better and find ways to prevent such tragic events. This dataset covers almost 200 mass shootings that happened between 1960 and 2023. It provides details about each incident, with a total of 154 features for each one. We will consider the following key features:



|**COLUMN NAME**|**DATA TYPE**|**DESCRIPTION**|
| :-: | :-: | :-: |
|State Code|INT|From 1 to 52|
|School Performance|INT|| 0  Poor | 1  Average | 2  Good ||
|Younger Siblings|INT|From 0 to N. # younger siblings|
|Older Siblings|INT|From 0 to N. #  older siblings|
|Childhood Socioeconomic Status|INT|| 0  Low | 1  Middle  | 2  Upper ||
|History of Domestic Abuse|INT|| 0 – No evidence | 1 – Evidence ||
|Hate Group or Chat Room Affiliation|INT|| 0 – No evidence | 1 – Evidence ||
|Childhood Trauma|INT|| 0 – No evidence | 1 – Evidence ||
|Physically Abused|INT|| 0 – No evidence | 1 – Evidence ||
|Sexually Abused|INT|| 0 – No evidence | 1 – Evidence ||
|Parental Substance Abuse|INT|| 0 – No evidence | 1 – Evidence ||
|Sign of Being in Crisis|INT|| 0 – No evidence | 1 – Evidence ||
|Suicidality|INT|| 0 – No evidence | 1 – Evidence ||

In our research, we applied a sequential methodology encompassing feature selection, anomaly detection and randomforest classification for predictive modeling, as detailed below:


**Feature Selection**

To pinpoint the most significant attributes within our dataset, our initial step involved conducting a Correlation Matrix analysis. This examination enabled us to grasp the interconnections between variables and identify potential instances of multicollinearity. Features displaying substantial pairwise correlations were systematically excluded to enhance the dataset's robustness and interpretability. Subsequently, we evaluated the independence of our variables by employing the "*mutual\_info\_classif*" method from the sklearn library to measure mutual information. This step ensured that all variables remained free from interdependencies and allowed us to reduce the amount of variables from 154 to 28 . To further validate our feature selection, we utilized the *SelectFromModel* library, incorporating a *RandomForestClassifier* as our estimator. This additional validation step supports the integrity of our chosen features, contributing to the overall reliability of our analysis.

**Anomaly Detection model as classifier**

For anomaly detection within the dataset, we utilized the Isolation Forest algorithm. Isolation Forest is a machine learning algorithm designed for anomaly detection. It efficiently isolates anomalies (outliers) in a dataset by partitioning it into subsets using random feature selection, making it highly effective for identifying rare and unusual data points. This unsupervised learning method effectively isolated data points that deviated significantly from the majority of the dataset. We found that, depending on the contamination parameters from 5% to 20% of the mass shooters records were considered anomalies in respect to the whole dataset. Therefore we choose to set a high contamination parameter so that the model is more likely to produce false negatives but at the same time, more capable of detecting the true positives. By removing the outliers and setting a threshold under wich we discard the prediction, we observed that the model is capable of correctly identify 100% of mass shooters observation, however, without data manipulation, the accuracy scores around 75 – 80 %.

**Classification with RandomForestClassifier**

In the second phase, we constructed a Random Forest Classifier for the prediction of new potential mass shooters. Building upon the feature selection process, this classifier was designed to classify an individual as potential mass shooters or not. To optimize the model's performance, we employed a Grid Search approach to fine-tune its hyperparameters systematically. We found out that the best parameters were the following: *{'max\_depth': None, 'min\_samples\_leaf': 1, 'min\_samples\_split': 2, 'n\_estimators': 50}*. This comprehensive approach ensured that our predictive model achieved the highest levels of accuracy and generalizability.


**Analysis and Evaluation:**

We performed a cross validation test with 5 folds for both Isolation Forest and RandomForestClassifier.  The results are shown below:



||IForest|RndForestClass|
| :-: | :-: | :-: |
|Accuracy score|` `0.82|1|
|Precision score|1|1|

The interpretation of these results can be msleading, due to the peculiarity of our dataset. By analyzing the nature of the phenomenon and its specificity, we know that, for example, 97% of mass shooters are man, most of them are in lower-middle classes, they tends to be suicidal and most of the times show sign of crisis before perpetrating such acts. With this information, we generated records that represents people that are most likely to be non mass shooters. This fake observations are correctly detected as anomalies by the Isolation Forest and correctly classified as non shooter s by the Random Forest Classifier. However, many of these observations becomes false positives when the records are less clear in terms of features that strongly identify them as non mass shooters meaning that the boundaries between shooters and non shooters are not clearly defined.



**Conclusions**

Our research recognized the potential value of machine learning methods in threat assessment, taking into account a wide array of attributes, including behavior, sociological and environmental context. However, applying these methods to predict future mass shooters while considering ethical, legal, and practical challenges is a complex endeavor. The holistic approach to threat assessment, encompassing various dimensions, underscores the importance of addressing the root causes of mass shootings, including mental health, media exposure, access to firearms, and societal factors in general.

In summary, while machine learning methods can contribute to threat assessment, their application in predicting mass shooters must be undertaken with utmost caution, transparency, and adherence to ethical and legal principles. Predictions should be integrated into a comprehensive strategy that encompasses intervention, prevention, and the holistic addressing of underlying factors contributing to violence.



***References***

1\. Fox, J. A., & Fridel, E. E. (2022). Public Mass Shootings: Database Amasses Details of a Half Century of U.S. Mass Shootings with Firearms, Generating Psychosocial Histories. NIJ Journal, 282, 2-13.

2\. Geller, L., Booty, M., & Crifasi, C. K. (2021). Study: Two-Thirds of Mass Shootings Linked to Domestic Violence. Injury Epidemiology, 8(1), 1-6. doi: 10.1186/s40621-021-00333-4

3\. Kivisto, A. J., & Phalen, P. L. (2018). Domestic Violence and Mass Shootings. Journal of Interpersonal Violence, 33(16), 2515-2534. doi: 10.1177/0886260517696871

4\. Swanson, J. W., McGinty, E. E., Fazel, S., & Mays, V. M. (2021). Mental Illness, Mass Shootings, and the Future. JAMA Network Open, 4(1), e2036222. doi: 10.1001/jamanetworkopen.2020.36222

5\. Zhang, Y., & Li, Y. (2022). An ML-Powered Risk Assessment System for Predicting Prospective Mass Shooting. Computers, 12(2), 42. doi: 10.3390/computers12020042



