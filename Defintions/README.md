## Some Basic Definitions

**Statistics** - It is the study of the collection, analysis, interpretation,presentation, and organization of data.

**Data Mining** - It is an interdisciplinary subfield of computerscience. It is the computational process of discovering patterns in large data sets (from data warehouse) involving methods at the intersection of artificial intelligence, machine learning, statistics and database systems.

**Data Analytics** - It is a process of inspecting, cleaning, transforming,and modeling data with the goal of discovering useful information, suggesting conclusions, and supporting decision making. This is also known as Business Analytics and is widely used in many industries to allow companies/organization to use the science of examining raw data with the purpose of drawing conclusions about that information and make better business decisions.

**Data Science** -  Data science is an interdisciplinary field about processes and systems to extract knowledge or insights from data in various forms, either structured or unstructured, which is a continuation of some of the data analysis fields such as statistics, machine learning, data mining, and predictive analytics, similar to Knowledge Discovery in Databases (KDD).

**Bayesian** - It describes that the probability of an event, based on conditions that might be related to the event.

**Bayes’s theorem** -  It describes the outcome probabilities of related (dependent) events using the concept of conditional probability.
Bayes theorem is stated mathematically as the following equation:
    
      P(A|B) = P(B|A) * P(A) / P(B)
&nbsp;&nbsp;&nbsp;&nbsp;Where A and B are events and P (B) ≠ 0

&nbsp;&nbsp;&nbsp;&nbsp;P(A) and P(B) are the probabilities of observing A and B without regard to each other.

&nbsp;&nbsp;&nbsp;&nbsp;P(A|B), a conditional probability, is the probability of observing event A given that B is true.

&nbsp;&nbsp;&nbsp;&nbsp;P(B|A) is the probability of observing event B given that A is true

**Data Analatics** - **4 Types**

&nbsp;&nbsp;&nbsp;&nbsp;**Descriptive Analytics** - Describes the past to tell us “What has happened?” any activity or method that helps us to describe or summarize raw data into something interpretable by humans can be termed ‘Descriptive Analytics’. eg - count, min, max, sum, average,percentage, and percent change
    
&nbsp;&nbsp;&nbsp;&nbsp;**Diagnostic Analytics** - Next step to the descriptive analytics that examines data or information to answer the question, “Why did it happen?,” and it is characterized by techniques such as drilldown, data discovery, data mining, correlations, and causation. for example, Excel, Tableau, Qlikview, Spotfire, and D3, etc., are available build tools that enable diagnostic analytics.

&nbsp;&nbsp;&nbsp;&nbsp;**Predictive Analytics** - It is the ability to make predictions or estimations of likelihoods about unknown future events based on the past or historic patterns. Predictive analytics will give us insight into “What might happen?”; it uses many techniques from data mining, statistics, modeling, machine learning, and artificial intelligence to analyze current data to make predictions about the future.

&nbsp;&nbsp;&nbsp;&nbsp;**Prescriptive Analytics** - Prescriptive analytics is related to all other three forms of analytics that is, descriptive, diagnostic, and predictive analytics.The endeavor of prescriptive analytics is to measure the future decision’s effect to enable the decision makers to foresee the possible outcomes before the actual decisions are made. Prescriptive analytic systems are a combination of business rules, machine learning algorithms, tools that can be applied against historic and real-time data feed.

**Machine Learning** - It is a collection of algorithms and techniques used to create computational systems that learn from data in order to make predictions and inferences.

**Machine learning** tasks can be categorized into three groups based on the desired output and the kind of input required to produce it.

+ **Supervised Learning** - goal of the algorithm is to learn patterns in the data and build a general set of rules to map input to the class or event.
there are two types commonly used as supervised learning algorithms.

    - **Regression** - The output to be predicted is a continuous number in relevance with a given input dataset.

    - **Classification** - The output to be predicted is the actual or the probability of an event/class and the number of classes to be predicted can be two or more.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Building supervised learning machine learning models has three stages:**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Training** : The algorithm will be provided with historical input data with the mapped output. The algorithm will learn the patterns within the input data for each output and represent that as a statistical equation, which is also commonly known as a model.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Testing or validation** : In this phase the performance of the trained model is evaluated, usually by applying it on a dataset (that was not used as part of the training) to predict the class or event.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Prediction** : Here we apply the trained model to a data set that was not part of either the training or testing. The prediction will be used to drive business decisions

+ **Unsupervised Learning** - used when the desired output class/event is unknown for historical data.

    - **Clustering** - Assume that the classes are not known beforehand for a given dataset. The goal here is to divide the input dataset into logical groups of related items.
    
    - **Dimension Reduction** - The goal is to simplify a large input dataset by mapping them to a lower dimensional space.
    
    - **Anomaly Detection** - Anomaly detection is also commonly known as outlier detection is the identification of items, events or observations which do not conform to an expected pattern or behavior in comparison with other items in a given dataset.

+ **Reinforcement Learning** - The basic objective of reinforcement learning algorithms is to map situations to actions that yield the maximum final reward. While mapping the action, the algorithm should not just consider the immediate reward but also next and all subsequent rewards. Some reinforcement learning techniques are the following:

    - Markov decision process
    - Q-learning
    - Temporal Difference methods
    - Monte-Carlo methods
    
**Frameworks for building ML Systems** - These process frameworks guide and carry the machine learning tasks and its applications. Efforts were made to use data mining process frameworks that will guide the implementation of data mining on big or huge amount of data. Mainly three data mining process frameworks have been most popular, and widely practiced by data mining experts/researchers to build machine learning systems. These models are the following:
+ **Knowledge Discovery Databases (KDD) process model** - It is an integration of multiple technologies for data management such as data warehousing, statistic machine learning, decision support, visualization, and parallel computing. As the name suggests, Knowledge Discovery Databases center around the overall process of knowledge discovery from data that covers the entire life cycle of data that includes how the data are stored, how it is accessed, how algorithms can be scaled to enormous datasets efficiently, how results can
  be interpreted and visualized. There are five stages in KDD
  
  - **Selection** - In this step, selection and integration of the target data from possibly many different and heterogeneous sources is performed. Then the correct subset of variables and data samples relevant to the analysis task is retrieved from the database.
  - **Preprocessing** - Preprocessing and cleaning should improve the quality of data and mining results by enhancing the actual mining process.
  - **Transformation** - In this step, data is transformed or consolidated into forms appropriate for mining, that is, finding useful features to represent the data depending on the goal of the task. data transformation techniques: Smoothing,Aggregation,Generalization,Normalization,Feature construction,Data reduction techniques,Compression,
  - **Data Mining** - In this step, machine learning algorithms are applied to extract data patterns.
  - **Interpretation / Evaluation** - This step is focused on interpreting the mined patterns to make them understandable# by the user, such as summarization and visualization.
  
 + **CRoss Industrial Standard Process for Data Mining (CRISP – DM)** - This framework is an idealized sequence of activities. It is an iterative process and many of the tasks backtrack to previous tasks and repeat certain actions to bring more clarity. There are six major phases
   - **Business Understanding** - the focus at this stage is to understand the overall project objectives and expectations from a business perspective
   - **Data Understanding** - initial data are collected that were identified as requirements in the previous phase.
   - **Data Preparation** - This phase is all about cleaning the data so that it’s ready to be used for the model building phase
   - **Modeling** - various appropriate machine learning algorithms are applied onto the clean dataset, and their parameters are tuned to the optimal possible values.
   - **Evaluation** - In this stage a benchmarking exercise will be carried out among all the different models that were identified to have been giving high accuracy.
   - **Deployment** - focus in this phase is the usability of the model output.
        
 + **Sample, Explore, Modify, Model and Assess (SEMMA)** - five sequential steps to understand it better.
   - **Sample** - This step is all about selecting the subset of the right volume dataset from a large dataset provided for building the model
   - **Explore** - In this phase activities are carried out to understand the data gaps and relationship with each other.(UniVariate,MultiVariate Analysis)
   - **Modify** - In this phase variables are cleaned where required. New derived features are created by applying business logic to existing features based on the requirement. Variables are transformed if necessary.
   - **Model** - In this phase, various modeling or data mining techniques are applied on the preprocessed data to benchmark their performance against desired outcomes.
   - **Assess** - This is the last phase. Here model performance is evaluated against the test data (not used in model training) to ensure reliability and business usefulness.
