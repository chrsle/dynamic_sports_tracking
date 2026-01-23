# Scientific Papers from Other Sports Applicable to Hockey

A comprehensive reference of academic research from baseball, basketball, soccer, football, and other sports that could be translated to hockey analytics.

---

## Table of Contents

1. [Action Valuation & Expected Value Frameworks](#1-action-valuation--expected-value-frameworks)
2. [Spatial Control & Pitch/Ice Ownership](#2-spatial-control--pitchice-ownership)
3. [Pass Networks & Graph Neural Networks](#3-pass-networks--graph-neural-networks)
4. [Trajectory Prediction & Movement Modeling](#4-trajectory-prediction--movement-modeling)
5. [Reinforcement Learning for Decision-Making](#5-reinforcement-learning-for-decision-making)
6. [Injury Prediction & Workload Management](#6-injury-prediction--workload-management)
7. [Defensive Analysis & Pressing Metrics](#7-defensive-analysis--pressing-metrics)
8. [Player Tracking & Computer Vision](#8-player-tracking--computer-vision)
9. [Shot Quality & Deception Metrics](#9-shot-quality--deception-metrics)
10. [Win Probability Models](#10-win-probability-models)
11. [Player Embeddings & Similarity Models](#11-player-embeddings--similarity-models)
12. [Formation & Tactical Detection](#12-formation--tactical-detection)
13. [Group Activity Recognition](#13-group-activity-recognition)
14. [Integrated Datasets & Benchmarks](#14-integrated-datasets--benchmarks)

---

## 1. Action Valuation & Expected Value Frameworks

### VAEP: Valuing Actions by Estimating Probabilities

**Paper:** Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). "Actions Speak Louder than Goals: Valuing Player Actions in Soccer." *ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 1851-1861.

**Source Sport:** Soccer

**Methodology:**
- Assigns value to each on-ball action based on its impact on game outcome
- Uses SPADL (Soccer Player Action Description Language) to standardize event data
- Computes 151 features from the last 3 actions of a game state
- Two probabilistic models: P(score) and P(concede) in next few actions
- Value = ΔP(scores) + (-ΔP(concedes))

**Hockey Translation:**
- Value every on-puck action (passes, carries, shots, dumps)
- Build HADL (Hockey Action Description Language)
- Account for hockey-specific actions: line changes, icing, forechecking

**Code Repository:** github.com/ML-KULeuven/socceraction

**Citation:**
```
@inproceedings{Decroos2019,
  author = {Decroos, Tom and Bransen, Lotte and Van Haaren, Jan and Davis, Jesse},
  title = {Actions Speak Louder Than Goals: Valuing Player Actions in Soccer},
  booktitle = {KDD '19},
  year = {2019},
  pages = {1851--1861}
}
```

---

### Expected Threat (xT)

**Paper:** Singh, K. (2018). "Introducing Expected Threat." Blog post.

**Original Framework:** Rudd, S. (2011). "A Framework for Tactical Analysis and Individual Offensive Production Assessment in Soccer Using Markov Chains." *New England Symposium on Statistics in Sports*.

**Source Sport:** Soccer

**Methodology:**
- Markov chain model assigning threat values to pitch zones
- xT(x,y) = s(x,y) × g(x,y) + m(x,y) × Σ T(x,y → z,w) × xT(z,w)
- Iterates until convergence (typically 4-5 iterations)
- Rewards actions that move ball to higher-threat zones

**Hockey Translation:**
- 20×10 grid for hockey rink (vs. 16×12 for soccer)
- Add behind-net zones for cycle plays
- Incorporate dump-in as action type
- Weight by game state (5v5, PP, PK)

**Code Repositories:**
- ML-KULeuven/socceraction (Python)
- gkrhines/xThreatR (R)
- sharmaabhishekk/xt-derivation-julia (Julia)

---

### Expected Possession Value (EPV)

**Paper:** Fernández, J., Bornn, L., & Cervone, D. (2019). "Decomposing the Immeasurable Sport: A Deep Learning Expected Possession Value Framework for Soccer." *13th MIT Sloan Sports Analytics Conference*.

**Basketball Version:** Cervone, D., D'Amour, A., Bornn, L., & Goldsberry, K. (2016). "A Multiresolution Stochastic Process Model for Predicting Basketball Possession Outcomes." *Journal of the American Statistical Association*, 111(514), 585-599.

**Source Sports:** Soccer, Basketball

**Methodology:**
- Frame-by-frame probability of scoring/conceding
- Decomposes into: pass value, ball drives, shot actions
- Uses deep learning on spatiotemporal tracking data
- Multiresolution stochastic process in basketball version

**Hockey Translation:**
- Continuous EPV using NHL EDGE tracking data
- Value every moment of possession, not just events
- Account for line changes as possession transitions

**Key Innovation:** Values the "off-ball" contributions that lead to good shooting positions.

---

### Critical Comparison: xT vs VAEP

**Paper:** Van Roy, M., Robberechts, P., Decroos, T., & Davis, J. (2020). "Valuing On-the-Ball Actions in Soccer: A Critical Comparison of xT and VAEP." *AAAI Workshop on Artificial Intelligence in Team Sports*.

**Key Findings:**
- xT: Location-only, simple Markov chain, interpretable
- VAEP: Action sequences, includes defensive value, requires more data
- Different models produce different top player rankings
- Major differences in how they value specific actions

**Hockey Implication:** Choose model based on data availability and interpretability needs.

---

## 2. Spatial Control & Pitch/Ice Ownership

### Voronoi Diagrams for Dominance Space

**Paper:** Efthimiou, C.J. (2021). "The Voronoi Diagram in Soccer: A Theoretical Study to Measure Dominance Space." *arXiv:2107.05714*.

**Extended Version:** Efthimiou, C.J. (2022). "A Physics-Driven Study of Dominance Space in Soccer." *arXiv:2202.00414*.

**Source Sport:** Soccer

**Methodology:**
- Traditional Voronoi: Pitch divided by proximity to players
- Physics-driven extension: Accounts for player velocity, acceleration, direction
- Introduces asymmetric influence (players control more in running direction)
- Includes frictional forces (air resistance, muscle energy consumption)

**Hockey Translation:**
- "Ice control" surfaces showing team dominance
- Weight by skating speed (hockey players move faster than soccer players)
- Account for goalie position and crease control
- Model defensive gap management

**Key Finding:** Standard Voronoi is only correct when all players have same speed.

---

### Pitch Control Models

**Paper:** Spearman, W. (2018). "Beyond Expected Goals." *MIT Sloan Sports Analytics Conference*.

**Paper:** Fernández, J., & Bornn, L. (2018). "Wide Open Spaces: A Statistical Technique for Measuring Space Creation in Professional Soccer." *MIT Sloan Sports Analytics Conference*.

**Source Sport:** Soccer

**Methodology:**
- Probability density function using bivariate Gaussian distribution
- Predicts likelihood of team gaining possession at any point
- Accounts for player velocity, ball travel speed, control time
- Visualized as continuous "heat map" of control probability

**Hockey Translation:**
- Real-time ice control visualization
- Measure space creation by off-puck skaters
- Evaluate defensive positioning relative to optimal

**Validation:** 91% accuracy predicting pass recipient using pitch control.

---

### Neighbor-Based Pitch Ownership (KNN Approach)

**Paper:** "A Neighbor-based Approach to Pitch Ownership Models in Soccer." (2025). *arXiv:2501.05870*.

**Source Sport:** Soccer

**Methodology:**
- Uses K-Nearest Neighbors algorithm
- More flexible than traditional Voronoi or Spearman models
- Introduces uncertainty via distance
- Combines multiple Voronoi diagram variants

**Hockey Translation:**
- Flexible ice ownership model
- Account for uncertainty in contested areas
- Combine with tracking data for real-time visualization

---

## 3. Pass Networks & Graph Neural Networks

### Network Science for Passing Analysis

**Paper:** Buldú, J.M., et al. (2018). "Using Network Science to Analyse Football Passing Networks: Dynamics, Space, Time, and the Multilayer Nature of the Game." *Frontiers in Psychology*, 9:1900.

**Paper:** Buldú, J.M., et al. (2019). "Defining a Historic Football Team: Using Network Science to Analyze Guardiola's F.C. Barcelona." *Scientific Reports*, 9:13602.

**Source Sport:** Soccer

**Methodology:**
- Teams represented as complex networks (players = nodes, passes = edges)
- Network metrics: clustering coefficient, betweenness centrality, eigenvector centrality
- Identifies unique team "signatures" and playing styles
- Analyzes temporal evolution of network structure

**Hockey Translation:**
- Build passing networks for hockey lines
- Identify playmaker vs. finisher roles via centrality metrics
- Detect recurring 3-4 player passing patterns (give-and-go, cycle plays)
- Compare line chemistry across different combinations

**Key Metrics:**
- Clustering coefficient (local connectivity)
- Betweenness centrality (players who connect others)
- Average path length (efficiency of ball movement)
- Network density (passing frequency)

---

### Graph Convolutional Networks for Defensive Analysis

**Paper:** Stats Perform AI Team. (2021). "Making Offensive Play Predictable - Using a Graph Convolutional Network to Understand Defensive Performance in Soccer." *MIT Sloan Sports Analytics Conference*.

**Source Sport:** Soccer

**Methodology:**
- Three models: xPass, xReceiver, xThreat
- Edge block + node block GNN architecture
- Measures defensive value by preventing actions before they occur
- Handles variable number of players and missing tracking data

**Hockey Translation:**
- Measure defensive "prevention" value (blocked passing lanes)
- Credit defensemen for shots that never happened
- Evaluate forechecking effectiveness
- Model defensive scheme classification

**Key Innovation:** "The aim of a defense is to make offensive play predictable."

---

### Complex Multiplex Passing Network (CMPN)

**Paper:** "CMPN: Modeling and Analysis of Soccer Teams Using Complex Multiplex Passing Network." *Chaos, Solitons & Fractals* (2023).

**Source Sport:** Soccer

**Methodology:**
- Multiple layers representing specific pass types
- Achieves >90% accuracy predicting attacking play outcomes
- Combines topological features with machine learning

**Hockey Translation:**
- Separate layers for: tape-to-tape passes, saucer passes, dump-ins, chip plays
- Predict zone entry success based on passing patterns
- Identify which pass types lead to high-danger chances

---

## 4. Trajectory Prediction & Movement Modeling

### Social LSTM for Player Trajectories

**Paper:** Alahi, A., et al. (2016). "Social LSTM: Human Trajectory Prediction in Crowded Spaces." *IEEE Conference on Computer Vision and Pattern Recognition*.

**Soccer Application:** github.com/jagjeet-singh/Social-lstm-for-predicting-player-trajectories-in-soccer

**Source:** Pedestrian tracking, adapted to soccer

**Methodology:**
- LSTM networks capture temporal dependencies
- "Social pooling" models interactions between nearby agents
- Predicts future positions based on historical trajectories
- Accounts for sudden direction changes

**Hockey Translation:**
- Predict player trajectories 1-3 seconds ahead
- Model defensive positioning reactions
- Evaluate off-puck movement quality
- Simulate "ghosting" (what optimal defenders would do)

---

### Defender CNN-LSTM Model

**Paper:** Amazon Science. "Prediction of Defensive Player Trajectories in NFL Games with Defender CNN-LSTM Model."

**Source Sport:** American Football

**Methodology:**
- 1D convolutions extract features from multiple players
- LSTM captures temporal dependencies
- Social pooling accounts for nearby player interactions
- Predicts defensive back trajectories based on perceived target receiver

**Hockey Translation:**
- Predict defenseman reactions to puck movement
- Model backchecking trajectories
- Evaluate goalie positioning based on shooter location

**Key Insight:** Individual trajectories affected by personal assignments, overall strategy, and surrounding player movements.

---

### Deep Learning for Basketball Trajectories

**Paper:** Shah, R., et al. "Applying Deep Learning to Basketball Trajectories." *Semantic Scholar*.

**Source Sport:** Basketball

**Methodology:**
- RNNs predict whether 3-point shots will be successful
- Models learn ball trajectory without physics knowledge
- Uses SportVU tracking data (25 fps)
- Outperforms feature-rich ML models

**Hockey Translation:**
- Predict shot success based on puck trajectory
- Model deflection probability
- Evaluate passing accuracy predictions

---

## 5. Reinforcement Learning for Decision-Making

### Q-Ball: Deep RL for Basketball

**Paper:** Yanai, C., Solomon, A., Katz, G., Shapira, B., & Rokach, L. (2022). "Q-Ball: Modeling Basketball Games Using Deep Reinforcement Learning." *AAAI Conference on Artificial Intelligence*.

**Source Sport:** Basketball

**Methodology:**
- Models latent connections among player movements, actions, performance
- Assigns scores to player and team performance
- Applications: evaluating game decisions, tactical recommendations
- Uses SportVU tracking data

**Hockey Translation:**
- Evaluate decision quality (shoot vs. pass vs. carry)
- Optimize line deployment based on game state
- Generate tactical recommendations for coaches

---

### Double-Teaming Strategy with Deep RL

**Paper:** Wang, J.R., et al. (2018). "The Advantage of Doubling: A Deep Reinforcement Learning Approach to Studying Double Teams in the NBA." *arXiv:1803.02940*.

**Source Sport:** Basketball

**Methodology:**
- Deep RL learns optimal double-team strategies
- NothingButNet (NBNet) CNN architecture
- Maps state-action pairs to expected cumulative reward
- Compares learned strategy vs. actual strategies

**Hockey Translation:**
- Optimize forechecking strategies
- Model when to pinch vs. stay back as defenseman
- Evaluate aggressive vs. conservative goalie positioning

---

### ReLiable: Offline RL for Tactical Strategies

**Paper:** "ReLiable: Offline Reinforcement Learning for Tactical Strategies in Professional Basketball Games." *ACM CIKM 2022*.

**Source Sport:** Basketball

**Methodology:**
- Offline Deep Q-Network trained on historical NBA data
- Guides player decisions in real-time
- Addresses heterogeneous signals and massive action/outcome space
- Handles incomplete observations

**Hockey Translation:**
- Train on historical NHL play-by-play data
- Suggest optimal actions given game state
- Account for incomplete tracking data

---

### Hierarchical Deep RL for Team Sports

**Paper:** Meng, X. (2025). "AI-powered Tactical Optimization in Dynamic Team Sports: A Hierarchical Reinforcement Learning Approach." *ScienceDirect*.

**Source Sports:** Basketball, Soccer, Rugby

**Key Results:**
- 34.7% improvement in tactical accuracy
- 28.3% improvement in decision-making speed
- 41.2% improvement in computational efficiency
- 23.6% improvement in scoring efficiency (NBA G-League)

**Hockey Translation:**
- Multi-level decision hierarchy (game → period → shift → play)
- Real-time tactical suggestions
- Human-AI collaboration in coaching

---

## 6. Injury Prediction & Workload Management

### ACWR and Machine Learning

**Paper:** Tsilimigkras, G., et al. "Machine Learning-Based Prediction of Muscle Injury Risk in Professional Football." *Sports Medicine - Open* (2022).

**Source Sport:** Soccer/Football

**Methodology:**
- Acute to Chronic Workload Ratio (ACWR)
- Deviation of Maximum from Average (DEV)
- Support Vector Machine-based feature selection
- Accuracy: 0.78, Sensitivity: 0.73, Specificity: 0.85

**Hockey Translation:**
- Build ACWR using: distance skated, high-intensity events, shift patterns
- Monitor ice time spikes relative to seasonal baseline
- Predict soft tissue injury risk windows

**Key Features:**
- GPS training data (total distance, high-speed running)
- Physiological responses (heart rate)
- Previous injury history
- Sleep quality and recovery metrics

---

### Footballer Workload Footprint (FWF)

**Paper:** "Advanced Feature Engineering in Acute:Chronic Workload Ratio (ACWR) Calculation for Injury Forecasting in Elite Soccer." *PLOS ONE* (2025).

**Source Sport:** Soccer

**Methodology:**
- Represents workload variables as discrete time series
- Temporal matrix (FWF) integrates multiple workload variables
- Calculus-based (integral/differential) operations
- More sophisticated than traditional ACWR

**Hockey Translation:**
- Build FWF matrix from NHL EDGE tracking data
- Include: distance, acceleration events, time on ice, shift intensity
- Apply calculus-based transformations for injury risk

---

### Comprehensive Monitoring Approach

**Paper:** "Predicting Injury and Illness with Machine Learning in Elite Youth Soccer: A Comprehensive Monitoring Approach over 3 Months." *PMC* (2023).

**Source Sport:** Youth Soccer

**Variables Used (65 total):**
- Training load (EWMA of last 2 sessions)
- ACWR ratios
- Sleep quality and duration
- Jump height (CMJ)
- Blood markers (cfDNA, CRP, ferritin)

**Key Findings:**
- Sleep quality was most important predictor for injury
- ACWR of tempo runs ranked #2
- Blood variables important for illness prediction

**Hockey Translation:**
- Integrate multiple data sources (tracking, biometric, subjective)
- Monitor sleep and recovery alongside workload
- Use blood markers for return-to-play decisions

---

### GPS Metrics and ML for Rugby

**Paper:** "Global Positioning System-Derived Metrics and Machine Learning Models for Injury Prediction in Professional Rugby Union Players." *PMC* (2025).

**Source Sport:** Rugby

**Methodology:**
- 17 GPS-derived metrics
- Multiple EWMA ACWR ratios (3:14, 3:21, 7:21, 7:28)
- Monotony and strain calculations
- Position-specific models

**Hockey Translation:**
- Position-specific injury models (forwards vs. defensemen vs. goalies)
- Multiple ACWR timeframes for different injury types
- Account for contact (hits, blocks) and non-contact workload

---

## 7. Defensive Analysis & Pressing Metrics

### exPress: Contextual Pressing Valuation

**Paper:** Stephanos, K. (2025). "exPress: Contextual Valuation of Individual Players Within Pressing Situations in Soccer." *MIT Sloan Sports Analytics Conference*.

**Source Sport:** Soccer

**Methodology:**
- Values individual player contributions to pressing
- Context-aware (accounts for game state, opponent quality)
- Measures disruption of opponent build-up play

**Hockey Translation:**
- Value forechecking contributions
- Measure forced turnovers and their quality
- Credit players for disrupting breakouts

---

### Defensive Skill Spatial Structure

**Paper:** Franks, A., Miller, A., Bornn, L., & Goldsberry, K. (2015). "Characterizing the Spatial Structure of Defensive Skill in Professional Basketball." *Annals of Applied Statistics*, 9:194-121.

**Source Sport:** Basketball

**Methodology:**
- Spatial analysis of defensive positioning
- Measures influence on opponent shooting performance
- Uses tracking data to quantify "good defense"

**Hockey Translation:**
- Evaluate defensive positioning relative to shooter
- Credit defensemen for reducing shot quality
- Model "defensive gravity" (attention drawn)

---

## 8. Player Tracking & Computer Vision

### Learning Feature Representations from Tracking

**Paper:** Horton, M. (2020). "Learning Feature Representations from Football Tracking." *14th MIT Sloan Sports Analytics Conference*.

**Source Sport:** American Football

**Methodology:**
- Deep learning on raw tracking data
- Learns meaningful features without hand-engineering
- Generalizes across different plays and situations

**Hockey Translation:**
- Learn representations from NHL EDGE data
- Discover patterns not visible to human analysts
- Build foundation models for multiple downstream tasks

---

### AutoStats: Computer Vision for College Basketball

**Paper:** Stats Perform. (2021). "Predicting NBA Talent from Enormous Amounts of College Basketball Tracking Data." *MIT Sloan Sports Analytics Conference*.

**Source Sport:** Basketball

**Methodology:**
- Computer vision generates tracking data from video
- Predicts NBA-level skills from college data
- Identifies traits not measurable before

**Hockey Translation:**
- Generate tracking data from junior/college hockey video
- Predict NHL success from pre-draft data
- Evaluate international prospects without live tracking

---

## 9. Shot Quality & Deception Metrics

### Pitch Tunneling (Baseball)

**Paper:** Smith, B., & Smith, B. (2021). "Using Baseball Seams to Alter Pitch Direction: The Seam Shifted Wake." *Sports Engineering*.

**Original Research:** Baseball Prospectus (2017). Pitch Tunneling metrics.

**Source Sport:** Baseball

**Methodology:**
- Measures how similar different pitches appear at "tunnel point"
- Tunnel point: ~24 feet from plate, 175ms decision time
- Quantifies deception through trajectory overlap
- Identifies optimal release point consistency

**Hockey Translation:**
- "Shot tunneling" for goalies
- Measure how similar wrist shot vs. snapshot appear at release
- Quantify shooter deception through body mechanics
- Identify elite shooters who disguise shot type

---

### Seam-Shifted Wake (Puck Aerodynamics)

**Paper:** Smith, B. (Utah State). Research on non-Magnus forces from seam orientation.

**Source Sport:** Baseball

**Methodology:**
- Seam orientation creates forces independent of spin
- Causes movement not explained by traditional physics
- Measured via Hawk-Eye: observed vs. inferred spin axis

**Hockey Translation:**
- Puck flutter and aerodynamics
- Spin rates and orientation effects
- Optimal release mechanics for "knuckling" shots

---

## 10. Win Probability Models

### Bayesian In-Game Win Probability

**Paper:** Robberechts, P., Van Haaren, J., & Davis, J. (2021). "A Bayesian Approach to In-Game Win Probability in Soccer." *ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.

**Source Sport:** Soccer

**Methodology:**
- Models future goals as temporal stochastic process
- Uses 10 contextual features per team (goals, red cards, yellow cards, xT, momentum)
- Bayesian framework for uncertainty quantification
- Well-calibrated probabilities throughout match

**Hockey Translation:**
- Adapt for higher-scoring nature of hockey
- Include power play states as key context
- Model goalie pull scenarios
- Account for overtime/shootout possibilities

**Key Innovation:** Time-varying coefficient model captures momentum shifts.

---

### NFL Win Probability with Random Forests

**Paper:** Lock, D., & Nettleton, D. (2014). "Using Random Forests to Estimate Win Probability Before Each Play of an NFL Game." *Journal of Quantitative Analysis in Sports*, 10(2), 197-205.

**Paper:** Burke, B. (2010). "WPA Explained." *Advanced Football Analytics*.

**Source Sport:** American Football

**Methodology:**
- Features: score differential, time remaining, down, distance, field position
- Random forest or XGBoost classifiers
- Win Probability Added (WPA) for player evaluation

**Hockey Translation:**
- Features: score, time, period, manpower situation, zone
- WPA for evaluating clutch performance
- Real-time broadcast graphics

**Existing Hockey Work:** MoneyPuck Live In-Game Model (moneypuck.com)

---

### Simulation Study on Win Probability Estimation

**Paper:** Brill, R.S., et al. (2024). "Exploring the Difficulty of Estimating Win Probability: A Simulation Study." *arXiv:2406.16171*.

**Source Sport:** American Football (simulated)

**Key Findings:**
- Observational data has strong dependence structure (same game = same outcome)
- Effective sample size much smaller than raw sample size
- Naive bootstrapped confidence intervals too narrow
- Introduces "fractional bootstrap" for calibrated intervals

**Hockey Translation:**
- Important methodological considerations for model validation
- Need to account for within-game correlation
- Uncertainty quantification essential for decision support

---

## 11. Player Embeddings & Similarity Models

### NBA2Vec: Dense Player Representations

**Paper:** "NBA2Vec: Dense Feature Representations of NBA Players." *arXiv:2302.13386*.

**Source Sport:** Basketball

**Methodology:**
- Neural network predicts play outcomes from 10-player lineups
- 8-dimensional player embeddings learned end-to-end
- Skip-gram inspired architecture
- Embeddings capture position and play style

**Hockey Translation:**
- NHL2Vec for player representations
- Learn from shift-level outcomes
- Embeddings for line chemistry optimization
- Similarity search for replacement players

**Key Finding:** Guards and centers separate cleanly; forwards distributed between both.

---

### Football2Vec: NLP for Soccer

**Paper:** Magdaci, O. (2023). "Embedding the Language of Football Using NLP." *Towards Data Science*.

**Code Repository:** github.com/ofirmg/football2vec

**Source Sport:** Soccer

**Methodology:**
- Treats game as "language" with actions as "words"
- Word2Vec (Action2Vec) for 32-dimensional action embeddings
- Doc2Vec (PlayerMatch2Vec) for player-match representations
- Semantic analogies work (e.g., pass direction transformations)

**Hockey Translation:**
- Hockey2Vec using event sequences
- Actions: shots, passes, hits, blocks, faceoffs
- Player similarity for trade/draft analysis
- Style fingerprints for coaching decisions

---

### Play2Vec: Sports Play Retrieval

**Paper:** Wang, Z., et al. (2020). "Effective and Efficient Sports Play Retrieval with Deep Representation Learning." *KDD*.

**Code Repository:** github.com/zhengwang125/play2vec

**Source Sport:** Soccer

**Methodology:**
- Maps tracking sequences to grid-based segment matrices
- Builds "sports corpus" using Jaccard similarity
- Skip-gram model learns segment embeddings
- Denoising Sequence Encoder-Decoder (DSED) combines segments

**Hockey Translation:**
- Retrieve similar plays from historical database
- "Find me plays that look like this power play setup"
- Opponent tendencies analysis
- Teaching tool for player development

---

### GCN Player Recommendation System

**Paper:** "Footballer Player Recommendation Model Using Graph Convolutional Networks." *Springer* (2024).

**Source Sport:** Soccer

**Methodology:**
- K-means clustering creates player archetypes
- Graph built from cosine similarity between players
- GCN learns rich node embeddings
- 100% accuracy with shared-label aggregation

**Hockey Translation:**
- Build player similarity graphs from stats
- Recommend replacement signings
- Identify undervalued players in different leagues
- Draft prospect comparisons

---

### Transfer Market Valuation with SHAP

**Paper:** "When Interpretable Machine Learning Meets the Beautiful Game: A Predictive Analytics Approach to Soccer Player Valuation." *Sport, Business and Management* (2025).

**Source Sport:** Soccer

**Methodology:**
- Random Forest, XGBoost, Gradient Boosting for transfer fee prediction
- SHAP values for feature importance and explainability
- Key drivers: contract remaining, team rating, age
- 831 transfers from Big Five leagues

**Hockey Translation:**
- NHL contract/trade value prediction
- Identify market inefficiencies
- Explainable valuations for front office decisions
- RFA/UFA negotiation support

---

## 12. Formation & Tactical Detection

### SoccerCPD: Change-Point Detection for Formations

**Paper:** Kim, H., et al. (2022). "SoccerCPD: Formation and Role Change-Point Detection in Soccer Matches Using Spatiotemporal Tracking Data." *ACM SIGKDD*.

**Source Sport:** Soccer

**Methodology:**
- Discrete g-segmentation algorithm
- Role adjacency matrices represent formations
- Hamming distance for permutation sequences
- Unsupervised detection of tactical changes

**Hockey Translation:**
- Detect system changes (1-3-1, 1-2-2, box, etc.)
- Identify when teams switch between aggressive/conservative
- Automatic labeling of power play formations
- In-game tactical adjustment detection

---

### CNN-Based Phase of Play Detection

**Paper:** "What Happens to Your Team's Formation During a Match? A Tactical Deep Dive." (2025).

**Source Sport:** Soccer (Bundesliga)

**Methodology:**
- CNN classifies phases of play (build-up, mid-block, high-block, etc.)
- Formation detected within each phase
- Hierarchical clustering for formation archetypes
- 7 seasons of tracking data

**Hockey Translation:**
- Phase detection: offensive zone, neutral zone, defensive zone, transition
- Formation within each zone
- Forecheck vs. trap identification
- Breakout pattern classification

---

### Formation Identification Survey

**Paper:** "The Principles of Tactical Formation Identification in Association Football (Soccer) — A Survey." *Frontiers in Sports and Active Living* (2025).

**Source Sport:** Soccer

**Key Principles:**
1. Preprocessing: match segments, normalized locations
2. Data representation: average locations, hand-engineered features, graphs
3. Identification: templates or clustering
4. Evaluation: design and qualitative criteria

**Hockey Translation:**
- Comprehensive framework for hockey system classification
- Role-based vs. location-based representations
- Temporal vs. spatial aggregation trade-offs
- Validation approaches without ground truth

---

### Big Data and Tactical Analysis Review

**Paper:** Rein, R., & Memmert, D. (2016). "Big Data and Tactical Analysis in Elite Soccer: Future Challenges and Opportunities for Sports Science." *SpringerPlus*, 5:1410.

**Source Sport:** Soccer

**Key Themes:**
- Volume: 86-300 MB per match, 400 GB per season
- Variety: tracking, event, physiological, video
- Velocity: real-time processing requirements
- Machine learning for pattern discovery

**Hockey Translation:**
- NHL EDGE generates similar data volumes
- Integration challenges across data sources
- Real-time tactical feedback systems
- Theoretical frameworks for tactical behavior

---

## 13. Group Activity Recognition

### NETS: Neural Embeddings in Team Sports

**Paper:** Hauri, S., & Vucetic, S. (2022). "Group Activity Recognition in Basketball Tracking Data -- Neural Embeddings in Team Sports (NETS)." *arXiv:2209.00451*.

**Source Sport:** Basketball

**Methodology:**
- Transformer + LSTM architecture
- Team-wise pooling layer
- Recognizes group activities from tracking data
- Automatic labeling algorithm for training data

**Hockey Translation:**
- Recognize plays: cycle, give-and-go, dump-and-chase
- Defensive schemes: man-to-man, zone
- Special teams patterns
- Breakout recognition

---

### Multi-Task Learning for Player Tracking

**Paper:** "Multi-task Learning for Joint Re-identification, Team Affiliation, and Role Classification for Sports Visual Tracking." *arXiv:2401.09942* (2024).

**Source Sport:** Soccer

**Methodology:**
- Single backbone for three tasks
- Part-based player representations (PRTReID)
- Team clustering without predefined classes
- Role classification: player, goalkeeper, referee, staff

**Hockey Translation:**
- Jersey number recognition in hockey broadcasts
- Team identification from video
- Role classification including officials
- Enhanced tracking through occlusions

---

## 14. Integrated Datasets & Benchmarks

### IDSSE: Integrated Dataset for Soccer

**Paper:** "An Integrated Dataset of Spatiotemporal and Event Data in Elite Soccer." *Scientific Data* (2025).

**Source:** Bundesliga (DFL)

**Contents:**
- Synchronized tracking (25 Hz) and event data
- Match information (weather, attendance, formations)
- Multiple matches with ground truth annotations
- Code for visualization on GitHub

**Hockey Translation:**
- Need for similar integrated hockey datasets
- Synchronization challenges between data sources
- Benchmarking standards for hockey analytics
- Reproducibility in sports research

**GitHub:** github.com/spoho-datascience/idsse-data

---

### SoccerNet: Large-Scale Video Dataset

**Paper:** Deliège, A., et al. (2021). "SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos." *CVPR Workshop*.

**Source Sport:** Soccer

**Contents:**
- 500+ games, 764 hours of video
- 17 action classes, 300,000+ annotations
- Camera calibration, player tracking
- Game state reconstruction challenge

**Hockey Translation:**
- Foundation for hockey video understanding
- Action spotting: goals, saves, hits, fights
- Camera calibration for broadcast analytics
- Player tracking from video

---

## Summary: Priority Implementation Matrix

| Paper/Concept | Complexity | Data Required | Hockey Novelty | Publication Potential |
|---------------|------------|---------------|----------------|----------------------|
| xT for Zone Entries | Medium | Public PBP | High | LINHAC, Hockey Graphs |
| VAEP for Hockey | Medium | Event data | High | MIT Sloan, JQAS |
| Pass Network GNN | Medium | PBP + EDGE | High | MIT Sloan |
| Ice Control (Voronoi) | Low-Medium | Tracking | Medium | Academic journal |
| Shot Deception Metrics | High | Video/CV | Very High | MIT Sloan, peer review |
| Trajectory Prediction | High | Full tracking | High | CVPR, AAAI |
| ACWR Injury Model | Low | Basic stats | Medium | Sports Medicine |
| Pressing/Forechecking Value | Medium | Tracking | High | LINHAC |
| RL Decision Optimization | High | Historical PBP | Very High | Top-tier venue |

---

## Additional Cross-Sport Papers Worth Exploring

| Paper | Sport | Key Innovation | Hockey Application |
|-------|-------|----------------|-------------------|
| "Ghosting" (Lucey et al., 2013) | Basketball | Simulates optimal defender trajectories | Defensive positioning evaluation |
| "Bhostgusters" (Seidl et al., 2018) | Basketball | Interactive play sketching with AI | Coaching tool for play design |
| "DeepHoops" (Sicilia et al., 2019) | Basketball | Micro-action evaluation with DL | Shift-level player analysis |
| "At the Helm" (Chu et al., 2020) | Sailing | Multi-agent strategy optimization | Line deployment optimization |
| "Moneyball 2.0" (Brown, 2021) | Baseball | Market inefficiency exploitation | Cap management analytics |
| "Rally Analyzer" (Wei et al., 2016) | Tennis | Sequential point modeling | Faceoff sequence analysis |

---

## Key Conferences for Publication

1. **MIT Sloan Sports Analytics Conference** - Premier venue, all sports
2. **LINHAC (Linköping Hockey Analytics Conference)** - Hockey-specific, accepts student papers
3. **Journal of Quantitative Analysis in Sports (JQAS)** - Peer-reviewed academic journal
4. **KDD (Knowledge Discovery and Data Mining)** - Top ML conference, sports track
5. **AAAI Workshop on AI in Team Sports** - AI/ML focus
6. **Frontiers in Psychology/Sports Science** - Open access, interdisciplinary

---

## Code Repositories & Resources

| Repository | Language | URL | Use Case |
|------------|----------|-----|----------|
| socceraction | Python | github.com/ML-KULeuven/socceraction | VAEP, xT, SPADL |
| mplsoccer | Python | mplsoccer.readthedocs.io | Visualization |
| nhl-api-py | Python | github.com/coreyjs/nhl-api-py | NHL data access |
| statsbombpy | Python | github.com/statsbomb/statsbombpy | Event data format |
| Friends-of-Tracking | Python | github.com/Friends-of-Tracking-Data-FoTD | Tutorials |
| football2vec | Python | github.com/ofirmg/football2vec | Player embeddings |
| play2vec | Python | github.com/zhengwang125/play2vec | Play retrieval |
| idsse-data | Python | github.com/spoho-datascience/idsse-data | Integrated dataset |

---

## Summary of Categories

| Category | # Papers | Top Priority for Hockey |
|----------|----------|------------------------|
| Action Valuation | 5 | VAEP adaptation for hockey events |
| Spatial Control | 4 | Ice control with velocity weighting |
| Pass Networks/GNN | 5 | Line chemistry graph models |
| Trajectory Prediction | 3 | Defensive positioning prediction |
| Reinforcement Learning | 5 | Decision optimization for coaches |
| Injury Prediction | 6 | ACWR with tracking data |
| Defensive Analysis | 2 | Forechecking valuation |
| Player Tracking/CV | 3 | Tracking from broadcast video |
| Shot Quality | 2 | Shot deception metrics |
| Win Probability | 3 | Bayesian model with power play states |
| Player Embeddings | 6 | NHL2Vec for similarity/chemistry |
| Formation Detection | 5 | System change-point detection |
| Group Activity | 2 | Play pattern recognition |
| Integrated Datasets | 2 | Benchmark creation need |

**Total: 53+ papers**

---

## References (Selected)

1. Cervone, D., et al. (2016). "A Multiresolution Stochastic Process Model for Predicting Basketball Possession Outcomes." *JASA*, 111(514), 585-599.

2. Decroos, T., et al. (2019). "Actions Speak Louder than Goals: Valuing Player Actions in Soccer." *KDD '19*, 1851-1861.

3. Fernández, J., & Bornn, L. (2018). "Wide Open Spaces: A Statistical Technique for Measuring Space Creation in Professional Soccer." *MIT Sloan*.

4. Fernández, J., Bornn, L., & Cervone, D. (2019). "Decomposing the Immeasurable Sport." *MIT Sloan*.

5. Buldú, J.M., et al. (2018). "Using Network Science to Analyse Football Passing Networks." *Frontiers in Psychology*, 9:1900.

6. Efthimiou, C.J. (2021). "The Voronoi Diagram in Soccer." *arXiv:2107.05714*.

7. Van Roy, M., et al. (2020). "Valuing On-the-Ball Actions in Soccer: xT vs VAEP." *AAAI Workshop*.

8. Stats Perform. (2021). "Making Offensive Play Predictable - GCN for Defensive Performance." *MIT Sloan*.

9. Yanai, C., et al. (2022). "Q-Ball: Modeling Basketball Games Using Deep RL." *AAAI*.

10. "Machine Learning for Understanding and Predicting Injuries in Football." (2022). *Sports Medicine - Open*.

11. Robberechts, P., Van Haaren, J., & Davis, J. (2021). "A Bayesian Approach to In-Game Win Probability in Soccer." *ACM SIGKDD*.

12. Kim, H., et al. (2022). "SoccerCPD: Formation and Role Change-Point Detection." *ACM SIGKDD*.

13. Hauri, S., & Vucetic, S. (2022). "NETS: Neural Embeddings in Team Sports." *arXiv:2209.00451*.

14. Deliège, A., et al. (2021). "SoccerNet-v2: Dataset and Benchmarks for Broadcast Soccer Videos." *CVPR Workshop*.

15. Franks, A., et al. (2015). "Characterizing the Spatial Structure of Defensive Skill in Professional Basketball." *Annals of Applied Statistics*, 9:194-121.

---

*Document compiled: January 2026*
*Purpose: Reference for translating sports analytics research to hockey*
*Repository: dynamic_sports_tracking*
