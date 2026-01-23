"""
Hockey Analytics Module

This package implements cutting-edge sports analytics research translated
to hockey, filling identified gaps in hockey analytics.

Modules:
    expected_threat: Expected Threat (xT) Framework - Markov chain model
                    valuing every location and action on the ice

    zone_entry_xt: Zone Entry xT Model - Specialized xT for zone entries
                   with lane analysis and speed adjustments

    shot_deception: Shot Deception Metrics - Baseball pitch tunneling
                    translated to hockey shooter deception

    line_chemistry_gnn: Line Chemistry GNN - Graph Neural Network analysis
                        of passing networks and line effectiveness

    fatigue_acwr: Fatigue/ACWR Model - Acute:Chronic Workload Ratio
                  for injury risk and load management

    goalie_save_model: Enhanced Goalie Save Difficulty - Beyond GSAx
                       with positioning, screens, and sequences

    ice_control: Ice Control Model - Spatial dominance surfaces
                 and defensive scheme classification

    obso: Off-Ball Scoring Opportunities - Valuing off-puck
          movement and space creation

    transition_analysis: Transition/Counterattack Analysis - Odd-man
                        rush success prediction and recovery analysis

    nhl_api: NHL API Integration - Utilities for fetching NHL data
             including EDGE tracking metrics

    epv: Expected Possession Value - Continuous EPV framework
         for frame-by-frame valuation

    shooter_gravity: Shooter Gravity Metrics - Defensive attention
                    and space creation measurement

    win_probability: Bayesian Win Probability Model - Real-time
                    win probability with strength states and momentum

    vaep: VAEP (Valuing Actions by Estimating Probabilities) -
          Action-level valuation considering sequences and defense

    player_embeddings: NHL2Vec Player Embeddings - Dense vector
                      representations for similarity and chemistry

    tactical_detection: Tactical/Formation Detection - System
                       classification and change-point detection

    trajectory_prediction: Player Trajectory Prediction - Social LSTM
                          based movement prediction and ghosting

    play_recognition: Play Pattern Recognition - NETS-inspired
                     group activity recognition for hockey plays

    rl_decision: RL Decision Optimizer - Reinforcement learning
                 for line deployment and tactical decisions

    workload_footprint: Advanced Workload Footprint - FWF methodology
                       for injury prediction and load management

    forechecking: Forechecking Valuation - exPress-based model
                  for valuing pressure and forechecking actions

    contract_value: Contract/Trade Value Analyzer - SHAP-style
                   explanations for player valuations

    pass_model: Pass Probability Model - xPass framework
               for pass success prediction and playmaking

Research Origins:
    - Soccer: xT (Karun Singh), EPV (Fernández & Bornn), Pitch Control,
              VAEP (Decroos), SoccerCPD (Kim), Football2Vec
    - Basketball: EPV (Cervone), RAPTOR/EPM, Gravity metrics,
                  NBA2Vec, Social LSTM, NETS
    - Baseball: Pitch tunneling, Seam-shifted wake (Dr. Barton Smith)
    - Football: ACWR injury prediction, GNN passing networks,
                Win probability models, Defender CNN-LSTM

References:
    - MIT Sloan Sports Analytics Conference papers
    - LINHAC (Linköping Hockey Analytics Conference)
    - ML-KULeuven/socceraction library
    - FiveThirtyEight RAPTOR methodology
    - ACM SIGKDD sports analytics papers
    - AAAI Workshop on AI in Team Sports
"""

__version__ = "1.0.0"
__author__ = "Hockey Analytics Research Team"

# Core xT models
from .expected_threat import (
    HockeyExpectedThreat,
    xTActionValuer,
    ActionType,
    StrengthState,
    create_sample_xT_model,
)

from .zone_entry_xt import (
    ZoneEntryxT,
    ZoneExitxT,
    ZoneEntry,
    EntryType,
    EntryLane,
    EntryOutcome,
)

# Shot analysis
from .shot_deception import (
    ShotDeceptionAnalyzer,
    ShooterProfile,
    ShotTrackingData,
    PuckFlutterAnalyzer,
    ShotType as DeceptionShotType,
)

# Network analysis
from .line_chemistry_gnn import (
    PassingNetwork,
    LineChemistryPredictor,
    PlayerNode,
    PassEvent,
    PassType,
)

# Workload and fatigue
from .fatigue_acwr import (
    ACWRModel,
    WorkloadCalculator,
    GameWorkload,
    PlayerWorkloadProfile,
    ThirdPeriodFatigueAnalyzer,
)

# Goalie analysis
from .goalie_save_model import (
    SaveDifficultyModel,
    GoalieAnalyzer,
    ReboundControlModel,
    GoalieProfile,
    ShotEvent,
    SaveType,
)

# Spatial analysis
from .ice_control import (
    IceControlModel,
    DefensiveSchemeClassifier,
    PlayerPosition,
)

from .obso import (
    OBSOModel,
    OBSOEvent,
    FrameSnapshot,
)

# Transition analysis
from .transition_analysis import (
    RushSuccessModel,
    TransitionSpeedAnalyzer,
    DefensiveRecoveryModel,
    RushEvent,
    RushType,
    RushOutcome,
)

# Data integration
from .nhl_api import (
    NHLAPIClient,
    PlayByPlayParser,
    PlayEvent,
    PlayerEdgeStats,
    GoalieEdgeStats,
    GameType,
)

# Advanced valuation
from .epv import (
    EPVModel,
    PossessionFrame,
    PossessionSequence,
    PossessionState,
)

from .shooter_gravity import (
    ShooterGravityModel,
    GoalieAttentionModel,
    ShooterGravityProfile,
    GravityEvent,
)

# Win probability
from .win_probability import (
    HockeyWinProbability,
    WinProbabilityTracker,
    GameContext,
    WinProbabilityResult,
    GameState as WPGameState,
    StrengthState as WPStrengthState,
)

# VAEP action valuation
from .vaep import (
    HockeyVAEP,
    VAEPAccumulator,
    HADLAction,
    VAEPValue,
    ActionType as VAEPActionType,
    ActionResult,
)

# Player embeddings
from .player_embeddings import (
    NHL2Vec,
    PlayerRecommendationSystem,
    Action2Vec,
    PlayerStats,
    PlayerEmbedding,
    Position,
    PlayerRole,
)

# Tactical detection
from .tactical_detection import (
    HockeyTacticalDetector,
    TacticalTrendAnalyzer,
    DefensiveSystem,
    OffensiveSystem,
    PowerPlayFormation,
    PenaltyKillFormation,
    GamePhase,
    FormationAnalysis,
    TacticalChangePoint,
)

# Trajectory prediction
from .trajectory_prediction import (
    HockeyTrajectoryPredictor,
    GhostingAnalyzer,
    GoalieTrajectoryModel,
    TrajectoryEvaluator,
    TrajectoryPrediction,
    TrackingFrame,
    GhostingResult,
)

# Play recognition
from .play_recognition import (
    NETSPlayRecognizer,
    PlayPatternMatcher,
    PlaySimilaritySearch,
    PlayEffectivenessAnalyzer,
    OffensivePlay,
    DefensivePlay,
    SpecialTeamsPlay,
    PlayRecognition,
    PlaySequence,
    GameFrame,
)

# RL Decision Optimizer
from .rl_decision import (
    HockeyDQN,
    LineDeploymentOptimizer,
    HierarchicalTacticalRL,
    DecisionEvaluator,
    Action as RLAction,
    StrengthState as RLStrengthState,
    GameZone,
    GameState as RLGameState,
    ActionResult as RLActionResult,
    Experience,
    QNetwork,
)

# Advanced Workload Footprint
from .workload_footprint import (
    HockeyWorkloadFootprint,
    InjuryRiskPredictor,
    RecoveryStatusTracker,
    DailyWorkload,
    WorkloadFootprint,
    InjuryRiskPrediction,
    Position as WorkloadPosition,
    InjuryType,
    RecoveryStatus,
)

# Forechecking Valuation
from .forechecking import (
    HockeyForecheckValuation,
    ForecheckTracker,
    BackcheckValuation,
    ForecheckEvent,
    PlayerForecheckValue,
    ForecheckAnalysis,
    ForecheckType,
    ForecheckOutcome,
    PressureSituation,
    PlayerPosition as ForecheckPlayerPosition,
)

# Contract/Trade Value Analyzer
from .contract_value import (
    HockeyContractValuation,
    TradeValueAnalyzer,
    MarketInefficiencyFinder,
    PlayerPerformance,
    ContractValuation,
    TradeValue,
    ContractStatus,
    Position as ContractPosition,
    PerformanceTier,
)

# Pass Probability Model
from .pass_model import (
    HockeyPassModel,
    DefensivePassAnalyzer,
    PlaymakingEvaluator,
    PassAttempt,
    PassProbability,
    DefensivePassCredit,
    PassType as PassModelType,
    PassOutcome,
    PassContext,
    PlayerPosition as PassPlayerPosition,
)


# Convenience functions
def create_full_analytics_suite(
    grid_x: int = 20,
    grid_y: int = 10,
) -> dict:
    """
    Create a complete suite of analytics models.

    Args:
        grid_x: Grid resolution along rink length
        grid_y: Grid resolution along rink width

    Returns:
        Dictionary with all initialized models
    """
    # Initialize NHL2Vec for player embeddings
    nhl2vec = NHL2Vec()

    return {
        # Core valuation models
        'xT': HockeyExpectedThreat(grid_length=grid_x, grid_width=grid_y),
        'zone_entry_xT': ZoneEntryxT(),
        'vaep': HockeyVAEP(),
        'epv': EPVModel(grid_x=grid_x * 2, grid_y=grid_y * 2),

        # Shot/scoring analysis
        'shot_deception': ShotDeceptionAnalyzer(),
        'gravity': ShooterGravityModel(),

        # Team/player analysis
        'line_chemistry': LineChemistryPredictor(),
        'nhl2vec': nhl2vec,
        'player_recommendation': PlayerRecommendationSystem(nhl2vec),

        # Workload and injury
        'acwr': ACWRModel(),

        # Goalie analysis
        'goalie': GoalieAnalyzer(),

        # Spatial analysis
        'ice_control': IceControlModel(),
        'obso': OBSOModel(),

        # Transition and rush
        'rush': RushSuccessModel(),
        'transition': TransitionSpeedAnalyzer(),

        # Win probability
        'win_probability': HockeyWinProbability(),
        'wp_tracker': WinProbabilityTracker(),

        # Tactical analysis
        'tactical': HockeyTacticalDetector(),
        'tactical_trends': TacticalTrendAnalyzer(),

        # Movement prediction
        'trajectory': HockeyTrajectoryPredictor(),
        'ghosting': GhostingAnalyzer(HockeyTrajectoryPredictor()),
        'goalie_trajectory': GoalieTrajectoryModel(),

        # Play recognition
        'play_recognizer': NETSPlayRecognizer(),
        'play_patterns': PlayPatternMatcher(),
        'play_search': PlaySimilaritySearch(),
        'play_effectiveness': PlayEffectivenessAnalyzer(),

        # RL Decision Optimizer
        'rl_decision': HockeyDQN(),
        'line_optimizer': LineDeploymentOptimizer(),
        'hierarchical_rl': HierarchicalTacticalRL(),
        'decision_evaluator': DecisionEvaluator(),

        # Advanced Workload Footprint
        'workload_footprint': HockeyWorkloadFootprint(),
        'injury_predictor': InjuryRiskPredictor(),
        'recovery_tracker': RecoveryStatusTracker(),

        # Forechecking Valuation
        'forecheck': HockeyForecheckValuation(),
        'forecheck_tracker': ForecheckTracker(),
        'backcheck': BackcheckValuation(),

        # Contract/Trade Value
        'contract_value': HockeyContractValuation(),
        'trade_analyzer': TradeValueAnalyzer(),
        'market_inefficiency': MarketInefficiencyFinder(),

        # Pass Probability Model
        'pass_model': HockeyPassModel(),
        'defensive_pass': DefensivePassAnalyzer(),
        'playmaking': PlaymakingEvaluator(),

        # Data integration
        'nhl_client': NHLAPIClient(),
    }


# Module listing for documentation
MODULES = [
    "expected_threat",
    "zone_entry_xt",
    "shot_deception",
    "line_chemistry_gnn",
    "fatigue_acwr",
    "goalie_save_model",
    "ice_control",
    "obso",
    "transition_analysis",
    "nhl_api",
    "epv",
    "shooter_gravity",
    "win_probability",
    "vaep",
    "player_embeddings",
    "tactical_detection",
    "trajectory_prediction",
    "play_recognition",
    "rl_decision",
    "workload_footprint",
    "forechecking",
    "contract_value",
    "pass_model",
]

# Research gap mapping
RESEARCH_GAPS = {
    "expected_threat": {
        "source_sports": ["soccer", "basketball"],
        "original_research": ["Karun Singh xT", "Sarah Rudd Markov chains"],
        "hockey_gap": "Continuous EPV model using NHL tracking data",
    },
    "zone_entry_xt": {
        "source_sports": ["soccer"],
        "original_research": ["Eric Tulsky zone entries"],
        "hockey_gap": "Lane-specific xT analysis for zone entries",
    },
    "shot_deception": {
        "source_sports": ["baseball"],
        "original_research": ["Baseball Prospectus tunneling", "Dr. Barton Smith"],
        "hockey_gap": "Shot tunneling metrics for shooter deception",
    },
    "line_chemistry_gnn": {
        "source_sports": ["soccer", "basketball"],
        "original_research": ["CMPN networks", "Flow motifs"],
        "hockey_gap": "GNN analysis of passing networks for line chemistry",
    },
    "fatigue_acwr": {
        "source_sports": ["soccer", "football"],
        "original_research": ["Gabbett ACWR", "ML injury prediction"],
        "hockey_gap": "Workload monitoring with hockey-specific stress factors",
    },
    "goalie_save_model": {
        "source_sports": ["baseball"],
        "original_research": ["Pitch framing research"],
        "hockey_gap": "Save difficulty beyond GSAx with positioning/screens",
    },
    "ice_control": {
        "source_sports": ["soccer"],
        "original_research": ["Fernández & Bornn pitch control"],
        "hockey_gap": "Continuous ice control surfaces",
    },
    "obso": {
        "source_sports": ["soccer"],
        "original_research": ["OBSO probabilistic models"],
        "hockey_gap": "Off-puck movement quality metrics",
    },
    "transition_analysis": {
        "source_sports": ["soccer"],
        "original_research": ["Counterattack GNN models"],
        "hockey_gap": "Odd-man rush success probability based on all positions",
    },
    "epv": {
        "source_sports": ["basketball", "soccer"],
        "original_research": ["Cervone EPV", "Fernández EPV decomposition"],
        "hockey_gap": "Frame-by-frame continuous possession valuation",
    },
    "shooter_gravity": {
        "source_sports": ["basketball"],
        "original_research": ["RAPTOR", "Steph Curry gravity studies"],
        "hockey_gap": "Systematic measurement of defensive attention",
    },
    "win_probability": {
        "source_sports": ["soccer", "football"],
        "original_research": ["Robberechts Bayesian WP", "NFL Random Forest WP"],
        "hockey_gap": "Real-time win probability with power play states",
    },
    "vaep": {
        "source_sports": ["soccer"],
        "original_research": ["Decroos VAEP", "ML-KULeuven SPADL"],
        "hockey_gap": "Action-level valuation with defensive credit",
    },
    "player_embeddings": {
        "source_sports": ["basketball", "soccer"],
        "original_research": ["NBA2Vec", "Football2Vec", "Play2Vec"],
        "hockey_gap": "Dense player representations for similarity/chemistry",
    },
    "tactical_detection": {
        "source_sports": ["soccer"],
        "original_research": ["SoccerCPD", "Formation identification surveys"],
        "hockey_gap": "Automated system/formation detection and change points",
    },
    "trajectory_prediction": {
        "source_sports": ["pedestrian", "basketball", "football"],
        "original_research": ["Social LSTM", "NFL Defender CNN-LSTM", "Ghosting"],
        "hockey_gap": "Player trajectory prediction and optimal positioning",
    },
    "play_recognition": {
        "source_sports": ["basketball", "soccer"],
        "original_research": ["NETS", "Play2Vec", "Group activity recognition"],
        "hockey_gap": "Automatic play pattern recognition from tracking",
    },
    "rl_decision": {
        "source_sports": ["soccer", "basketball"],
        "original_research": ["Q-Ball", "ReLiable", "Deep RL decision-making"],
        "hockey_gap": "RL-based line deployment and tactical decisions",
    },
    "workload_footprint": {
        "source_sports": ["soccer", "rugby"],
        "original_research": ["FWF methodology", "GPS load monitoring"],
        "hockey_gap": "Advanced workload footprint with shift-level tracking",
    },
    "forechecking": {
        "source_sports": ["soccer"],
        "original_research": ["exPress", "PPDA", "Pressing intensity"],
        "hockey_gap": "Systematic forechecking and backcheck valuation",
    },
    "contract_value": {
        "source_sports": ["baseball", "basketball"],
        "original_research": ["SHAP value explanations", "WAR-based contracts"],
        "hockey_gap": "Explainable contract valuations with market analysis",
    },
    "pass_model": {
        "source_sports": ["soccer"],
        "original_research": ["xPass", "GCN passing networks", "Pass success models"],
        "hockey_gap": "Pass completion probability with defensive credit",
    },
}
