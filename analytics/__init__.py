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

Research Origins:
    - Soccer: xT (Karun Singh), EPV (Fernández & Bornn), Pitch Control
    - Basketball: EPV (Cervone), RAPTOR/EPM, Gravity metrics
    - Baseball: Pitch tunneling, Seam-shifted wake (Dr. Barton Smith)
    - Football: ACWR injury prediction, GNN passing networks

References:
    - MIT Sloan Sports Analytics Conference papers
    - LINHAC (Linköping Hockey Analytics Conference)
    - ML-KULeuven/socceraction library
    - FiveThirtyEight RAPTOR methodology
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
    return {
        'xT': HockeyExpectedThreat(grid_length=grid_x, grid_width=grid_y),
        'zone_entry_xT': ZoneEntryxT(),
        'shot_deception': ShotDeceptionAnalyzer(),
        'line_chemistry': LineChemistryPredictor(),
        'acwr': ACWRModel(),
        'goalie': GoalieAnalyzer(),
        'ice_control': IceControlModel(),
        'obso': OBSOModel(),
        'rush': RushSuccessModel(),
        'transition': TransitionSpeedAnalyzer(),
        'epv': EPVModel(grid_x=grid_x * 2, grid_y=grid_y * 2),
        'gravity': ShooterGravityModel(),
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
}
