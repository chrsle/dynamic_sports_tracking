"""
Soccer Analytics Dashboard - Pure Python Implementation
Using Dash (Plotly) for visualization
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json

# ============================================================================
# CONSTANTS
# ============================================================================

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
BOX_LENGTH = 16.5
BOX_WIDTH = 40.32
SIX_YARD_LENGTH = 5.5
SIX_YARD_WIDTH = 18.32
GOAL_WIDTH = 7.32
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_SPOT = 11.0

# Sigma Computing Color Palette
COLORS = {
    # Team colors (Sigma categorical)
    'home': '#1976D2',
    'home_light': 'rgba(25, 118, 210, 0.15)',
    'away': '#EF5350',
    'away_light': 'rgba(239, 83, 80, 0.15)',

    # Pitch (keeping green for visibility)
    'pitch': '#2d5a3c',
    'lines': 'rgba(255, 255, 255, 0.9)',
    'ball': '#ffffff',

    # Status colors (Sigma)
    'goal': '#2A8D5C',
    'saved': '#F6BD16',
    'blocked': '#9fa8a7',
    'off_target': '#757575',

    # Surface colors (Sigma light theme)
    'bg': '#f5f5f5',
    'card': '#ffffff',
    'border': '#e0e0e0',
    'border_light': '#f0f0f0',

    # Text (Sigma)
    'text': '#292929',
    'text_secondary': '#575757',
    'text_tertiary': '#9fa8a7',

    # Additional Sigma categorical
    'cyan': '#52CBFF',
    'purple': '#6E4BD2',
    'orange': '#FB9649',
    'lime': '#87DC44',
    'teal': '#68DFC5',
    'yellow': '#F6BD16',
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    x: float
    y: float

    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Player:
    player_id: int
    team: str
    x: float
    y: float
    jersey_number: int
    velocity: float = 0.0
    distance_covered: float = 0.0
    sprints: int = 0


@dataclass
class Shot:
    shot_id: int
    team: str
    x: float
    y: float
    result: str
    xg: float
    time: float
    distance: float


# ============================================================================
# GAME STATE
# ============================================================================

class GameState:
    """Manages the entire game state"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.frame = 0
        self.match_time = 0.0
        self.is_playing = False
        self.playback_speed = 1.0

        # Score and xG
        self.score = {'home': 0, 'away': 0}
        self.xg = {'home': 0.0, 'away': 0.0}
        self.possession = {'home': 50.0, 'away': 50.0}
        self.momentum = 0.5

        # Ball
        self.ball = Position(PITCH_LENGTH / 2, PITCH_WIDTH / 2)

        # Players
        self.home_players: List[Player] = []
        self.away_players: List[Player] = []
        self._init_players()

        # Shots and timeline
        self.shots: List[Shot] = []
        self.xg_timeline: List[dict] = []

        # Man advantage
        self.man_advantage = {'is_active': False, 'home_players': 11, 'away_players': 11}

        # Goalie metrics
        self.home_goalie = {'coverage': 65, 'positioning': 0.75, 'saves': 0, 'distance': 3}
        self.away_goalie = {'coverage': 65, 'positioning': 0.75, 'saves': 0, 'distance': 3}

        # Fatigue
        self.fatigue: Dict[int, float] = {}

        # Pressure
        self.pressure = {
            'home': {'ppda': 10.0, 'press_success': 35.0, 'counter_press': 0.7},
            'away': {'ppda': 12.0, 'press_success': 30.0, 'counter_press': 0.6}
        }

        # Patterns
        self.patterns = {
            'home': {'build_up': 'Medium', 'width': 45, 'key_passes': 0},
            'away': {'build_up': 'Fast', 'width': 40, 'key_passes': 0}
        }

        # Insights
        self.insights: List[str] = ['Match started']

    def _init_players(self):
        """Initialize player formations"""
        # Home team (4-3-3)
        home_formation = [
            (5, 34, 1), (25, 10, 2), (25, 25, 4), (25, 43, 5), (25, 58, 3),
            (45, 20, 6), (45, 34, 8), (45, 48, 10),
            (70, 15, 7), (75, 34, 9), (70, 53, 11)
        ]
        self.home_players = [
            Player(i + 1, 'home', x, y, num)
            for i, (x, y, num) in enumerate(home_formation)
        ]

        # Away team (4-4-2)
        away_formation = [
            (100, 34, 1), (80, 10, 2), (80, 25, 4), (80, 43, 5), (80, 58, 3),
            (60, 8, 7), (60, 27, 8), (60, 41, 6), (60, 60, 11),
            (35, 25, 9), (35, 43, 10)
        ]
        self.away_players = [
            Player(i + 12, 'away', x, y, num)
            for i, (x, y, num) in enumerate(away_formation)
        ]

        # Initialize fatigue
        for p in self.home_players + self.away_players:
            self.fatigue[p.player_id] = 0.0

    def update(self):
        """Update game state for one frame"""
        if not self.is_playing:
            return

        self.frame += 1
        self.match_time += 1 / 30 * self.playback_speed

        self._update_ball()
        self._update_players()
        self._maybe_generate_shot()
        self._update_possession()
        self._update_momentum()
        self._update_fatigue()
        self._update_pressure()
        self._update_patterns()
        self._update_goalies()

    def _update_ball(self):
        target_x = 80 if self.momentum > 0.5 else 25
        target_y = PITCH_WIDTH / 2 + random.uniform(-20, 20)

        self.ball.x += (target_x - self.ball.x) * 0.02 + random.uniform(-2, 2)
        self.ball.y += (target_y - self.ball.y) * 0.02 + random.uniform(-2, 2)

        self.ball.x = max(0, min(PITCH_LENGTH, self.ball.x))
        self.ball.y = max(0, min(PITCH_WIDTH, self.ball.y))

    def _update_players(self):
        for player in self.home_players + self.away_players:
            dx = self.ball.x - player.x
            dy = self.ball.y - player.y
            dist = math.sqrt(dx**2 + dy**2)

            move_x = random.uniform(-0.5, 0.5)
            move_y = random.uniform(-0.5, 0.5)

            if dist < 20:
                move_x += dx * 0.03
                move_y += dy * 0.03

            player.x += move_x
            player.y += move_y

            # Keep in bounds
            if player.team == 'home':
                player.x = max(0, min(95, player.x))
            else:
                player.x = max(10, min(105, player.x))
            player.y = max(2, min(66, player.y))

            # Update velocity and distance
            player.velocity = math.sqrt(move_x**2 + move_y**2) * 30
            player.distance_covered += player.velocity / 30

            if player.velocity > 7:
                player.sprints += 1

    def _maybe_generate_shot(self):
        if random.random() > 0.002:
            return

        team = 'home' if self.ball.x > 70 else 'away' if self.ball.x < 35 else None
        if not team:
            return

        shot_x = self.ball.x + random.uniform(-5, 5)
        shot_y = self.ball.y + random.uniform(-5, 5)

        if team == 'home':
            shot_x = max(70, min(100, shot_x))
            goal_x = PITCH_LENGTH
        else:
            shot_x = max(5, min(35, shot_x))
            goal_x = 0

        goal_y = PITCH_WIDTH / 2
        distance = math.sqrt((shot_x - goal_x)**2 + (shot_y - goal_y)**2)

        # Calculate xG
        if distance < 6:
            xg = 0.6
        elif distance < 12:
            xg = 0.25
        elif distance < 20:
            xg = 0.1
        else:
            xg = 0.03

        angle = abs(math.atan2(shot_y - goal_y, abs(shot_x - goal_x)))
        xg *= max(0.3, 1 - angle / math.pi)

        # Determine result
        rand = random.random()
        if rand < xg * 0.8:
            result = 'goal'
            self.score[team] += 1
            self.insights.insert(0, f"⚽ GOAL! {team.upper()} scores! (xG: {xg:.2f})")
        elif rand < xg + 0.2:
            result = 'saved'
            if team == 'home':
                self.away_goalie['saves'] += 1
            else:
                self.home_goalie['saves'] += 1
        elif rand < xg + 0.4:
            result = 'blocked'
        elif rand < xg + 0.5:
            result = 'post'
        else:
            result = 'off_target'

        self.xg[team] += xg

        shot = Shot(
            shot_id=len(self.shots) + 1,
            team=team,
            x=shot_x,
            y=shot_y,
            result=result,
            xg=round(xg, 3),
            time=self.match_time,
            distance=round(distance, 1)
        )
        self.shots.append(shot)

        self.xg_timeline.append({
            'time': self.match_time,
            'home_xg': self.xg['home'],
            'away_xg': self.xg['away']
        })

        if result != 'goal':
            self.insights.insert(0, f"Shot by {team.upper()} - {result} (xG: {xg:.2f})")

        # Keep insights manageable
        self.insights = self.insights[:15]

    def _update_possession(self):
        target = 60 if self.ball.x > 52.5 else 40
        self.possession['home'] += (target - self.possession['home']) * 0.01
        self.possession['away'] = 100 - self.possession['home']

    def _update_momentum(self):
        ball_factor = (self.ball.x - 52.5) / 52.5 * 0.3
        self.momentum += ball_factor * 0.01 + random.uniform(-0.02, 0.02)
        self.momentum = max(0, min(1, self.momentum))

    def _update_fatigue(self):
        minutes = self.match_time / 60
        for player in self.home_players + self.away_players:
            distance = player.distance_covered
            fatigue_level = min(1.0, (distance / 1000) * (minutes / 45) * 0.5) if minutes > 0 else 0
            self.fatigue[player.player_id] = fatigue_level

    def _update_pressure(self):
        self.pressure['home']['ppda'] = 8 + random.uniform(-2, 2)
        self.pressure['home']['press_success'] = 35 + random.uniform(-10, 10)
        self.pressure['home']['counter_press'] = 0.7 + random.uniform(-0.2, 0.2)

        self.pressure['away']['ppda'] = 10 + random.uniform(-2, 2)
        self.pressure['away']['press_success'] = 30 + random.uniform(-10, 10)
        self.pressure['away']['counter_press'] = 0.6 + random.uniform(-0.2, 0.2)

    def _update_patterns(self):
        options = ['Slow', 'Medium', 'Fast', 'Direct']
        self.patterns['home']['build_up'] = random.choice(options[:3])
        self.patterns['home']['width'] = 45 + random.uniform(-10, 10)
        self.patterns['home']['key_passes'] = int(self.match_time / 60 * 2)

        self.patterns['away']['build_up'] = random.choice(options)
        self.patterns['away']['width'] = 40 + random.uniform(-10, 10)
        self.patterns['away']['key_passes'] = int(self.match_time / 60 * 1.5)

    def _update_goalies(self):
        home_gk = self.home_players[0]
        away_gk = self.away_players[0]

        self.home_goalie['coverage'] = 60 + home_gk.x * 2
        self.home_goalie['positioning'] = 0.7 + random.uniform(-0.1, 0.2)
        self.home_goalie['distance'] = home_gk.x

        self.away_goalie['coverage'] = 60 + (105 - away_gk.x) * 2
        self.away_goalie['positioning'] = 0.7 + random.uniform(-0.1, 0.2)
        self.away_goalie['distance'] = 105 - away_gk.x

    def format_time(self) -> str:
        mins = int(self.match_time // 60)
        secs = int(self.match_time % 60)
        return f"{mins:02d}:{secs:02d}"


# Global game state
game_state = GameState()


# ============================================================================
# PITCH FIGURE
# ============================================================================

def create_pitch_figure(view_mode='positions'):
    """Create the pitch visualization"""
    fig = go.Figure()

    # Pitch background
    fig.add_shape(type='rect', x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH,
                  fillcolor=COLORS['pitch'], line=dict(color=COLORS['lines'], width=2))

    # Center line
    fig.add_shape(type='line', x0=PITCH_LENGTH/2, y0=0, x1=PITCH_LENGTH/2, y1=PITCH_WIDTH,
                  line=dict(color=COLORS['lines'], width=2))

    # Center circle
    theta = np.linspace(0, 2*np.pi, 50)
    cx = PITCH_LENGTH/2 + CENTER_CIRCLE_RADIUS * np.cos(theta)
    cy = PITCH_WIDTH/2 + CENTER_CIRCLE_RADIUS * np.sin(theta)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode='lines',
                            line=dict(color=COLORS['lines'], width=2), showlegend=False))

    # Left penalty area
    box_y_start = (PITCH_WIDTH - BOX_WIDTH) / 2
    fig.add_shape(type='rect', x0=0, y0=box_y_start, x1=BOX_LENGTH, y1=box_y_start + BOX_WIDTH,
                  line=dict(color=COLORS['lines'], width=2))

    # Left six-yard box
    six_y_start = (PITCH_WIDTH - SIX_YARD_WIDTH) / 2
    fig.add_shape(type='rect', x0=0, y0=six_y_start, x1=SIX_YARD_LENGTH, y1=six_y_start + SIX_YARD_WIDTH,
                  line=dict(color=COLORS['lines'], width=2))

    # Right penalty area
    fig.add_shape(type='rect', x0=PITCH_LENGTH - BOX_LENGTH, y0=box_y_start,
                  x1=PITCH_LENGTH, y1=box_y_start + BOX_WIDTH,
                  line=dict(color=COLORS['lines'], width=2))

    # Right six-yard box
    fig.add_shape(type='rect', x0=PITCH_LENGTH - SIX_YARD_LENGTH, y0=six_y_start,
                  x1=PITCH_LENGTH, y1=six_y_start + SIX_YARD_WIDTH,
                  line=dict(color=COLORS['lines'], width=2))

    # Penalty spots
    fig.add_trace(go.Scatter(x=[PENALTY_SPOT, PITCH_LENGTH - PENALTY_SPOT],
                            y=[PITCH_WIDTH/2, PITCH_WIDTH/2],
                            mode='markers', marker=dict(color=COLORS['lines'], size=5),
                            showlegend=False))

    # Goals
    goal_y_start = (PITCH_WIDTH - GOAL_WIDTH) / 2
    fig.add_shape(type='rect', x0=-2, y0=goal_y_start, x1=0, y1=goal_y_start + GOAL_WIDTH,
                  fillcolor='#ffd700', line=dict(color='#ffd700', width=2))
    fig.add_shape(type='rect', x0=PITCH_LENGTH, y0=goal_y_start,
                  x1=PITCH_LENGTH + 2, y1=goal_y_start + GOAL_WIDTH,
                  fillcolor='#ffd700', line=dict(color='#ffd700', width=2))

    if view_mode == 'positions':
        _add_players_to_figure(fig)
    elif view_mode == 'shotmap':
        _add_shotmap_to_figure(fig)
    elif view_mode == 'heatmap':
        _add_heatmap_to_figure(fig)
        _add_players_to_figure(fig)

    # Layout - Sigma Computing style
    fig.update_layout(
        xaxis=dict(range=[-5, PITCH_LENGTH + 5], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-5, PITCH_WIDTH + 5], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor='x', scaleratio=1, fixedrange=True),
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        showlegend=False
    )

    return fig


def _add_players_to_figure(fig):
    """Add player markers to the pitch"""
    # Home players
    home_x = [p.x for p in game_state.home_players]
    home_y = [p.y for p in game_state.home_players]
    home_text = [str(p.jersey_number) for p in game_state.home_players]

    fig.add_trace(go.Scatter(
        x=home_x, y=home_y, mode='markers+text',
        marker=dict(color=COLORS['home'], size=20, line=dict(color='white', width=2)),
        text=home_text, textposition='middle center',
        textfont=dict(color='white', size=10, family='Arial Black'),
        name='Home', showlegend=False
    ))

    # Away players
    away_x = [p.x for p in game_state.away_players]
    away_y = [p.y for p in game_state.away_players]
    away_text = [str(p.jersey_number) for p in game_state.away_players]

    fig.add_trace(go.Scatter(
        x=away_x, y=away_y, mode='markers+text',
        marker=dict(color=COLORS['away'], size=20, line=dict(color='white', width=2)),
        text=away_text, textposition='middle center',
        textfont=dict(color='white', size=10, family='Arial Black'),
        name='Away', showlegend=False
    ))

    # Ball
    fig.add_trace(go.Scatter(
        x=[game_state.ball.x], y=[game_state.ball.y], mode='markers',
        marker=dict(color=COLORS['ball'], size=12, line=dict(color='black', width=2)),
        name='Ball', showlegend=False
    ))


def _add_shotmap_to_figure(fig):
    """Add shot markers to the pitch"""
    for shot in game_state.shots:
        color = {
            'goal': COLORS['goal'],
            'saved': COLORS['saved'],
            'blocked': COLORS['blocked'],
            'off_target': COLORS['off_target'],
            'post': COLORS['off_target']
        }.get(shot.result, COLORS['off_target'])

        size = 10 + shot.xg * 30

        fig.add_trace(go.Scatter(
            x=[shot.x], y=[shot.y], mode='markers',
            marker=dict(
                color=color, size=size, opacity=0.8,
                line=dict(color=COLORS['home'] if shot.team == 'home' else COLORS['away'], width=2)
            ),
            hovertemplate=f"xG: {shot.xg:.3f}<br>Result: {shot.result}<br>Distance: {shot.distance}m",
            showlegend=False
        ))


def _add_heatmap_to_figure(fig):
    """Add heatmap overlay"""
    # Create a grid for heatmap
    grid_size = 10
    heatmap = np.zeros((int(PITCH_LENGTH / grid_size) + 1, int(PITCH_WIDTH / grid_size) + 1))

    for player in game_state.home_players + game_state.away_players:
        gx = int(player.x / grid_size)
        gy = int(player.y / grid_size)
        if 0 <= gx < heatmap.shape[0] and 0 <= gy < heatmap.shape[1]:
            heatmap[gx, gy] += 1

    # Add heatmap as contour
    x_vals = np.arange(0, PITCH_LENGTH + grid_size, grid_size)
    y_vals = np.arange(0, PITCH_WIDTH + grid_size, grid_size)

    fig.add_trace(go.Heatmap(
        z=heatmap.T,
        x=x_vals,
        y=y_vals,
        colorscale='YlOrRd',
        opacity=0.4,
        showscale=False
    ))


# ============================================================================
# CHART FIGURES
# ============================================================================

def create_xg_timeline_figure():
    """Create xG timeline chart - Sigma Computing style"""
    fig = go.Figure()

    if game_state.xg_timeline:
        times = [p['time'] for p in game_state.xg_timeline]
        home_xg = [p['home_xg'] for p in game_state.xg_timeline]
        away_xg = [p['away_xg'] for p in game_state.xg_timeline]

        fig.add_trace(go.Scatter(
            x=times, y=home_xg, mode='lines', name='Home xG',
            line=dict(color=COLORS['home'], width=2),
            fill='tozeroy', fillcolor='rgba(25, 118, 210, 0.1)'
        ))

        fig.add_trace(go.Scatter(
            x=times, y=away_xg, mode='lines', name='Away xG',
            line=dict(color=COLORS['away'], width=2),
            fill='tozeroy', fillcolor='rgba(239, 83, 80, 0.1)'
        ))

    fig.update_layout(
        xaxis=dict(
            title='Match Time (s)',
            color=COLORS['text_secondary'],
            gridcolor=COLORS['border_light'],
            linecolor=COLORS['border'],
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title='Cumulative xG',
            color=COLORS['text_secondary'],
            gridcolor=COLORS['border_light'],
            linecolor=COLORS['border'],
            tickfont=dict(size=10)
        ),
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family='Raleway, Inter, sans-serif', size=11),
        margin=dict(l=40, r=20, t=20, b=40),
        height=180,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10)
        )
    )

    return fig


def create_possession_figure():
    """Create possession bar chart - Sigma Computing style"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=['Possession'],
        x=[game_state.possession['home']],
        orientation='h',
        marker_color=COLORS['home'],
        name='Home',
        text=f"{game_state.possession['home']:.0f}%",
        textposition='inside',
        textfont=dict(color='white', size=11, family='Inter, sans-serif')
    ))

    fig.add_trace(go.Bar(
        y=['Possession'],
        x=[game_state.possession['away']],
        orientation='h',
        marker_color=COLORS['away'],
        name='Away',
        text=f"{game_state.possession['away']:.0f}%",
        textposition='inside',
        textfont=dict(color='white', size=11, family='Inter, sans-serif')
    ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(showticklabels=False, showgrid=False, range=[0, 100], zeroline=False),
        yaxis=dict(showticklabels=False),
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        margin=dict(l=10, r=10, t=10, b=10),
        height=40,
        showlegend=False
    )

    return fig


def create_fatigue_figure():
    """Create fatigue heatmap for all players - Sigma Computing style"""
    players = game_state.home_players + game_state.away_players
    fatigue_values = [game_state.fatigue.get(p.player_id, 0) for p in players]
    labels = [f"{'H' if p.team == 'home' else 'A'}{p.jersey_number}" for p in players]

    # Sigma categorical colors for fatigue levels
    colors = []
    for f in fatigue_values:
        if f > 0.7:
            colors.append(COLORS['away'])  # Red for high fatigue
        elif f > 0.4:
            colors.append(COLORS['yellow'])  # Yellow for medium
        else:
            colors.append(COLORS['goal'])  # Green for low

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=fatigue_values,
        marker_color=colors,
        text=[f"{f*100:.0f}%" for f in fatigue_values],
        textposition='outside',
        textfont=dict(size=9)
    ))

    fig.update_layout(
        xaxis=dict(
            color=COLORS['text_secondary'],
            tickangle=45,
            tickfont=dict(size=9),
            gridcolor=COLORS['border_light']
        ),
        yaxis=dict(
            range=[0, 1.1],
            title='Fatigue',
            color=COLORS['text_secondary'],
            tickfont=dict(size=9),
            gridcolor=COLORS['border_light'],
            titlefont=dict(size=10)
        ),
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], size=10, family='Raleway, Inter, sans-serif'),
        margin=dict(l=40, r=10, t=10, b=50),
        height=180
    )

    return fig


def create_goalie_figure(team='home'):
    """Create goalkeeper positioning visualization - Sigma Computing style"""
    goalie = game_state.home_goalie if team == 'home' else game_state.away_goalie

    fig = go.Figure()

    # Goal outline (Sigma minimal style)
    fig.add_shape(type='rect', x0=0, y0=0, x1=100, y1=50,
                  line=dict(color=COLORS['text'], width=2))

    # Coverage area
    coverage_width = goalie['coverage']
    coverage_color = COLORS['home_light'] if team == 'home' else COLORS['away_light']
    fig.add_shape(type='rect',
                  x0=(100 - coverage_width) / 2, y0=0,
                  x1=(100 + coverage_width) / 2, y1=50,
                  fillcolor=coverage_color,
                  line=dict(width=0))

    # Goalkeeper position
    gk_x = 50
    gk_y = 5 + goalie['distance'] * 3

    fig.add_trace(go.Scatter(
        x=[gk_x], y=[gk_y], mode='markers',
        marker=dict(
            color=COLORS['home'] if team == 'home' else COLORS['away'],
            size=18,
            line=dict(color=COLORS['card'], width=2)
        ),
        showlegend=False
    ))

    # Quality indicator (Sigma colors)
    if goalie['positioning'] > 0.7:
        quality_color = COLORS['goal']
    elif goalie['positioning'] > 0.4:
        quality_color = COLORS['yellow']
    else:
        quality_color = COLORS['away']

    fig.add_annotation(
        x=50, y=55, text=f"Quality: {goalie['positioning']*100:.0f}%",
        showarrow=False, font=dict(color=quality_color, size=11, family='Inter, sans-serif')
    )

    fig.update_layout(
        xaxis=dict(range=[-10, 110], showgrid=False, showticklabels=False, fixedrange=True, zeroline=False),
        yaxis=dict(range=[-5, 60], showgrid=False, showticklabels=False, fixedrange=True, zeroline=False),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['card'],
        margin=dict(l=10, r=10, t=10, b=10),
        height=110
    )

    return fig


# ============================================================================
# DASH APP
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Sigma Computing Design System CSS
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Soccer Analytics Dashboard</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Sigma Computing Design System */
        :root {
            --sigma-charcoal: #292929;
            --sigma-gray-600: #575757;
            --sigma-gray-400: #9fa8a7;
            --sigma-gray-200: #e0e0e0;
            --sigma-gray-100: #f0f0f0;
            --sigma-white: #ffffff;
            --sigma-blue: #1976D2;
            --sigma-red: #EF5350;
            --sigma-green: #2A8D5C;
            --sigma-yellow: #F6BD16;
        }

        body {
            background-color: #f5f5f5 !important;
            font-family: 'Raleway', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: #292929 !important;
            font-size: 14px;
        }

        .card {
            background-color: #ffffff !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 4px !important;
            margin-bottom: 12px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        }

        .card-header {
            background-color: #ffffff !important;
            border-bottom: 1px solid #f0f0f0 !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            font-size: 0.8125rem !important;
            color: #292929 !important;
        }

        .card-body {
            padding: 12px 16px !important;
        }

        .stat-label {
            color: #9fa8a7 !important;
            font-size: 0.6875rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .stat-value {
            color: #292929 !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
        }

        .home-color { color: #1976D2 !important; }
        .away-color { color: #EF5350 !important; }

        .score-display {
            font-size: 2rem !important;
            font-weight: 700 !important;
            font-family: 'SF Mono', Monaco, monospace !important;
        }

        .xg-display {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            font-family: 'SF Mono', Monaco, monospace !important;
        }

        .time-display {
            font-size: 1.125rem !important;
            font-family: 'SF Mono', Monaco, monospace !important;
            color: #2A8D5C !important;
            background: #f0f0f0 !important;
            padding: 4px 12px !important;
            border-radius: 2px !important;
            border: 1px solid #e0e0e0 !important;
        }

        .insight-item {
            padding: 6px 10px;
            margin: 3px 0;
            background: #f5f5f5;
            border-radius: 2px;
            font-size: 0.75rem;
            border-left: 3px solid #e0e0e0;
            color: #575757;
        }

        .insight-goal { border-left-color: #2A8D5C !important; }
        .insight-shot { border-left-color: #F6BD16 !important; }

        .btn-control {
            margin: 2px;
            border-radius: 2px !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
        }

        .btn-primary {
            background-color: #1976D2 !important;
            border-color: #1976D2 !important;
        }

        .btn-secondary {
            background-color: #f0f0f0 !important;
            border-color: #e0e0e0 !important;
            color: #575757 !important;
        }

        .btn-success {
            background-color: #2A8D5C !important;
            border-color: #2A8D5C !important;
        }

        .btn-warning {
            background-color: #F6BD16 !important;
            border-color: #F6BD16 !important;
            color: #292929 !important;
        }

        h2, h6, .h2, .h6 {
            color: #292929 !important;
        }

        .text-white {
            color: #292929 !important;
        }

        .text-secondary {
            color: #9fa8a7 !important;
        }

        .form-select, .dropdown-toggle {
            font-size: 0.75rem !important;
            border-radius: 2px !important;
        }

        /* Badge styling */
        .badge {
            font-size: 0.6875rem !important;
            font-weight: 600 !important;
            border-radius: 2px !important;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #f0f0f0; }
        ::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #9fa8a7; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

app.layout = dbc.Container([
    # Interval for updates
    dcc.Interval(id='interval-component', interval=33, n_intervals=0),
    dcc.Store(id='view-mode', data='positions'),

    # Header - Sigma Computing style
    dbc.Row([
        dbc.Col([
            html.H2("Soccer Analytics", className="mb-0", style={'fontSize': '1rem', 'fontWeight': '600', 'color': '#292929'})
        ], width=4),
        dbc.Col([
            html.Div([
                html.Span(id='home-team-name', children="HOME", className="home-color me-3",
                         style={'fontSize': '0.875rem', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.02em'}),
                html.Span(id='home-score', children="0", className="score-display home-color me-2"),
                html.Span(" - ", className="score-display", style={'color': '#292929'}),
                html.Span(id='away-score', children="0", className="score-display away-color ms-2"),
                html.Span(id='away-team-name', children="AWAY", className="away-color ms-3",
                         style={'fontSize': '0.875rem', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.02em'}),
            ], className="text-center")
        ], width=4),
        dbc.Col([
            html.Div([
                html.Span(id='match-time', children="00:00", className="time-display me-3"),
                html.Span(id='frame-counter', children="Frame: 0", style={'color': '#9fa8a7', 'fontSize': '0.75rem'})
            ], className="text-end")
        ], width=4),
    ], className="py-2 mb-3", style={'backgroundColor': '#ffffff', 'borderRadius': '4px', 'border': '1px solid #e0e0e0', 'boxShadow': '0 1px 2px rgba(0,0,0,0.04)'}),

    # Playback Controls
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("⏮ -10s", id='btn-skip-back', color="secondary", size="sm", className="btn-control"),
                dbc.Button("◀ Prev", id='btn-prev', color="secondary", size="sm", className="btn-control"),
                dbc.Button("▶ Play", id='btn-play', color="success", size="sm", className="btn-control"),
                dbc.Button("Next ▶", id='btn-next', color="secondary", size="sm", className="btn-control"),
                dbc.Button("+10s ⏭", id='btn-skip-forward', color="secondary", size="sm", className="btn-control"),
            ]),
            html.Span(" Speed: ", className="text-secondary ms-3"),
            dcc.Dropdown(
                id='speed-select',
                options=[
                    {'label': '0.5x', 'value': 0.5},
                    {'label': '1x', 'value': 1.0},
                    {'label': '2x', 'value': 2.0},
                    {'label': '4x', 'value': 4.0},
                ],
                value=1.0,
                clearable=False,
                style={'width': '80px', 'display': 'inline-block', 'verticalAlign': 'middle'}
            ),
        ], width=12, className="mb-3")
    ]),

    # Main Content
    dbc.Row([
        # Left Column - Pitch
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("Pitch View"),
                    dbc.ButtonGroup([
                        dbc.Button("Positions", id='btn-positions', color="primary", size="sm", className="ms-3"),
                        dbc.Button("Shot Map", id='btn-shotmap', color="secondary", size="sm"),
                        dbc.Button("Heatmap", id='btn-heatmap', color="secondary", size="sm"),
                    ])
                ]),
                dbc.CardBody([
                    dcc.Graph(id='pitch-graph', config={'displayModeBar': False})
                ])
            ])
        ], width=7),

        # Right Column - Stats
        dbc.Col([
            # xG Display
            dbc.Card([
                dbc.CardHeader("Expected Goals (xG)"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='home-xg', children="0.00", className="xg-display home-color text-center"),
                            html.Div("Home xG", className="stat-label text-center")
                        ], width=6),
                        dbc.Col([
                            html.Div(id='away-xg', children="0.00", className="xg-display away-color text-center"),
                            html.Div("Away xG", className="stat-label text-center")
                        ], width=6),
                    ])
                ])
            ]),

            # xG Timeline
            dbc.Card([
                dbc.CardHeader("xG Timeline"),
                dbc.CardBody([
                    dcc.Graph(id='xg-timeline', config={'displayModeBar': False})
                ])
            ]),

            # Possession
            dbc.Card([
                dbc.CardHeader("Possession"),
                dbc.CardBody([
                    dcc.Graph(id='possession-graph', config={'displayModeBar': False})
                ])
            ]),
        ], width=5),
    ]),

    # Second Row - Analytics
    dbc.Row([
        # Player Analytics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "Player Analytics ",
                    dcc.Dropdown(
                        id='player-select',
                        options=[],
                        placeholder="Select Player",
                        style={'width': '150px', 'display': 'inline-block'}
                    )
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div("Distance", className="stat-label"),
                            html.Div(id='player-distance', children="0 m", className="stat-value")
                        ], width=3),
                        dbc.Col([
                            html.Div("Velocity", className="stat-label"),
                            html.Div(id='player-velocity', children="0 m/s", className="stat-value")
                        ], width=3),
                        dbc.Col([
                            html.Div("Sprints", className="stat-label"),
                            html.Div(id='player-sprints', children="0", className="stat-value")
                        ], width=3),
                        dbc.Col([
                            html.Div("Fatigue", className="stat-label"),
                            html.Div(id='player-fatigue', children="0%", className="stat-value")
                        ], width=3),
                    ])
                ])
            ])
        ], width=6),

        # Goalkeeper Analytics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "Goalkeeper Analytics ",
                    dcc.Dropdown(
                        id='goalie-select',
                        options=[
                            {'label': 'Home GK', 'value': 'home'},
                            {'label': 'Away GK', 'value': 'away'}
                        ],
                        value='home',
                        clearable=False,
                        style={'width': '120px', 'display': 'inline-block'}
                    )
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='goalie-graph', config={'displayModeBar': False})
                        ], width=6),
                        dbc.Col([
                            html.Div("Coverage", className="stat-label"),
                            html.Div(id='goalie-coverage', children="65%", className="stat-value"),
                            html.Div("Saves", className="stat-label mt-2"),
                            html.Div(id='goalie-saves', children="0", className="stat-value"),
                        ], width=6),
                    ])
                ])
            ])
        ], width=6),
    ], className="mt-3"),

    # Third Row - Advanced Analytics
    dbc.Row([
        # Pressure Analytics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Pressure Analytics"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Home", className="home-color"),
                            html.Div("PPDA", className="stat-label"),
                            html.Div(id='home-ppda', children="10.0", className="stat-value"),
                            html.Div("Press Success", className="stat-label mt-1"),
                            html.Div(id='home-press-success', children="35%", className="stat-value"),
                        ], width=6),
                        dbc.Col([
                            html.H6("Away", className="away-color"),
                            html.Div("PPDA", className="stat-label"),
                            html.Div(id='away-ppda', children="12.0", className="stat-value"),
                            html.Div("Press Success", className="stat-label mt-1"),
                            html.Div(id='away-press-success', children="30%", className="stat-value"),
                        ], width=6),
                    ])
                ])
            ])
        ], width=3),

        # Pattern Analytics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Play Patterns"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Home", className="home-color"),
                            html.Div("Build-up", className="stat-label"),
                            html.Div(id='home-buildup', children="Medium", className="stat-value"),
                            html.Div("Key Passes", className="stat-label mt-1"),
                            html.Div(id='home-keypasses', children="0", className="stat-value"),
                        ], width=6),
                        dbc.Col([
                            html.H6("Away", className="away-color"),
                            html.Div("Build-up", className="stat-label"),
                            html.Div(id='away-buildup', children="Fast", className="stat-value"),
                            html.Div("Key Passes", className="stat-label mt-1"),
                            html.Div(id='away-keypasses', children="0", className="stat-value"),
                        ], width=6),
                    ])
                ])
            ])
        ], width=3),

        # Fatigue Monitor
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Fatigue Monitor"),
                dbc.CardBody([
                    dcc.Graph(id='fatigue-graph', config={'displayModeBar': False})
                ])
            ])
        ], width=3),

        # Live Insights
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Live Insights"),
                dbc.CardBody([
                    html.Div(id='insights-feed', style={'maxHeight': '200px', 'overflowY': 'auto'})
                ])
            ])
        ], width=3),
    ], className="mt-3"),

    # Shots List
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "Recent Shots ",
                    dbc.Badge(id='shot-count', children="0", color="secondary", className="ms-2")
                ]),
                dbc.CardBody([
                    html.Div(id='shots-list', style={'maxHeight': '150px', 'overflowY': 'auto'})
                ])
            ])
        ], width=12),
    ], className="mt-3"),

], fluid=True, style={'backgroundColor': '#f5f5f5', 'minHeight': '100vh', 'padding': '16px'})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('btn-play', 'children'),
     Output('btn-play', 'color')],
    [Input('btn-play', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_play(n_clicks):
    game_state.is_playing = not game_state.is_playing
    if game_state.is_playing:
        return "⏸ Pause", "warning"
    else:
        return "▶ Play", "success"


@app.callback(
    Output('speed-select', 'value'),
    [Input('speed-select', 'value')]
)
def update_speed(speed):
    game_state.playback_speed = speed
    return speed


@app.callback(
    Output('view-mode', 'data'),
    [Input('btn-positions', 'n_clicks'),
     Input('btn-shotmap', 'n_clicks'),
     Input('btn-heatmap', 'n_clicks')],
    [State('view-mode', 'data')],
    prevent_initial_call=True
)
def update_view_mode(pos_clicks, shot_clicks, heat_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return 'positions'

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-positions':
        return 'positions'
    elif button_id == 'btn-shotmap':
        return 'shotmap'
    elif button_id == 'btn-heatmap':
        return 'heatmap'

    return 'positions'


@app.callback(
    [Output('btn-positions', 'color'),
     Output('btn-shotmap', 'color'),
     Output('btn-heatmap', 'color')],
    [Input('view-mode', 'data')]
)
def update_view_buttons(view_mode):
    colors = ['secondary', 'secondary', 'secondary']
    if view_mode == 'positions':
        colors[0] = 'primary'
    elif view_mode == 'shotmap':
        colors[1] = 'primary'
    elif view_mode == 'heatmap':
        colors[2] = 'primary'
    return colors


@app.callback(
    Output('player-select', 'options'),
    [Input('interval-component', 'n_intervals')]
)
def update_player_options(n):
    options = []
    for p in game_state.home_players:
        options.append({'label': f"Home #{p.jersey_number}", 'value': p.player_id})
    for p in game_state.away_players:
        options.append({'label': f"Away #{p.jersey_number}", 'value': p.player_id})
    return options


@app.callback(
    [Output('player-distance', 'children'),
     Output('player-velocity', 'children'),
     Output('player-sprints', 'children'),
     Output('player-fatigue', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('player-select', 'value')]
)
def update_player_stats(n, player_id):
    if not player_id:
        return "- m", "- m/s", "-", "-"

    player = None
    for p in game_state.home_players + game_state.away_players:
        if p.player_id == player_id:
            player = p
            break

    if not player:
        return "- m", "- m/s", "-", "-"

    fatigue = game_state.fatigue.get(player_id, 0)

    return (
        f"{player.distance_covered:.0f} m",
        f"{player.velocity:.1f} m/s",
        str(player.sprints),
        f"{fatigue * 100:.0f}%"
    )


@app.callback(
    [Output('goalie-graph', 'figure'),
     Output('goalie-coverage', 'children'),
     Output('goalie-saves', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('goalie-select', 'value')]
)
def update_goalie_stats(n, team):
    fig = create_goalie_figure(team)
    goalie = game_state.home_goalie if team == 'home' else game_state.away_goalie
    return fig, f"{goalie['coverage']:.0f}%", str(goalie['saves'])


@app.callback(
    [Output('pitch-graph', 'figure'),
     Output('match-time', 'children'),
     Output('frame-counter', 'children'),
     Output('home-score', 'children'),
     Output('away-score', 'children'),
     Output('home-xg', 'children'),
     Output('away-xg', 'children'),
     Output('xg-timeline', 'figure'),
     Output('possession-graph', 'figure'),
     Output('fatigue-graph', 'figure'),
     Output('home-ppda', 'children'),
     Output('home-press-success', 'children'),
     Output('away-ppda', 'children'),
     Output('away-press-success', 'children'),
     Output('home-buildup', 'children'),
     Output('home-keypasses', 'children'),
     Output('away-buildup', 'children'),
     Output('away-keypasses', 'children'),
     Output('insights-feed', 'children'),
     Output('shots-list', 'children'),
     Output('shot-count', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('view-mode', 'data')]
)
def update_dashboard(n, view_mode):
    # Update game state
    game_state.update()

    # Create figures
    pitch_fig = create_pitch_figure(view_mode)
    xg_timeline_fig = create_xg_timeline_figure()
    possession_fig = create_possession_figure()
    fatigue_fig = create_fatigue_figure()

    # Create insights list
    insights_children = []
    for insight in game_state.insights[:10]:
        css_class = "insight-item"
        if "GOAL" in insight:
            css_class += " insight-goal"
        elif "Shot" in insight:
            css_class += " insight-shot"
        insights_children.append(html.Div(insight, className=css_class))

    # Create shots list
    shots_children = []
    for shot in reversed(game_state.shots[-10:]):
        time_str = f"{int(shot.time // 60):02d}:{int(shot.time % 60):02d}"
        color_class = "home-color" if shot.team == 'home' else "away-color"
        shots_children.append(html.Div([
            html.Span(time_str, className="text-secondary me-2"),
            html.Span(shot.team.upper(), className=f"{color_class} me-2"),
            html.Span(shot.result, className="me-2",
                     style={'color': COLORS.get(shot.result, '#666')}),
            html.Span(f"xG: {shot.xg:.2f}", className="text-white")
        ], className="insight-item"))

    return (
        pitch_fig,
        game_state.format_time(),
        f"Frame: {game_state.frame}",
        str(game_state.score['home']),
        str(game_state.score['away']),
        f"{game_state.xg['home']:.2f}",
        f"{game_state.xg['away']:.2f}",
        xg_timeline_fig,
        possession_fig,
        fatigue_fig,
        f"{game_state.pressure['home']['ppda']:.1f}",
        f"{game_state.pressure['home']['press_success']:.0f}%",
        f"{game_state.pressure['away']['ppda']:.1f}",
        f"{game_state.pressure['away']['press_success']:.0f}%",
        game_state.patterns['home']['build_up'],
        str(game_state.patterns['home']['key_passes']),
        game_state.patterns['away']['build_up'],
        str(game_state.patterns['away']['key_passes']),
        insights_children,
        shots_children,
        str(len(game_state.shots))
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Starting Soccer Analytics Dashboard...")
    print("Open http://localhost:8050 in your browser")
    app.run_server(debug=True, host='0.0.0.0', port=8050)
