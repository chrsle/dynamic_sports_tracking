/**
 * Soccer Analytics Dashboard
 * Real-time soccer match analytics visualization
 */

// ============================================================================
// CONSTANTS
// ============================================================================

const PITCH = {
    LENGTH: 105,
    WIDTH: 68,
    BOX_LENGTH: 16.5,
    BOX_WIDTH: 40.32,
    SIX_YARD_LENGTH: 5.5,
    SIX_YARD_WIDTH: 18.32,
    GOAL_WIDTH: 7.32,
    CENTER_CIRCLE_RADIUS: 9.15,
    PENALTY_SPOT: 11
};

const COLORS = {
    HOME: '#3b82f6',
    HOME_LIGHT: '#60a5fa',
    AWAY: '#ef4444',
    AWAY_LIGHT: '#f87171',
    BALL: '#ffffff',
    PITCH: '#1a472a',
    PITCH_LINES: 'rgba(255, 255, 255, 0.8)',
    GOAL: '#ffd700',
    HEATMAP_LOW: 'rgba(0, 100, 255, 0.1)',
    HEATMAP_HIGH: 'rgba(255, 50, 50, 0.8)'
};

// ============================================================================
// STATE
// ============================================================================

const state = {
    isPlaying: false,
    playbackSpeed: 1.0,
    currentFrame: 0,
    matchTime: 0,
    ws: null,
    connected: false,

    // Team data
    homeTeam: 'Home',
    awayTeam: 'Away',
    score: { home: 0, away: 0 },
    xg: { home: 0, away: 0 },
    possession: { home: 50, away: 50 },
    momentum: 0.5,

    // Players
    homePlayers: [],
    awayPlayers: [],
    ball: { x: 52.5, y: 34 },

    // Shots
    shots: [],
    xgTimeline: [],

    // View mode
    viewMode: 'positions',  // positions, heatmap, shotmap, pressure

    // Selected player
    selectedPlayerId: null,
    selectedGoalieTeam: 'home',

    // Heatmap data
    heatmapData: null,

    // Analytics
    fatigue: {},
    pressure: { home: {}, away: {} },
    patterns: { home: {}, away: {} },
    homeGoalie: null,
    awayGoalie: null,
    manAdvantage: { is_active: false }
};

// ============================================================================
// CANVAS & CHARTS
// ============================================================================

let pitchCanvas, pitchCtx;
let goalieCanvas, goalieCtx;
let xgChart, xgTimelineChart;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initCanvas();
    initCharts();
    initEventListeners();
    initWebSocket();
    initDemoMode();
});

function initCanvas() {
    pitchCanvas = document.getElementById('pitchCanvas');
    pitchCtx = pitchCanvas.getContext('2d');

    goalieCanvas = document.getElementById('goalieCanvas');
    goalieCtx = goalieCanvas.getContext('2d');

    // Set canvas size
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

function resizeCanvas() {
    const container = pitchCanvas.parentElement;
    const maxWidth = container.clientWidth - 32;
    const aspectRatio = PITCH.WIDTH / PITCH.LENGTH;

    pitchCanvas.width = Math.min(700, maxWidth);
    pitchCanvas.height = pitchCanvas.width * aspectRatio;

    drawPitch();
}

function initCharts() {
    // xG Comparison Chart (radial bar)
    xgChart = new ApexCharts(document.getElementById('xgChart'), {
        chart: {
            type: 'radialBar',
            height: 80,
            sparkline: { enabled: true }
        },
        series: [50, 50],
        colors: [COLORS.HOME, COLORS.AWAY],
        plotOptions: {
            radialBar: {
                hollow: { size: '30%' },
                track: { background: '#1a2235' },
                dataLabels: { show: false }
            }
        },
        labels: ['Home', 'Away']
    });
    xgChart.render();

    // xG Timeline Chart
    xgTimelineChart = new ApexCharts(document.getElementById('xgTimeline'), {
        chart: {
            type: 'area',
            height: 150,
            toolbar: { show: false },
            animations: { enabled: true, speed: 300 },
            background: 'transparent'
        },
        series: [
            { name: 'Home xG', data: [] },
            { name: 'Away xG', data: [] }
        ],
        colors: [COLORS.HOME, COLORS.AWAY],
        stroke: { curve: 'smooth', width: 2 },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.4,
                opacityTo: 0.1
            }
        },
        xaxis: {
            type: 'numeric',
            labels: {
                formatter: (val) => formatTime(val),
                style: { colors: '#6b7280' }
            }
        },
        yaxis: {
            labels: {
                formatter: (val) => val.toFixed(2),
                style: { colors: '#6b7280' }
            }
        },
        grid: {
            borderColor: '#374151',
            strokeDashArray: 4
        },
        tooltip: {
            theme: 'dark',
            x: { formatter: (val) => formatTime(val) },
            y: { formatter: (val) => val.toFixed(3) }
        },
        legend: { show: false }
    });
    xgTimelineChart.render();
}

function initEventListeners() {
    // Playback controls
    document.getElementById('playPauseBtn').addEventListener('click', togglePlayback);
    document.getElementById('prevFrameBtn').addEventListener('click', () => skipFrames(-1));
    document.getElementById('nextFrameBtn').addEventListener('click', () => skipFrames(1));
    document.getElementById('skipBackBtn').addEventListener('click', () => skipSeconds(-10));
    document.getElementById('skipForwardBtn').addEventListener('click', () => skipSeconds(10));

    // Speed control
    document.getElementById('speedSelect').addEventListener('change', (e) => {
        state.playbackSpeed = parseFloat(e.target.value);
        sendCommand({ action: 'playback', value: state.playbackSpeed });
    });

    // Frame slider
    document.getElementById('frameSlider').addEventListener('input', (e) => {
        const frame = parseInt(e.target.value);
        seekToFrame(frame);
    });

    // View toggles
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.viewMode = e.target.dataset.view;
            drawPitch();
        });
    });

    // Player select
    document.getElementById('playerSelect').addEventListener('change', (e) => {
        state.selectedPlayerId = e.target.value ? parseInt(e.target.value) : null;
        updatePlayerStats();
    });

    // Goalie team select
    document.getElementById('goalieTeamSelect').addEventListener('change', (e) => {
        state.selectedGoalieTeam = e.target.value;
        updateGoalieStats();
    });

    // Query modal
    document.getElementById('queryBtn').addEventListener('click', () => {
        document.getElementById('queryModal').style.display = 'flex';
    });

    document.getElementById('closeModal').addEventListener('click', () => {
        document.getElementById('queryModal').style.display = 'none';
    });

    document.getElementById('runQuery').addEventListener('click', runQuery);

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
}

function handleKeyboard(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    switch (e.code) {
        case 'Space':
            e.preventDefault();
            togglePlayback();
            break;
        case 'ArrowLeft':
            e.preventDefault();
            skipFrames(e.shiftKey ? -10 : -1);
            break;
        case 'ArrowRight':
            e.preventDefault();
            skipFrames(e.shiftKey ? 10 : 1);
            break;
        case 'KeyQ':
            document.getElementById('queryModal').style.display = 'flex';
            break;
        case 'Escape':
            document.getElementById('queryModal').style.display = 'none';
            break;
    }
}

// ============================================================================
// WEBSOCKET
// ============================================================================

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    try {
        state.ws = new WebSocket(wsUrl);

        state.ws.onopen = () => {
            state.connected = true;
            updateConnectionStatus(true);
        };

        state.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleMessage(data);
        };

        state.ws.onclose = () => {
            state.connected = false;
            updateConnectionStatus(false);
            // Try to reconnect after 3 seconds
            setTimeout(initWebSocket, 3000);
        };

        state.ws.onerror = () => {
            state.connected = false;
            updateConnectionStatus(false);
        };
    } catch (e) {
        console.log('WebSocket not available, using demo mode');
        updateConnectionStatus(false, 'Demo Mode');
    }
}

function sendCommand(command) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(command));
    }
}

function handleMessage(data) {
    switch (data.type) {
        case 'init':
            handleInit(data);
            break;
        case 'frame':
            handleFrame(data);
            break;
        case 'playback_state':
            handlePlaybackState(data);
            break;
        case 'query_result':
            handleQueryResult(data);
            break;
        default:
            // Handle as frame data if no type
            if (data.frame !== undefined) {
                handleFrame(data);
            }
    }
}

function handleInit(data) {
    if (data.team_config) {
        state.homeTeam = data.team_config.home_name;
        state.awayTeam = data.team_config.away_name;
        updateTeamNames();
    }
}

function handleFrame(data) {
    // Update state from frame data
    state.currentFrame = data.frame || state.currentFrame;
    state.matchTime = data.time || state.matchTime;

    if (data.ball) {
        state.ball = data.ball;
    }

    if (data.home_players) {
        state.homePlayers = data.home_players;
    }

    if (data.away_players) {
        state.awayPlayers = data.away_players;
    }

    if (data.xg) {
        state.xg = data.xg;
    }

    if (data.score) {
        state.score = data.score;
    }

    if (data.possession) {
        state.possession = data.possession;
    }

    if (data.momentum !== undefined) {
        state.momentum = data.momentum;
    }

    if (data.shots) {
        state.shots = data.shots;
    }

    if (data.xg_timeline) {
        state.xgTimeline = data.xg_timeline;
    }

    if (data.man_advantage) {
        state.manAdvantage = data.man_advantage;
    }

    if (data.home_goalie) {
        state.homeGoalie = data.home_goalie;
    }

    if (data.away_goalie) {
        state.awayGoalie = data.away_goalie;
    }

    if (data.fatigue) {
        state.fatigue = data.fatigue;
    }

    if (data.pressure) {
        state.pressure = data.pressure;
    }

    if (data.patterns) {
        state.patterns = data.patterns;
    }

    // Handle new shot event
    if (data.new_shot) {
        addInsight(data.new_shot);
    }

    // Update UI
    updateUI();
}

function handlePlaybackState(data) {
    state.isPlaying = data.is_playing;
    state.playbackSpeed = data.speed;
    state.currentFrame = data.frame;
    updatePlaybackUI();
}

function handleQueryResult(data) {
    displayQueryResults(data.data);
}

// ============================================================================
// DEMO MODE
// ============================================================================

let demoInterval;

function initDemoMode() {
    // Start demo mode if no WebSocket connection after 2 seconds
    setTimeout(() => {
        if (!state.connected) {
            startDemoMode();
        }
    }, 2000);
}

function startDemoMode() {
    updateConnectionStatus(false, 'Demo Mode');

    // Initialize players
    initializeDemoPlayers();

    // Start demo updates
    demoInterval = setInterval(generateDemoFrame, 33); // ~30 fps
}

function initializeDemoPlayers() {
    // Home team (4-3-3)
    const homeFormation = [
        { x: 5, y: 34, num: 1 },    // GK
        { x: 25, y: 10, num: 2 },   // RB
        { x: 25, y: 25, num: 4 },   // CB
        { x: 25, y: 43, num: 5 },   // CB
        { x: 25, y: 58, num: 3 },   // LB
        { x: 45, y: 20, num: 6 },   // CDM
        { x: 45, y: 34, num: 8 },   // CM
        { x: 45, y: 48, num: 10 },  // CAM
        { x: 70, y: 15, num: 7 },   // RW
        { x: 75, y: 34, num: 9 },   // ST
        { x: 70, y: 53, num: 11 }   // LW
    ];

    state.homePlayers = homeFormation.map((p, i) => ({
        player_id: i + 1,
        team: 'home',
        position: { x: p.x, y: p.y },
        jersey_number: p.num,
        velocity: 0,
        distance_covered: 0,
        sprints: 0
    }));

    // Away team (4-4-2)
    const awayFormation = [
        { x: 100, y: 34, num: 1 },  // GK
        { x: 80, y: 10, num: 2 },   // RB
        { x: 80, y: 25, num: 4 },   // CB
        { x: 80, y: 43, num: 5 },   // CB
        { x: 80, y: 58, num: 3 },   // LB
        { x: 60, y: 8, num: 7 },    // RM
        { x: 60, y: 27, num: 8 },   // CM
        { x: 60, y: 41, num: 6 },   // CM
        { x: 60, y: 60, num: 11 },  // LM
        { x: 35, y: 25, num: 9 },   // ST
        { x: 35, y: 43, num: 10 }   // ST
    ];

    state.awayPlayers = awayFormation.map((p, i) => ({
        player_id: i + 12,
        team: 'away',
        position: { x: p.x, y: p.y },
        jersey_number: p.num,
        velocity: 0,
        distance_covered: 0,
        sprints: 0
    }));

    // Populate player select
    populatePlayerSelect();
}

function generateDemoFrame() {
    if (!state.isPlaying) return;

    state.currentFrame++;
    state.matchTime += 1 / 30;

    // Update ball position
    updateDemoBall();

    // Update player positions
    updateDemoPlayers();

    // Maybe generate a shot
    maybeGenerateDemoShot();

    // Update possession
    updateDemoPossession();

    // Update momentum
    updateDemoMomentum();

    // Update fatigue
    updateDemoFatigue();

    // Update pressure
    updateDemoPressure();

    // Update patterns
    updateDemoPatterns();

    // Update goalie metrics
    updateDemoGoalies();

    // Update UI
    updateUI();
}

function updateDemoBall() {
    const targetX = state.momentum > 0.5 ? 80 : 25;
    const targetY = PITCH.WIDTH / 2 + (Math.random() - 0.5) * 40;

    state.ball.x += (targetX - state.ball.x) * 0.02 + (Math.random() - 0.5) * 2;
    state.ball.y += (targetY - state.ball.y) * 0.02 + (Math.random() - 0.5) * 2;

    state.ball.x = Math.max(0, Math.min(PITCH.LENGTH, state.ball.x));
    state.ball.y = Math.max(0, Math.min(PITCH.WIDTH, state.ball.y));
}

function updateDemoPlayers() {
    const allPlayers = [...state.homePlayers, ...state.awayPlayers];

    allPlayers.forEach(player => {
        const isHome = player.team === 'home';
        const pos = player.position;

        // Move towards ball if nearby
        const dx = state.ball.x - pos.x;
        const dy = state.ball.y - pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        let moveX = (Math.random() - 0.5) * 0.5;
        let moveY = (Math.random() - 0.5) * 0.5;

        if (dist < 20) {
            moveX += dx * 0.03;
            moveY += dy * 0.03;
        }

        pos.x += moveX;
        pos.y += moveY;

        // Keep in bounds
        if (isHome) {
            pos.x = Math.max(0, Math.min(95, pos.x));
        } else {
            pos.x = Math.max(10, Math.min(105, pos.x));
        }
        pos.y = Math.max(2, Math.min(66, pos.y));

        // Update velocity
        player.velocity = Math.sqrt(moveX * moveX + moveY * moveY) * 30;
        player.distance_covered += player.velocity / 30;

        if (player.velocity > 7) {
            player.sprints++;
        }
    });
}

function maybeGenerateDemoShot() {
    if (Math.random() > 0.002) return;

    const team = state.ball.x > 70 ? 'home' : state.ball.x < 35 ? 'away' : null;
    if (!team) return;

    let shotX = state.ball.x + (Math.random() - 0.5) * 10;
    let shotY = state.ball.y + (Math.random() - 0.5) * 10;

    if (team === 'home') {
        shotX = Math.max(70, Math.min(100, shotX));
    } else {
        shotX = Math.max(5, Math.min(35, shotX));
    }

    const goalX = team === 'home' ? PITCH.LENGTH : 0;
    const goalY = PITCH.WIDTH / 2;
    const distance = Math.sqrt((shotX - goalX) ** 2 + (shotY - goalY) ** 2);

    // Calculate xG
    let xg;
    if (distance < 6) {
        xg = 0.6;
    } else if (distance < 12) {
        xg = 0.25;
    } else if (distance < 20) {
        xg = 0.1;
    } else {
        xg = 0.03;
    }

    const angle = Math.abs(Math.atan2(shotY - goalY, Math.abs(shotX - goalX)));
    xg *= Math.max(0.3, 1 - angle / Math.PI);

    // Determine result
    const rand = Math.random();
    let result;
    if (rand < xg * 0.8) {
        result = 'goal';
        if (team === 'home') {
            state.score.home++;
        } else {
            state.score.away++;
        }
    } else if (rand < xg + 0.2) {
        result = 'saved';
    } else if (rand < xg + 0.4) {
        result = 'blocked';
    } else if (rand < xg + 0.5) {
        result = 'post';
    } else {
        result = 'off_target';
    }

    // Update xG
    if (team === 'home') {
        state.xg.home += xg;
    } else {
        state.xg.away += xg;
    }

    // Add shot
    const shot = {
        shot_id: state.shots.length + 1,
        team,
        position: { x: shotX, y: shotY },
        result,
        xg: Math.round(xg * 1000) / 1000,
        distance: Math.round(distance * 10) / 10,
        time: state.matchTime,
        time_formatted: formatTime(state.matchTime)
    };

    state.shots.push(shot);
    state.xgTimeline.push({
        time: state.matchTime,
        home_xg: state.xg.home,
        away_xg: state.xg.away
    });

    addInsight(shot);
}

function updateDemoPossession() {
    const targetPoss = state.ball.x > 52.5 ? 60 : 40;
    state.possession.home += (targetPoss - state.possession.home) * 0.01;
    state.possession.away = 100 - state.possession.home;
}

function updateDemoMomentum() {
    const ballFactor = (state.ball.x - 52.5) / 52.5 * 0.3;
    state.momentum += ballFactor * 0.01 + (Math.random() - 0.5) * 0.02;
    state.momentum = Math.max(0, Math.min(1, state.momentum));
}

function updateDemoFatigue() {
    const allPlayers = [...state.homePlayers, ...state.awayPlayers];
    const minutes = state.matchTime / 60;

    allPlayers.forEach(player => {
        const distance = player.distance_covered;
        const fatigueLevel = Math.min(1.0, (distance / 1000) * (minutes / 45) * 0.5);

        state.fatigue[player.player_id] = {
            player_id: player.player_id,
            fatigue_level: fatigueLevel,
            distance_covered: distance,
            avg_speed_decline: fatigueLevel * 20,
            sprint_decline: fatigueLevel * 35
        };
    });
}

function updateDemoPressure() {
    state.pressure = {
        home: {
            ppda: 8 + (Math.random() - 0.5) * 4,
            high_press_success_rate: 35 + (Math.random() - 0.5) * 20,
            counter_press_intensity: 0.7 + (Math.random() - 0.5) * 0.4
        },
        away: {
            ppda: 10 + (Math.random() - 0.5) * 4,
            high_press_success_rate: 30 + (Math.random() - 0.5) * 20,
            counter_press_intensity: 0.6 + (Math.random() - 0.5) * 0.4
        }
    };
}

function updateDemoPatterns() {
    const buildUpOptions = ['slow', 'medium', 'fast', 'direct'];
    state.patterns = {
        home: {
            build_up_speed: buildUpOptions[Math.floor(Math.random() * 3)],
            attacking_width: 45 + (Math.random() - 0.5) * 20,
            key_passes: Math.floor(state.matchTime / 60 * 2)
        },
        away: {
            build_up_speed: buildUpOptions[Math.floor(Math.random() * 4)],
            attacking_width: 40 + (Math.random() - 0.5) * 20,
            key_passes: Math.floor(state.matchTime / 60 * 1.5)
        }
    };
}

function updateDemoGoalies() {
    const homeGK = state.homePlayers.find(p => p.jersey_number === 1);
    const awayGK = state.awayPlayers.find(p => p.jersey_number === 1);

    if (homeGK) {
        state.homeGoalie = {
            position: homeGK.position,
            coverage_area: 60 + homeGK.position.x * 2,
            positioning_quality: 0.7 + Math.random() * 0.2,
            distance_from_line: homeGK.position.x,
            saves: state.shots.filter(s => s.team === 'away' && s.result === 'saved').length
        };
    }

    if (awayGK) {
        state.awayGoalie = {
            position: awayGK.position,
            coverage_area: 60 + (105 - awayGK.position.x) * 2,
            positioning_quality: 0.7 + Math.random() * 0.2,
            distance_from_line: 105 - awayGK.position.x,
            saves: state.shots.filter(s => s.team === 'home' && s.result === 'saved').length
        };
    }
}

// ============================================================================
// UI UPDATES
// ============================================================================

function updateUI() {
    updateMatchInfo();
    updateXG();
    updatePossession();
    updateMomentum();
    updateCharts();
    updateShots();
    updatePlayerStats();
    updateGoalieStats();
    updatePressureStats();
    updatePatternStats();
    updateFatigueGrid();
    updateManAdvantage();
    drawPitch();
    drawGoalieViz();
}

function updateMatchInfo() {
    document.getElementById('matchTime').textContent = formatTime(state.matchTime);
    document.getElementById('homeScore').textContent = state.score.home;
    document.getElementById('awayScore').textContent = state.score.away;
    document.getElementById('frameCounter').textContent = `Frame: ${state.currentFrame}`;
}

function updateTeamNames() {
    document.getElementById('homeTeamName').textContent = state.homeTeam;
    document.getElementById('awayTeamName').textContent = state.awayTeam;
}

function updateXG() {
    document.getElementById('homeXg').textContent = state.xg.home.toFixed(2);
    document.getElementById('awayXg').textContent = state.xg.away.toFixed(2);
}

function updatePossession() {
    document.getElementById('homePoss').textContent = Math.round(state.possession.home);
    document.getElementById('awayPoss').textContent = Math.round(state.possession.away);
    document.getElementById('homePossFill').style.width = `${state.possession.home}%`;
}

function updateMomentum() {
    document.getElementById('momentumMarker').style.left = `${state.momentum * 100}%`;
}

function updateCharts() {
    // Update xG radial chart
    const totalXg = state.xg.home + state.xg.away || 1;
    xgChart.updateSeries([
        (state.xg.home / totalXg * 100),
        (state.xg.away / totalXg * 100)
    ]);

    // Update xG timeline
    if (state.xgTimeline.length > 0) {
        const homeData = state.xgTimeline.map(p => ({ x: p.time, y: p.home_xg }));
        const awayData = state.xgTimeline.map(p => ({ x: p.time, y: p.away_xg }));

        xgTimelineChart.updateSeries([
            { name: 'Home xG', data: homeData },
            { name: 'Away xG', data: awayData }
        ]);
    }
}

function updateShots() {
    const container = document.getElementById('shotsList');
    const recentShots = state.shots.slice(-10).reverse();

    container.innerHTML = recentShots.map(shot => `
        <div class="shot-item ${shot.team}">
            <span class="shot-time">${shot.time_formatted || formatTime(shot.time)}</span>
            <span class="shot-result ${shot.result}">${shot.result}</span>
            <span class="shot-xg">xG: ${shot.xg.toFixed(2)}</span>
        </div>
    `).join('');

    document.getElementById('shotCount').textContent = state.shots.length;

    // Update shot markers on timeline
    updateShotMarkers();
}

function updateShotMarkers() {
    const container = document.getElementById('shotMarkers');
    const maxTime = state.matchTime || 1;

    container.innerHTML = state.shots.map(shot => {
        const position = (shot.time / maxTime * 100);
        return `<div class="shot-marker ${shot.result}" style="left: ${position}%"></div>`;
    }).join('');
}

function updatePlayerStats() {
    const playerId = state.selectedPlayerId;
    if (!playerId) return;

    const player = [...state.homePlayers, ...state.awayPlayers]
        .find(p => p.player_id === playerId);

    if (!player) return;

    document.getElementById('playerDistance').textContent = `${Math.round(player.distance_covered)} m`;
    document.getElementById('playerSprints').textContent = player.sprints;
    document.getElementById('playerVelocity').textContent = `${player.velocity.toFixed(1)} m/s`;

    const fatigue = state.fatigue[playerId];
    if (fatigue) {
        document.getElementById('playerFatigue').style.width = `${fatigue.fatigue_level * 100}%`;
    }
}

function updateGoalieStats() {
    const goalie = state.selectedGoalieTeam === 'home' ? state.homeGoalie : state.awayGoalie;
    if (!goalie) return;

    document.getElementById('goalieCoverage').textContent = `${Math.round(goalie.coverage_area)}%`;
    document.getElementById('goaliePositioning').style.width = `${goalie.positioning_quality * 100}%`;
    document.getElementById('goalieSaves').textContent = goalie.saves || 0;
    document.getElementById('goalieDistance').textContent = `${goalie.distance_from_line?.toFixed(1) || 0} m`;
}

function updatePressureStats() {
    if (state.pressure.home) {
        document.getElementById('homePPDA').textContent = state.pressure.home.ppda?.toFixed(1) || '-';
        document.getElementById('homePressSuccess').textContent = `${Math.round(state.pressure.home.high_press_success_rate || 0)}%`;
        document.getElementById('homeCounterPress').style.width = `${(state.pressure.home.counter_press_intensity || 0) * 70}%`;
    }

    if (state.pressure.away) {
        document.getElementById('awayPPDA').textContent = state.pressure.away.ppda?.toFixed(1) || '-';
        document.getElementById('awayPressSuccess').textContent = `${Math.round(state.pressure.away.high_press_success_rate || 0)}%`;
        document.getElementById('awayCounterPress').style.width = `${(state.pressure.away.counter_press_intensity || 0) * 70}%`;
    }
}

function updatePatternStats() {
    if (state.patterns.home) {
        document.getElementById('homeBuildUp').textContent = capitalize(state.patterns.home.build_up_speed || 'medium');
        document.getElementById('homeWidth').textContent = `${Math.round(state.patterns.home.attacking_width || 45)}m`;
        document.getElementById('homeKeyPasses').textContent = state.patterns.home.key_passes || 0;
    }

    if (state.patterns.away) {
        document.getElementById('awayBuildUp').textContent = capitalize(state.patterns.away.build_up_speed || 'medium');
        document.getElementById('awayWidth').textContent = `${Math.round(state.patterns.away.attacking_width || 40)}m`;
        document.getElementById('awayKeyPasses').textContent = state.patterns.away.key_passes || 0;
    }
}

function updateFatigueGrid() {
    const container = document.getElementById('fatigueGrid');
    const allPlayers = [...state.homePlayers, ...state.awayPlayers];

    container.innerHTML = allPlayers.map(player => {
        const fatigue = state.fatigue[player.player_id];
        const level = fatigue?.fatigue_level || 0;
        const levelClass = level > 0.7 ? 'danger' : level > 0.4 ? 'warning' : '';

        return `
            <div class="fatigue-item ${player.team}">
                <span class="player-num">${player.jersey_number}</span>
                <div class="mini-bar">
                    <div class="mini-fill ${levelClass}" style="width: ${level * 100}%"></div>
                </div>
            </div>
        `;
    }).join('');
}

function updateManAdvantage() {
    const badge = document.getElementById('manAdvantage');
    if (state.manAdvantage.is_active) {
        badge.style.display = 'block';
        badge.querySelector('.advantage-badge').textContent = state.manAdvantage.situation || '11v10';
    } else {
        badge.style.display = 'none';
    }
}

function updateConnectionStatus(connected, text = null) {
    const status = document.getElementById('connectionStatus');
    const dot = status.querySelector('.status-dot');
    const span = status.querySelector('span:last-child');

    if (connected) {
        dot.classList.add('connected');
        span.textContent = 'Connected';
    } else {
        dot.classList.remove('connected');
        span.textContent = text || 'Disconnected';
    }
}

function updatePlaybackUI() {
    const playIcon = document.getElementById('playIcon');
    const pauseIcon = document.getElementById('pauseIcon');

    if (state.isPlaying) {
        playIcon.style.display = 'none';
        pauseIcon.style.display = 'block';
    } else {
        playIcon.style.display = 'block';
        pauseIcon.style.display = 'none';
    }
}

function populatePlayerSelect() {
    const select = document.getElementById('playerSelect');
    select.innerHTML = '<option value="">Select Player</option>';

    state.homePlayers.forEach(p => {
        select.innerHTML += `<option value="${p.player_id}">Home #${p.jersey_number}</option>`;
    });

    state.awayPlayers.forEach(p => {
        select.innerHTML += `<option value="${p.player_id}">Away #${p.jersey_number}</option>`;
    });
}

function addInsight(shot) {
    const container = document.getElementById('insightsFeed');
    const isGoal = shot.result === 'goal';

    const insight = document.createElement('div');
    insight.className = `insight ${isGoal ? 'goal' : 'shot'}`;
    insight.innerHTML = `
        <span class="insight-time">${shot.time_formatted || formatTime(shot.time)}</span>
        <span class="insight-text">
            ${isGoal ? '⚽ GOAL!' : 'Shot'} by ${capitalize(shot.team)} -
            ${shot.result} (xG: ${shot.xg.toFixed(2)})
        </span>
    `;

    container.insertBefore(insight, container.firstChild);

    // Keep only last 10 insights
    while (container.children.length > 10) {
        container.removeChild(container.lastChild);
    }
}

// ============================================================================
// PITCH DRAWING
// ============================================================================

function drawPitch() {
    const ctx = pitchCtx;
    const canvas = pitchCanvas;
    const w = canvas.width;
    const h = canvas.height;

    // Scale factors
    const scaleX = w / PITCH.LENGTH;
    const scaleY = h / PITCH.WIDTH;

    // Clear canvas
    ctx.fillStyle = COLORS.PITCH;
    ctx.fillRect(0, 0, w, h);

    // Draw pitch markings
    ctx.strokeStyle = COLORS.PITCH_LINES;
    ctx.lineWidth = 2;

    // Outer boundary
    ctx.strokeRect(0, 0, w, h);

    // Center line
    ctx.beginPath();
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, PITCH.CENTER_CIRCLE_RADIUS * scaleX, 0, Math.PI * 2);
    ctx.stroke();

    // Center spot
    ctx.fillStyle = COLORS.PITCH_LINES;
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 3, 0, Math.PI * 2);
    ctx.fill();

    // Penalty areas (left)
    const boxWidth = PITCH.BOX_LENGTH * scaleX;
    const boxHeight = PITCH.BOX_WIDTH * scaleY;
    const boxY = (h - boxHeight) / 2;

    ctx.strokeRect(0, boxY, boxWidth, boxHeight);

    // Six-yard box (left)
    const sixWidth = PITCH.SIX_YARD_LENGTH * scaleX;
    const sixHeight = PITCH.SIX_YARD_WIDTH * scaleY;
    const sixY = (h - sixHeight) / 2;

    ctx.strokeRect(0, sixY, sixWidth, sixHeight);

    // Penalty spot (left)
    ctx.beginPath();
    ctx.arc(PITCH.PENALTY_SPOT * scaleX, h / 2, 3, 0, Math.PI * 2);
    ctx.fill();

    // Penalty areas (right)
    ctx.strokeRect(w - boxWidth, boxY, boxWidth, boxHeight);
    ctx.strokeRect(w - sixWidth, sixY, sixWidth, sixHeight);

    // Penalty spot (right)
    ctx.beginPath();
    ctx.arc(w - PITCH.PENALTY_SPOT * scaleX, h / 2, 3, 0, Math.PI * 2);
    ctx.fill();

    // Goals
    const goalWidth = 3;
    const goalHeight = PITCH.GOAL_WIDTH * scaleY;
    const goalY = (h - goalHeight) / 2;

    ctx.fillStyle = COLORS.GOAL;
    ctx.fillRect(-goalWidth, goalY, goalWidth, goalHeight);
    ctx.fillRect(w, goalY, goalWidth, goalHeight);

    // Draw based on view mode
    switch (state.viewMode) {
        case 'positions':
            drawPlayers(scaleX, scaleY);
            break;
        case 'heatmap':
            drawHeatmap(scaleX, scaleY);
            drawPlayers(scaleX, scaleY);
            break;
        case 'shotmap':
            drawShotMap(scaleX, scaleY);
            break;
        case 'pressure':
            drawPressureZones(scaleX, scaleY);
            drawPlayers(scaleX, scaleY);
            break;
    }
}

function drawPlayers(scaleX, scaleY) {
    const ctx = pitchCtx;

    // Draw home players
    state.homePlayers.forEach(player => {
        const x = player.position.x * scaleX;
        const y = player.position.y * scaleY;

        ctx.fillStyle = COLORS.HOME;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();

        // Jersey number
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(player.jersey_number, x, y);
    });

    // Draw away players
    state.awayPlayers.forEach(player => {
        const x = player.position.x * scaleX;
        const y = player.position.y * scaleY;

        ctx.fillStyle = COLORS.AWAY;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();

        // Jersey number
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(player.jersey_number, x, y);
    });

    // Draw ball
    const ballX = state.ball.x * scaleX;
    const ballY = state.ball.y * scaleY;

    ctx.fillStyle = COLORS.BALL;
    ctx.beginPath();
    ctx.arc(ballX, ballY, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawHeatmap(scaleX, scaleY) {
    const ctx = pitchCtx;
    const resolution = 10;

    // Aggregate player positions into heatmap
    const heatmap = {};
    const allPlayers = [...state.homePlayers, ...state.awayPlayers];

    allPlayers.forEach(player => {
        const gridX = Math.floor(player.position.x / resolution);
        const gridY = Math.floor(player.position.y / resolution);
        const key = `${gridX},${gridY}`;
        heatmap[key] = (heatmap[key] || 0) + 1;
    });

    const maxVal = Math.max(...Object.values(heatmap), 1);

    Object.entries(heatmap).forEach(([key, val]) => {
        const [gridX, gridY] = key.split(',').map(Number);
        const intensity = val / maxVal;

        const x = gridX * resolution * scaleX;
        const y = gridY * resolution * scaleY;
        const w = resolution * scaleX;
        const h = resolution * scaleY;

        const r = Math.round(255 * intensity);
        const g = Math.round(100 * (1 - intensity));
        const b = Math.round(255 * (1 - intensity));

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.3 + intensity * 0.4})`;
        ctx.fillRect(x, y, w, h);
    });
}

function drawShotMap(scaleX, scaleY) {
    const ctx = pitchCtx;

    state.shots.forEach(shot => {
        const x = shot.position.x * scaleX;
        const y = shot.position.y * scaleY;
        const size = 5 + shot.xg * 20;

        // Color based on result
        let color;
        switch (shot.result) {
            case 'goal':
                color = '#10b981';
                break;
            case 'saved':
                color = '#f59e0b';
                break;
            case 'blocked':
                color = '#6b7280';
                break;
            default:
                color = '#374151';
        }

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.8;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();

        // Border
        ctx.strokeStyle = shot.team === 'home' ? COLORS.HOME : COLORS.AWAY;
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.globalAlpha = 1;
    });

    // Legend
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#fff';
    ctx.fillText('Size = xG', 10, 20);
}

function drawPressureZones(scaleX, scaleY) {
    const ctx = pitchCtx;

    // Draw pressure intensity based on player clustering
    const allPlayers = [...state.homePlayers, ...state.awayPlayers];
    const gridSize = 15;

    for (let x = 0; x < PITCH.LENGTH; x += gridSize) {
        for (let y = 0; y < PITCH.WIDTH; y += gridSize) {
            let homeCount = 0;
            let awayCount = 0;

            allPlayers.forEach(player => {
                const dx = player.position.x - (x + gridSize / 2);
                const dy = player.position.y - (y + gridSize / 2);
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < gridSize) {
                    if (player.team === 'home') homeCount++;
                    else awayCount++;
                }
            });

            if (homeCount > 0 || awayCount > 0) {
                const total = homeCount + awayCount;
                const homeRatio = homeCount / total;

                const r = Math.round(59 + (239 - 59) * (1 - homeRatio));
                const g = Math.round(130 + (68 - 130) * (1 - homeRatio));
                const b = Math.round(246 + (68 - 246) * (1 - homeRatio));

                ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.2 + total * 0.1})`;
                ctx.fillRect(x * scaleX, y * scaleY, gridSize * scaleX, gridSize * scaleY);
            }
        }
    }
}

function drawGoalieViz() {
    const ctx = goalieCtx;
    const canvas = goalieCanvas;
    const w = canvas.width;
    const h = canvas.height;

    // Clear
    ctx.fillStyle = '#0a0f1a';
    ctx.fillRect(0, 0, w, h);

    const goalie = state.selectedGoalieTeam === 'home' ? state.homeGoalie : state.awayGoalie;
    if (!goalie) return;

    // Draw goal
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.strokeRect(20, 20, w - 40, h - 40);

    // Draw coverage area
    const coverage = goalie.coverage_area / 100;
    const coverageWidth = (w - 40) * coverage;
    const coverageX = (w - coverageWidth) / 2;

    ctx.fillStyle = state.selectedGoalieTeam === 'home' ?
        'rgba(59, 130, 246, 0.3)' : 'rgba(239, 68, 68, 0.3)';
    ctx.fillRect(coverageX, 20, coverageWidth, h - 40);

    // Draw goalkeeper position
    const gkX = w / 2;
    const gkY = h - 30 - goalie.distance_from_line * 2;

    ctx.fillStyle = state.selectedGoalieTeam === 'home' ? COLORS.HOME : COLORS.AWAY;
    ctx.beginPath();
    ctx.arc(gkX, gkY, 8, 0, Math.PI * 2);
    ctx.fill();

    // Quality indicator
    const quality = goalie.positioning_quality;
    ctx.fillStyle = quality > 0.7 ? '#10b981' : quality > 0.4 ? '#f59e0b' : '#ef4444';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Quality: ${(quality * 100).toFixed(0)}%`, w / 2, 12);
}

// ============================================================================
// PLAYBACK CONTROLS
// ============================================================================

function togglePlayback() {
    state.isPlaying = !state.isPlaying;
    updatePlaybackUI();

    if (state.connected) {
        sendCommand({ action: state.isPlaying ? 'play' : 'pause' });
    }
}

function skipFrames(count) {
    if (state.connected) {
        sendCommand({ action: 'skip', value: count });
    } else {
        state.currentFrame = Math.max(0, state.currentFrame + count);
    }
}

function skipSeconds(seconds) {
    const frames = Math.round(seconds * 30);
    skipFrames(frames);
}

function seekToFrame(frame) {
    state.currentFrame = frame;
    if (state.connected) {
        sendCommand({ action: 'seek', value: frame });
    }
}

// ============================================================================
// QUERY INTERFACE
// ============================================================================

function runQuery() {
    const queryType = document.getElementById('queryType').value;
    const team = document.getElementById('queryTeam').value;
    const minXg = parseFloat(document.getElementById('queryMinXg').value) || 0;

    let results = [];

    switch (queryType) {
        case 'shots':
            results = state.shots.filter(s => {
                if (team && s.team !== team) return false;
                if (minXg > 0 && s.xg < minXg) return false;
                return true;
            });
            break;
        case 'xg_moments':
            results = state.shots
                .filter(s => s.xg >= 0.2)
                .sort((a, b) => b.xg - a.xg);
            break;
        case 'player':
            // Would need player events tracking
            results = state.shots.filter(s => s.player_id === state.selectedPlayerId);
            break;
    }

    displayQueryResults(results);
}

function displayQueryResults(results) {
    const container = document.getElementById('queryResults');

    if (!results || results.length === 0) {
        container.innerHTML = '<div class="query-result-item">No results found</div>';
        return;
    }

    container.innerHTML = results.map((item, i) => `
        <div class="query-result-item">
            <strong>#${i + 1}</strong> -
            ${item.time_formatted || formatTime(item.time)} -
            ${capitalize(item.team)} -
            ${item.result} (xG: ${item.xg?.toFixed(3) || 'N/A'})
        </div>
    `).join('');
}

// ============================================================================
// UTILITIES
// ============================================================================

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}
