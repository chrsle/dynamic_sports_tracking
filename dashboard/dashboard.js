/**
 * Hockey Analytics Dashboard
 * Real-time analytics visualization
 */

class HockeyDashboard {
    constructor() {
        this.websocket = null;
        this.charts = {};
        this.rinkView = 'heatmap';
        this.heatmapData = null;
        this.playerPositions = { home: [], away: [] };
        this.puckPosition = null;

        // New tracking data
        this.shots = [];
        this.goaliePositions = {
            home: { x: -85, y: 0, depth: 5, angle: 28, quality: 'optimal' },
            away: { x: 85, y: 0, depth: 5, angle: 28, quality: 'optimal' }
        };
        this.powerPlay = {
            is_power_play: false,
            team_with_advantage: null,
            home_players: 5,
            away_players: 5,
            time_remaining: 0,
            type: 'even'
        };
        this.xgTimeline = [];

        this.init();
    }

    init() {
        this.initCharts();
        this.initRinkCanvas();
        this.initEventListeners();
        this.connectWebSocket();

        // Start with demo data if no WebSocket
        setTimeout(() => {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                this.startDemoMode();
            }
        }, 2000);
    }

    // ==================== Charts ====================

    initCharts() {
        // xG Comparison Chart
        this.charts.xg = new ApexCharts(document.getElementById('xg-chart'), {
            chart: {
                type: 'bar',
                height: 60,
                sparkline: { enabled: true },
                toolbar: { show: false }
            },
            series: [{
                name: 'Home xG',
                data: [0]
            }, {
                name: 'Away xG',
                data: [0]
            }],
            plotOptions: {
                bar: {
                    horizontal: true,
                    barHeight: '70%',
                    distributed: false
                }
            },
            colors: ['#1a73e8', '#ea4335'],
            grid: { padding: { top: 0, bottom: 0 } },
            xaxis: { categories: ['xG'] },
            tooltip: { enabled: false }
        });
        this.charts.xg.render();

        // xG Timeline Chart
        this.charts.xgTimeline = new ApexCharts(document.getElementById('xg-timeline-chart'), {
            chart: {
                type: 'area',
                height: 100,
                sparkline: { enabled: false },
                toolbar: { show: false },
                animations: {
                    enabled: true,
                    dynamicAnimation: { speed: 300 }
                },
                zoom: { enabled: false }
            },
            series: [{
                name: 'Home xG',
                data: []
            }, {
                name: 'Away xG',
                data: []
            }],
            stroke: {
                curve: 'smooth',
                width: 2
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shadeIntensity: 1,
                    opacityFrom: 0.5,
                    opacityTo: 0.1
                }
            },
            colors: ['#1a73e8', '#ea4335'],
            xaxis: {
                type: 'numeric',
                labels: {
                    show: true,
                    formatter: (val) => Math.floor(val / 60) + ':' + String(Math.floor(val % 60)).padStart(2, '0'),
                    style: { fontSize: '10px', colors: '#9aa0a6' }
                },
                axisBorder: { show: false },
                axisTicks: { show: false }
            },
            yaxis: {
                labels: {
                    show: true,
                    formatter: (val) => val.toFixed(1),
                    style: { fontSize: '10px', colors: '#9aa0a6' }
                }
            },
            grid: {
                show: true,
                borderColor: '#e8eaed',
                strokeDashArray: 3,
                padding: { left: 10, right: 10 }
            },
            tooltip: {
                enabled: true,
                shared: true,
                x: {
                    formatter: (val) => Math.floor(val / 60) + ':' + String(Math.floor(val % 60)).padStart(2, '0')
                }
            },
            legend: { show: false }
        });
        this.charts.xgTimeline.render();

        // Scoring Chances Chart
        this.charts.chances = new ApexCharts(document.getElementById('chances-chart'), {
            chart: {
                type: 'area',
                height: 120,
                sparkline: { enabled: true },
                toolbar: { show: false },
                animations: {
                    enabled: true,
                    dynamicAnimation: { speed: 500 }
                }
            },
            series: [{
                name: 'Home',
                data: []
            }, {
                name: 'Away',
                data: []
            }],
            stroke: {
                curve: 'smooth',
                width: 2
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shadeIntensity: 1,
                    opacityFrom: 0.4,
                    opacityTo: 0.1
                }
            },
            colors: ['#1a73e8', '#ea4335'],
            tooltip: {
                enabled: true,
                x: { show: false }
            }
        });
        this.charts.chances.render();
    }

    // ==================== Rink Canvas ====================

    initRinkCanvas() {
        this.canvas = document.getElementById('rink-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.drawRink();
    }

    drawRink() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, w, h);

        // Ice background
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, w, h);

        // Rink outline
        ctx.strokeStyle = '#9aa0a6';
        ctx.lineWidth = 2;
        ctx.strokeRect(5, 5, w - 10, h - 10);

        // Center line (red)
        ctx.strokeStyle = '#ea4335';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(w / 2, 5);
        ctx.lineTo(w / 2, h - 5);
        ctx.stroke();

        // Blue lines
        ctx.strokeStyle = '#1a73e8';
        ctx.lineWidth = 2;

        // Left blue line
        ctx.beginPath();
        ctx.moveTo(w * 0.35, 5);
        ctx.lineTo(w * 0.35, h - 5);
        ctx.stroke();

        // Right blue line
        ctx.beginPath();
        ctx.moveTo(w * 0.65, 5);
        ctx.lineTo(w * 0.65, h - 5);
        ctx.stroke();

        // Goal lines
        ctx.strokeStyle = '#ea4335';
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(w * 0.06, 5);
        ctx.lineTo(w * 0.06, h - 5);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(w * 0.94, 5);
        ctx.lineTo(w * 0.94, h - 5);
        ctx.stroke();

        // Center circle
        ctx.strokeStyle = '#ea4335';
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, 30, 0, Math.PI * 2);
        ctx.stroke();

        // Center dot
        ctx.fillStyle = '#ea4335';
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, 4, 0, Math.PI * 2);
        ctx.fill();

        // Face-off circles
        const faceoffPositions = [
            { x: w * 0.19, y: h * 0.27 },
            { x: w * 0.19, y: h * 0.73 },
            { x: w * 0.81, y: h * 0.27 },
            { x: w * 0.81, y: h * 0.73 }
        ];

        ctx.strokeStyle = '#1a73e8';
        faceoffPositions.forEach(pos => {
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
            ctx.stroke();

            ctx.fillStyle = '#ea4335';
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
            ctx.fill();
        });

        // Goal creases
        ctx.fillStyle = 'rgba(26, 115, 232, 0.1)';
        ctx.strokeStyle = '#1a73e8';

        // Left crease
        ctx.beginPath();
        ctx.arc(w * 0.03, h / 2, 20, -Math.PI / 2, Math.PI / 2);
        ctx.fill();
        ctx.stroke();

        // Right crease
        ctx.beginPath();
        ctx.arc(w * 0.97, h / 2, 20, Math.PI / 2, -Math.PI / 2);
        ctx.fill();
        ctx.stroke();
    }

    drawHeatmap(data) {
        if (!data) return;

        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Draw base rink first
        this.drawRink();

        // Draw heatmap overlay
        const gridW = data[0].length;
        const gridH = data.length;
        const cellW = w / gridW;
        const cellH = h / gridH;

        for (let y = 0; y < gridH; y++) {
            for (let x = 0; x < gridW; x++) {
                const value = data[y][x];
                if (value > 0.1) {
                    const alpha = Math.min(value * 0.6, 0.7);
                    ctx.fillStyle = `rgba(234, 67, 53, ${alpha})`;
                    ctx.fillRect(x * cellW, y * cellH, cellW, cellH);
                }
            }
        }
    }

    drawPositions() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Draw base rink
        this.drawRink();

        // Draw home team players
        ctx.fillStyle = '#1a73e8';
        this.playerPositions.home.forEach(pos => {
            const x = (pos.x / 200 + 0.5) * w;
            const y = (pos.y / 85 + 0.5) * h;
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw away team players
        ctx.fillStyle = '#ea4335';
        this.playerPositions.away.forEach(pos => {
            const x = (pos.x / 200 + 0.5) * w;
            const y = (pos.y / 85 + 0.5) * h;
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw goalies (larger, distinctive markers)
        this.drawGoalies(ctx, w, h);

        // Draw puck
        if (this.puckPosition) {
            ctx.fillStyle = '#1a1a2e';
            const px = (this.puckPosition.x / 200 + 0.5) * w;
            const py = (this.puckPosition.y / 85 + 0.5) * h;
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    drawGoalies(ctx, w, h) {
        // Draw home goalie
        if (this.goaliePositions.home) {
            const pos = this.goaliePositions.home;
            const x = (pos.x / 200 + 0.5) * w;
            const y = (pos.y / 85 + 0.5) * h;

            // Goalie marker (star shape)
            ctx.fillStyle = '#1a73e8';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            this.drawStar(ctx, x, y, 5, 12, 6);
            ctx.fill();
            ctx.stroke();

            // Quality indicator ring
            ctx.strokeStyle = this.getQualityColor(pos.quality);
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(x, y, 15, 0, Math.PI * 2);
            ctx.stroke();
        }

        // Draw away goalie
        if (this.goaliePositions.away) {
            const pos = this.goaliePositions.away;
            const x = (pos.x / 200 + 0.5) * w;
            const y = (pos.y / 85 + 0.5) * h;

            ctx.fillStyle = '#ea4335';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            this.drawStar(ctx, x, y, 5, 12, 6);
            ctx.fill();
            ctx.stroke();

            ctx.strokeStyle = this.getQualityColor(pos.quality);
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(x, y, 15, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    drawStar(ctx, cx, cy, spikes, outerRadius, innerRadius) {
        let rot = Math.PI / 2 * 3;
        let x = cx;
        let y = cy;
        const step = Math.PI / spikes;

        ctx.beginPath();
        ctx.moveTo(cx, cy - outerRadius);
        for (let i = 0; i < spikes; i++) {
            x = cx + Math.cos(rot) * outerRadius;
            y = cy + Math.sin(rot) * outerRadius;
            ctx.lineTo(x, y);
            rot += step;

            x = cx + Math.cos(rot) * innerRadius;
            y = cy + Math.sin(rot) * innerRadius;
            ctx.lineTo(x, y);
            rot += step;
        }
        ctx.lineTo(cx, cy - outerRadius);
        ctx.closePath();
    }

    getQualityColor(quality) {
        switch (quality) {
            case 'optimal': return '#34a853';
            case 'good': return '#4285f4';
            case 'vulnerable': return '#fbbc04';
            case 'out_of_position': return '#ea4335';
            default: return '#9aa0a6';
        }
    }

    drawShotMap() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Draw base rink
        this.drawRink();

        // Draw each shot
        this.shots.forEach(shot => {
            const x = (shot.x / 200 + 0.5) * w;
            const y = (shot.y / 85 + 0.5) * h;

            // Size based on xG
            const radius = 4 + shot.xg * 20;

            // Color based on result
            let fillColor, strokeColor;
            switch (shot.result) {
                case 'goal':
                    fillColor = '#34a853';
                    strokeColor = '#2d9248';
                    break;
                case 'save':
                    fillColor = '#4285f4';
                    strokeColor = '#3b78e7';
                    break;
                case 'miss':
                    fillColor = 'rgba(154, 160, 166, 0.5)';
                    strokeColor = '#9aa0a6';
                    break;
                case 'block':
                    fillColor = '#fbbc04';
                    strokeColor = '#e8ab00';
                    break;
                default:
                    fillColor = '#9aa0a6';
                    strokeColor = '#5f6368';
            }

            // Team indicator (border)
            ctx.strokeStyle = shot.team === 'home' ? '#1a73e8' : '#ea4335';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
            ctx.stroke();

            // Shot marker
            ctx.fillStyle = fillColor;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // Goal indicator (special)
            if (shot.result === 'goal') {
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 10px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('G', x, y);
            }
        });

        // Draw goalies on shot map too
        this.drawGoalies(ctx, w, h);
    }

    drawHotspots(hotspots) {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Draw base rink
        this.drawRink();

        // Draw hotspot circles
        if (hotspots && hotspots.home) {
            ctx.fillStyle = 'rgba(26, 115, 232, 0.3)';
            ctx.strokeStyle = '#1a73e8';
            hotspots.home.forEach(hs => {
                const x = (hs.x / 200 + 0.5) * w;
                const y = (hs.y / 85 + 0.5) * h;
                const r = Math.sqrt(hs.size) * 3;
                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            });
        }

        if (hotspots && hotspots.away) {
            ctx.fillStyle = 'rgba(234, 67, 53, 0.3)';
            ctx.strokeStyle = '#ea4335';
            hotspots.away.forEach(hs => {
                const x = (hs.x / 200 + 0.5) * w;
                const y = (hs.y / 85 + 0.5) * h;
                const r = Math.sqrt(hs.size) * 3;
                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            });
        }
    }

    // ==================== Event Listeners ====================

    initEventListeners() {
        // View toggle buttons
        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.rinkView = e.target.dataset.view;
                this.updateRinkView();
            });
        });
    }

    updateRinkView() {
        // Show/hide appropriate legends
        const rinkLegend = document.getElementById('rink-legend');
        const shotLegend = document.getElementById('shot-legend');

        if (this.rinkView === 'shotmap') {
            rinkLegend.style.display = 'none';
            shotLegend.style.display = 'flex';
        } else {
            rinkLegend.style.display = 'flex';
            shotLegend.style.display = 'none';
        }

        switch (this.rinkView) {
            case 'heatmap':
                this.drawHeatmap(this.heatmapData);
                break;
            case 'positions':
                this.drawPositions();
                break;
            case 'shotmap':
                this.drawShotMap();
                break;
            case 'hotspots':
                this.drawHotspots(this.hotspots);
                break;
        }
    }

    // ==================== WebSocket ====================

    connectWebSocket() {
        try {
            this.websocket = new WebSocket('ws://localhost:8765');

            this.websocket.onopen = () => {
                console.log('Connected to analytics server');
                document.querySelector('.badge.live').style.background = '#34a853';
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.updateDashboard(data);
            };

            this.websocket.onclose = () => {
                console.log('Disconnected from server');
                document.querySelector('.badge.live').style.background = '#ea4335';
                setTimeout(() => this.connectWebSocket(), 5000);
            };

            this.websocket.onerror = (error) => {
                console.log('WebSocket error:', error);
            };
        } catch (e) {
            console.log('WebSocket not available, using demo mode');
        }
    }

    // ==================== Update Functions ====================

    updateDashboard(data) {
        if (data.teams) {
            document.getElementById('home-team').textContent = data.teams.home;
            document.getElementById('away-team').textContent = data.teams.away;
        }

        if (data.signal) {
            this.updateSignal(data.signal);
        }

        if (data.momentum) {
            this.updateMomentum(data.momentum);
        }

        if (data.xg) {
            this.updateXG(data.xg);
        }

        if (data.zones) {
            this.updateZones(data.zones);
        }

        if (data.positions) {
            this.updatePositions(data.positions);
        }

        if (data.insights) {
            this.updateInsights(data.insights);
        }

        if (data.heatmap) {
            this.heatmapData = data.heatmap;
        }

        if (data.hotspots) {
            this.hotspots = data.hotspots;
        }

        if (data.playerPositions) {
            this.playerPositions = data.playerPositions;
        }

        if (data.puckPosition) {
            this.puckPosition = data.puckPosition;
        }

        // New data handling
        if (data.shots) {
            this.shots = data.shots;
            this.updateShotStats();
        }

        if (data.goaliePositions) {
            this.goaliePositions = data.goaliePositions;
            this.updateGoalieDisplay();
        }

        if (data.powerPlay) {
            this.powerPlay = data.powerPlay;
            this.updatePowerPlayDisplay();
        }

        if (data.xgTimeline) {
            this.xgTimeline = data.xgTimeline;
            this.updateXGTimeline();
        }

        this.updateRinkView();
    }

    updateShotStats() {
        const homeShots = this.shots.filter(s => s.team === 'home');
        const awayShots = this.shots.filter(s => s.team === 'away');

        const homeGoals = homeShots.filter(s => s.result === 'goal').length;
        const awayGoals = awayShots.filter(s => s.result === 'goal').length;

        const homeSaves = awayShots.filter(s => s.result === 'save').length;
        const awaySaves = homeShots.filter(s => s.result === 'save').length;

        const homeAvgXG = homeShots.length > 0
            ? homeShots.reduce((sum, s) => sum + s.xg, 0) / homeShots.length
            : 0;
        const awayAvgXG = awayShots.length > 0
            ? awayShots.reduce((sum, s) => sum + s.xg, 0) / awayShots.length
            : 0;

        document.getElementById('shots-home').textContent = homeShots.length;
        document.getElementById('shots-away').textContent = awayShots.length;
        document.getElementById('goals-home').textContent = homeGoals;
        document.getElementById('goals-away').textContent = awayGoals;
        document.getElementById('saves-home').textContent = homeSaves;
        document.getElementById('saves-away').textContent = awaySaves;
        document.getElementById('avg-xg-home').textContent = homeAvgXG.toFixed(2);
        document.getElementById('avg-xg-away').textContent = awayAvgXG.toFixed(2);

        // Update score display
        document.getElementById('score').textContent = `${homeGoals} - ${awayGoals}`;
    }

    updateGoalieDisplay() {
        // Home goalie
        if (this.goaliePositions.home) {
            const home = this.goaliePositions.home;
            const qualityEl = document.getElementById('goalie-home-quality');
            qualityEl.textContent = home.quality.charAt(0).toUpperCase() + home.quality.slice(1).replace('_', ' ');
            qualityEl.className = 'goalie-quality ' + home.quality;

            document.getElementById('goalie-home-depth').textContent = home.depth.toFixed(1);
            document.getElementById('goalie-home-angle').textContent = Math.round(home.angle);
            document.getElementById('goalie-home-position').textContent = `(${home.y.toFixed(0)}, ${(home.x + 100).toFixed(0)})`;
        }

        // Away goalie
        if (this.goaliePositions.away) {
            const away = this.goaliePositions.away;
            const qualityEl = document.getElementById('goalie-away-quality');
            qualityEl.textContent = away.quality.charAt(0).toUpperCase() + away.quality.slice(1).replace('_', ' ');
            qualityEl.className = 'goalie-quality ' + away.quality;

            document.getElementById('goalie-away-depth').textContent = away.depth.toFixed(1);
            document.getElementById('goalie-away-angle').textContent = Math.round(away.angle);
            document.getElementById('goalie-away-position').textContent = `(${away.y.toFixed(0)}, ${(100 - away.x).toFixed(0)})`;
        }
    }

    updatePowerPlayDisplay() {
        const ppIndicator = document.getElementById('power-play-indicator');
        const ppBadge = document.getElementById('pp-badge');
        const ppType = document.getElementById('pp-type');
        const ppTime = document.getElementById('pp-time');

        if (this.powerPlay.is_power_play) {
            ppIndicator.style.display = 'flex';
            ppIndicator.className = 'power-play-indicator ' +
                (this.powerPlay.team_with_advantage === 'home' ? 'home-advantage' : 'away-advantage');

            ppBadge.textContent = this.powerPlay.team_with_advantage === 'home' ? 'PP' : 'PK';
            ppType.textContent = this.powerPlay.type;

            const mins = Math.floor(this.powerPlay.time_remaining / 60);
            const secs = Math.floor(this.powerPlay.time_remaining % 60);
            ppTime.textContent = `${mins}:${String(secs).padStart(2, '0')}`;
        } else {
            ppIndicator.style.display = 'none';
        }
    }

    updateXGTimeline() {
        if (this.xgTimeline.length === 0) return;

        // Prepare data for chart
        const homeData = this.xgTimeline.map(p => ({ x: p.time, y: p.home_xg }));
        const awayData = this.xgTimeline.map(p => ({ x: p.time, y: p.away_xg }));

        this.charts.xgTimeline.updateSeries([
            { name: 'Home xG', data: homeData },
            { name: 'Away xG', data: awayData }
        ]);
    }

    updateSignal(signal) {
        const indicator = document.getElementById('signal-indicator');
        const text = document.getElementById('signal-text');
        const fill = document.getElementById('confidence-fill');
        const value = document.getElementById('confidence-value');

        // Remove old classes
        indicator.className = 'signal-indicator';

        // Set signal text and class
        text.textContent = signal.text || 'NEUTRAL';

        if (signal.text.includes('STRONG_HOME')) {
            indicator.classList.add('strong-home');
        } else if (signal.text.includes('LEAN_HOME')) {
            indicator.classList.add('lean-home');
        } else if (signal.text.includes('STRONG_AWAY')) {
            indicator.classList.add('strong-away');
        } else if (signal.text.includes('LEAN_AWAY')) {
            indicator.classList.add('lean-away');
        }

        // Update confidence
        const confidence = (signal.confidence || 0.5) * 100;
        fill.style.width = confidence + '%';
        value.textContent = Math.round(confidence) + '%';

        // Update reasons
        const reasonsContainer = document.getElementById('signal-reasons');
        reasonsContainer.innerHTML = '';

        if (signal.reasons) {
            signal.reasons.forEach(reason => {
                const div = document.createElement('div');
                div.className = 'reason-item';
                div.textContent = reason;
                reasonsContainer.appendChild(div);
            });
        }
    }

    updateMomentum(momentum) {
        const marker = document.getElementById('momentum-marker');
        const trend = document.getElementById('momentum-trend');

        // Position marker (momentum -1 to 1 maps to 0% to 100%)
        const position = ((momentum.value || 0) + 1) / 2 * 100;
        marker.style.left = position + '%';

        // Update trend
        const trendIcon = trend.querySelector('.trend-icon');
        const trendText = trend.querySelector('.trend-text');

        if (momentum.trend === 'increasing_home') {
            trendIcon.textContent = '↗';
            trendText.textContent = 'Home building';
            trendIcon.style.color = '#1a73e8';
        } else if (momentum.trend === 'increasing_away') {
            trendIcon.textContent = '↘';
            trendText.textContent = 'Away building';
            trendIcon.style.color = '#ea4335';
        } else {
            trendIcon.textContent = '→';
            trendText.textContent = 'Stable';
            trendIcon.style.color = '#9aa0a6';
        }
    }

    updateXG(xg) {
        document.getElementById('xg-home').textContent = (xg.home || 0).toFixed(2);
        document.getElementById('xg-away').textContent = (xg.away || 0).toFixed(2);

        // Update chart
        this.charts.xg.updateSeries([
            { data: [xg.home || 0] },
            { data: [xg.away || 0] }
        ]);
    }

    updateZones(zones) {
        // Update zone bars
        const total = (zones.home?.offensive || 0) + (zones.home?.neutral || 0) + (zones.home?.defensive || 0);

        if (total > 0) {
            const offHome = ((zones.home?.offensive || 0) / total * 100).toFixed(1);
            const neuHome = ((zones.home?.neutral || 0) / total * 100).toFixed(1);
            const defHome = ((zones.home?.defensive || 0) / total * 100).toFixed(1);

            document.getElementById('zone-off-home').style.width = offHome + '%';
            document.getElementById('zone-neu-home').style.width = neuHome + '%';
            document.getElementById('zone-def-home').style.width = defHome + '%';
        }
    }

    updatePositions(positions) {
        document.getElementById('pos-optimal').textContent = positions.optimal || 0;
        document.getElementById('pos-good').textContent = positions.good || 0;
        document.getElementById('pos-suboptimal').textContent = positions.suboptimal || 0;
        document.getElementById('pos-critical').textContent = positions.critical || 0;
    }

    updateInsights(insights) {
        const feed = document.getElementById('insights-feed');

        insights.forEach(insight => {
            const div = document.createElement('div');
            div.className = 'insight-item';

            const icon = document.createElement('span');
            icon.className = 'insight-icon';
            icon.textContent = insight.icon || '→';

            const text = document.createElement('span');
            text.className = 'insight-text';
            text.textContent = insight.text;

            const time = document.createElement('span');
            time.className = 'insight-time';
            time.textContent = 'now';

            div.appendChild(icon);
            div.appendChild(text);
            div.appendChild(time);

            feed.insertBefore(div, feed.firstChild);

            // Limit to 10 insights
            while (feed.children.length > 10) {
                feed.removeChild(feed.lastChild);
            }
        });
    }

    // ==================== Demo Mode ====================

    startDemoMode() {
        console.log('Starting demo mode');

        // Initialize demo xG timeline
        this._demoTime = 0;
        this._demoXgHome = 0;
        this._demoXgAway = 0;

        // Initial data
        this.updateDashboard({
            teams: { home: 'Penguins', away: 'Oilers' },
            signal: {
                text: 'LEAN_HOME',
                confidence: 0.62,
                reasons: ['Home dominating zone time', 'Quality chances generated']
            },
            momentum: {
                value: 0.25,
                trend: 'increasing_home'
            },
            xg: { home: 1.23, away: 0.87 },
            zones: {
                home: { offensive: 45, neutral: 30, defensive: 25 },
                away: { offensive: 25, neutral: 30, defensive: 45 }
            },
            positions: {
                optimal: 3,
                good: 4,
                suboptimal: 2,
                critical: 1
            },
            shots: [],
            goaliePositions: {
                home: { x: -85, y: 0, depth: 5, angle: 28, quality: 'optimal' },
                away: { x: 85, y: 0, depth: 5, angle: 28, quality: 'optimal' }
            },
            powerPlay: {
                is_power_play: false,
                team_with_advantage: null,
                home_players: 5,
                away_players: 5,
                time_remaining: 0,
                type: 'even'
            },
            xgTimeline: []
        });

        // Add some demo insights
        this.updateInsights([
            { icon: '🔥', text: 'Home team surge detected' },
            { icon: '📊', text: 'xG advantage: Home +0.36' }
        ]);

        // Generate demo player positions
        this.generateDemoPositions();

        // Generate some initial shots
        this.generateDemoShots(5);

        // Update periodically
        setInterval(() => this.updateDemoData(), 2000);
    }

    generateDemoShots(count) {
        for (let i = 0; i < count; i++) {
            const team = Math.random() > 0.45 ? 'home' : 'away';
            let x, y;

            if (team === 'home') {
                x = 50 + Math.random() * 35;
                y = (Math.random() - 0.5) * 50;
            } else {
                x = -50 - Math.random() * 35;
                y = (Math.random() - 0.5) * 50;
            }

            const goalX = team === 'home' ? 89 : -89;
            const dist = Math.sqrt((x - goalX) ** 2 + y ** 2);
            const xg = Math.max(0.02, Math.min(0.5, 0.4 - dist * 0.008));

            const rand = Math.random();
            let result;
            if (rand < xg) result = 'goal';
            else if (rand < xg + 0.6) result = 'save';
            else if (rand < xg + 0.75) result = 'miss';
            else result = 'block';

            this.shots.push({
                x, y, team, xg, result,
                time: this._demoTime || 0,
                period: 1
            });
        }
        this.updateShotStats();
    }

    generateDemoPositions() {
        this.playerPositions = {
            home: [
                { x: -60, y: 0 },    // Goalie
                { x: -30, y: -20 },  // D
                { x: -30, y: 20 },   // D
                { x: 20, y: -25 },   // LW
                { x: 30, y: 0 },     // C
                { x: 20, y: 25 }     // RW
            ],
            away: [
                { x: 60, y: 0 },     // Goalie
                { x: 30, y: -20 },   // D
                { x: 30, y: 20 },    // D
                { x: -20, y: -25 },  // LW
                { x: -30, y: 0 },    // C
                { x: -20, y: 25 }    // RW
            ]
        };

        this.puckPosition = { x: 15, y: 5 };
    }

    updateDemoData() {
        // Increment demo time
        this._demoTime = (this._demoTime || 0) + 2;

        // Random momentum shift
        const momentumShift = (Math.random() - 0.5) * 0.1;
        const newMomentum = Math.max(-1, Math.min(1, this._lastMomentum + momentumShift || 0.2));
        this._lastMomentum = newMomentum;

        // Update signal based on momentum
        let signalText = 'NEUTRAL';
        if (newMomentum > 0.5) signalText = 'STRONG_HOME';
        else if (newMomentum > 0.2) signalText = 'LEAN_HOME';
        else if (newMomentum < -0.5) signalText = 'STRONG_AWAY';
        else if (newMomentum < -0.2) signalText = 'LEAN_AWAY';

        // Update xG
        this._demoXgHome = (this._demoXgHome || 1.23) + (newMomentum > 0 ? Math.random() * 0.03 : Math.random() * 0.01);
        this._demoXgAway = (this._demoXgAway || 0.87) + (newMomentum < 0 ? Math.random() * 0.03 : Math.random() * 0.01);

        // Add to xG timeline
        this.xgTimeline.push({
            time: this._demoTime,
            home_xg: this._demoXgHome,
            away_xg: this._demoXgAway
        });
        // Keep last 100 points
        if (this.xgTimeline.length > 100) {
            this.xgTimeline = this.xgTimeline.slice(-100);
        }

        // Move players slightly
        this.playerPositions.home.forEach(p => {
            p.x += (Math.random() - 0.4) * 5;
            p.y += (Math.random() - 0.5) * 3;
        });

        this.playerPositions.away.forEach(p => {
            p.x += (Math.random() - 0.6) * 5;
            p.y += (Math.random() - 0.5) * 3;
        });

        // Move puck
        this.puckPosition.x += (Math.random() - 0.4) * 10;
        this.puckPosition.y += (Math.random() - 0.5) * 5;

        // Update goalie positions (subtle movements)
        this.goaliePositions.home = {
            x: -85 + (Math.random() - 0.5) * 6,
            y: (Math.random() - 0.5) * 16,
            depth: 5 + (Math.random() - 0.5) * 6,
            angle: 28 + (Math.random() - 0.5) * 10,
            quality: Math.random() > 0.85 ? 'vulnerable' : Math.random() > 0.7 ? 'good' : 'optimal'
        };
        this.goaliePositions.away = {
            x: 85 + (Math.random() - 0.5) * 6,
            y: (Math.random() - 0.5) * 16,
            depth: 5 + (Math.random() - 0.5) * 6,
            angle: 28 + (Math.random() - 0.5) * 10,
            quality: Math.random() > 0.85 ? 'vulnerable' : Math.random() > 0.7 ? 'good' : 'optimal'
        };

        // Occasionally generate a shot
        if (Math.random() > 0.85) {
            this.generateDemoShots(1);
        }

        // Power play simulation (occasional)
        if (Math.random() > 0.97) {
            // Start power play
            const ppTeam = Math.random() > 0.5 ? 'home' : 'away';
            this.powerPlay = {
                is_power_play: true,
                team_with_advantage: ppTeam,
                home_players: ppTeam === 'home' ? 5 : 4,
                away_players: ppTeam === 'away' ? 5 : 4,
                time_remaining: 120,
                type: '5v4'
            };
        } else if (this.powerPlay.is_power_play) {
            // Decrement power play time
            this.powerPlay.time_remaining -= 2;
            if (this.powerPlay.time_remaining <= 0) {
                this.powerPlay = {
                    is_power_play: false,
                    team_with_advantage: null,
                    home_players: 5,
                    away_players: 5,
                    time_remaining: 0,
                    type: 'even'
                };
            }
        }

        this.updateDashboard({
            signal: {
                text: signalText,
                confidence: 0.5 + Math.abs(newMomentum) * 0.4,
                reasons: newMomentum > 0 ?
                    ['Home controlling possession', 'Offensive zone pressure'] :
                    newMomentum < 0 ?
                        ['Away generating chances', 'Defensive zone time'] :
                        ['Even play']
            },
            momentum: {
                value: newMomentum,
                trend: momentumShift > 0.03 ? 'increasing_home' :
                    momentumShift < -0.03 ? 'increasing_away' : 'stable'
            },
            xg: { home: this._demoXgHome, away: this._demoXgAway },
            positions: {
                optimal: Math.floor(Math.random() * 3) + 2,
                good: Math.floor(Math.random() * 4) + 3,
                suboptimal: Math.floor(Math.random() * 3) + 1,
                critical: Math.floor(Math.random() * 2)
            },
            shots: this.shots,
            goaliePositions: this.goaliePositions,
            powerPlay: this.powerPlay,
            xgTimeline: this.xgTimeline
        });

        // Occasional insight
        if (Math.random() > 0.7) {
            const insights = [
                { icon: '⚡', text: 'Power play opportunity' },
                { icon: '🎯', text: 'High-danger chance created' },
                { icon: '📈', text: 'Momentum shifting' },
                { icon: '🔄', text: 'Line change detected' },
                { icon: '🥅', text: 'Shot on goal' },
                { icon: '🧤', text: 'Goalie tracking puck' },
                { icon: '🏒', text: 'Zone entry attempt' }
            ];
            this.updateInsights([insights[Math.floor(Math.random() * insights.length)]]);
        }

        this.updateRinkView();
    }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new HockeyDashboard();
});
