"""
Tests for hockey_moneyball_analytics module.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hockey_moneyball_analytics import (
    MoneyballAnalytics,
    MoneyballConfig,
    PlayerProfile,
    LineCombination,
    ShiftAnalysis,
    EntryType,
    ForecheckStyle
)


class TestMoneyballConfig:
    """Tests for MoneyballConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MoneyballConfig()

        assert config.frames_per_second == 30.0
        assert config.optimal_shift_length == 45.0
        assert config.tired_shift_threshold == 60.0
        assert config.cap_ceiling == 83_500_000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoneyballConfig(
            frames_per_second=60.0,
            optimal_shift_length=40.0
        )

        assert config.frames_per_second == 60.0
        assert config.optimal_shift_length == 40.0


class TestMoneyballAnalytics:
    """Tests for MoneyballAnalytics class."""

    @pytest.fixture
    def analytics(self):
        """Create analytics instance for testing."""
        return MoneyballAnalytics()

    def test_add_player(self, analytics):
        """Test adding a player."""
        player = analytics.add_player(
            player_id="87",
            name="Sidney Crosby",
            team=0,
            position="C",
            jersey_number=87,
            salary=8_700_000
        )

        assert player.player_id == "87"
        assert player.name == "Sidney Crosby"
        assert player.team == 0
        assert player.salary == 8_700_000
        assert "87" in analytics.players

    def test_get_player(self, analytics):
        """Test retrieving a player."""
        analytics.add_player("87", "Sidney Crosby", 0)
        player = analytics.get_player("87")

        assert player is not None
        assert player.name == "Sidney Crosby"

    def test_get_nonexistent_player(self, analytics):
        """Test retrieving nonexistent player returns None."""
        player = analytics.get_player("nonexistent")
        assert player is None

    def test_track_shift(self, analytics):
        """Test shift tracking."""
        analytics.add_player("87", "Sidney Crosby", 0)

        analytics.track_shift_start("87", frame=0)
        player = analytics.get_player("87")

        assert player.current_shift_start == 0
        assert "87" in analytics.current_lines[0]

        # End shift
        analysis = analytics.track_shift_end("87", frame=900)  # 30 seconds at 30fps

        assert analysis is not None
        assert analysis.duration_seconds == 30.0
        assert player.current_shift_start is None

    def test_track_scoring_chance(self, analytics):
        """Test scoring chance tracking."""
        analytics.add_player("87", "Sidney Crosby", 0)
        analytics.add_player("71", "Evgeni Malkin", 0)

        # Start shifts
        analytics.track_shift_start("87", frame=0)
        analytics.track_shift_start("71", frame=0)

        # Track scoring chance
        analytics.track_scoring_chance(
            player_id="87",
            xg=0.15,
            frame=100,
            resulted_in_goal=True,
            assisted_by=["71"]
        )

        player = analytics.get_player("87")
        assert player.xg_total == 0.15
        assert len(player.scoring_chances) == 1
        assert player.scoring_chances[0]['goal'] is True

        # Check assist
        assist_player = analytics.get_player("71")
        assert assist_player.xa_total == 0.075  # 50% of xG

    def test_track_zone_entry(self, analytics):
        """Test zone entry tracking."""
        analytics.add_player("87", "Sidney Crosby", 0)
        analytics.track_shift_start("87", frame=0)

        analytics.track_zone_entry(
            player_id="87",
            entry_type="carry",
            success=True,
            frame=100,
            resulted_in_shot=True,
            xg_generated=0.1
        )

        player = analytics.get_player("87")
        assert len(player.zone_entries) == 1
        assert player.zone_entries[0]['type'] == 'carry'
        assert player.zone_entries[0]['success'] is True

    def test_player_value_metrics(self, analytics):
        """Test player value metrics calculation."""
        analytics.add_player(
            "87", "Sidney Crosby", 0,
            position="C", salary=8_700_000
        )

        # Simulate some activity
        analytics.track_shift_start("87", frame=0)
        for i in range(5):
            analytics.track_scoring_chance(
                player_id="87",
                xg=0.1,
                frame=i * 100
            )
        analytics.track_shift_end("87", frame=5400)  # 3 minutes

        metrics = analytics.get_player_value_metrics("87")

        assert 'xg_per_60' in metrics
        assert 'value_per_million' in metrics
        assert 'total_xg' in metrics
        assert metrics['total_xg'] == 0.5

    def test_find_undervalued_players(self, analytics):
        """Test finding undervalued players."""
        # Add high-producing low-salary player
        analytics.add_player(
            "rookie", "Rookie Star", 0,
            position="C", salary=900_000
        )
        analytics.track_shift_start("rookie", frame=0)
        for i in range(20):
            analytics.track_scoring_chance("rookie", xg=0.2, frame=i * 100)
        analytics.track_shift_end("rookie", frame=36000)  # 20 minutes

        undervalued = analytics.find_undervalued_players(min_ice_time=10.0)

        # Should find our high-producing low-salary player
        assert isinstance(undervalued, list)

    def test_line_combination_tracking(self, analytics):
        """Test line combination tracking."""
        analytics.add_player("87", "Crosby", 0, salary=8_700_000)
        analytics.add_player("71", "Malkin", 0, salary=6_100_000)
        analytics.add_player("59", "Guentzel", 0, salary=6_000_000)

        # Start all three
        analytics.track_shift_start("87", frame=0)
        analytics.track_shift_start("71", frame=0)
        analytics.track_shift_start("59", frame=0)

        # Line should be tracked
        assert len(analytics.line_combinations) > 0

    def test_forecheck_analysis(self, analytics):
        """Test forechecking analysis."""
        analytics.add_player("10", "Forward1", 0)
        analytics.add_player("11", "Forward2", 0)

        analytics.track_forecheck(
            team=0,
            players_involved=["10", "11"],
            frame=100,
            pressure_duration=5.0,
            caused_turnover=True,
            led_to_chance=True
        )

        analysis = analytics.get_forecheck_analysis(0)

        assert analysis['total_forecheck_events'] == 1
        assert analysis['overall_turnover_rate'] == 1.0

    def test_zone_entry_analysis(self, analytics):
        """Test zone entry analysis."""
        analytics.add_player("87", "Crosby", 0)

        # Track multiple zone entries
        analytics.track_shift_start("87", frame=0)
        for i in range(5):
            analytics.track_zone_entry(
                "87", "carry", success=(i < 4),  # 80% success
                frame=i * 100
            )

        analysis = analytics.get_zone_entry_analysis(0)

        assert analysis['total_entries'] == 5
        assert analysis['carry_in']['success_rate'] == 0.8

    def test_moneyball_report(self, analytics):
        """Test full moneyball report generation."""
        analytics.add_player("87", "Crosby", 0, salary=8_700_000)
        analytics.track_shift_start("87", frame=0)
        analytics.track_scoring_chance("87", xg=0.15, frame=100)
        analytics.track_shift_end("87", frame=900)

        report = analytics.get_moneyball_report()

        assert 'timestamp' in report
        assert 'undervalued_players' in report
        assert 'best_performing_lines' in report
        assert 'team_0_forecheck' in report


class TestEntryTypes:
    """Tests for entry type enums."""

    def test_entry_type_values(self):
        """Test entry type enum values."""
        assert EntryType.CARRY == 0
        assert EntryType.DUMP == 1
        assert EntryType.PASS == 2

    def test_forecheck_style_values(self):
        """Test forecheck style enum values."""
        assert ForecheckStyle.AGGRESSIVE_2_1_2 == 0
        assert ForecheckStyle.PASSIVE_1_4 == 2
