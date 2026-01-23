import sys
sys.path.insert(0, '..')

from hockey_moneyball_analytics import (
    MoneyballAnalytics, 
    MoneyballConfig,
    NHL_SALARY_DATA,
    EntryType,
    ForecheckStyle
)
import numpy as np
import json

# Initialize moneyball analytics
config = MoneyballConfig(
    frames_per_second=30.0,
    optimal_shift_length=45.0,
    tired_shift_threshold=60.0,
    replacement_level_xg60=0.35
)

analytics = MoneyballAnalytics(config)

# Add Team 0 (Penguins)
analytics.add_player("87", "Sidney Crosby", team=0, position="C", salary=8_700_000)
analytics.add_player("71", "Evgeni Malkin", team=0, position="C", salary=6_100_000)
analytics.add_player("59", "Jake Guentzel", team=0, position="LW", salary=6_000_000)
analytics.add_player("17", "Bryan Rust", team=0, position="RW", salary=5_125_000)
analytics.add_player("58", "Kris Letang", team=0, position="D", salary=6_100_000)

# Add Team 1 (Oilers)
analytics.add_player("97", "Connor McDavid", team=1, position="C", salary=12_500_000)
analytics.add_player("29", "Leon Draisaitl", team=1, position="C", salary=8_500_000)
analytics.add_player("93", "Ryan Nugent-Hopkins", team=1, position="LW", salary=5_125_000)
analytics.add_player("18", "Zach Hyman", team=1, position="LW", salary=5_500_000)
analytics.add_player("25", "Darnell Nurse", team=1, position="D", salary=9_250_000)

# Undervalued ELC players (Entry Level Contract)
analytics.add_player("91", "Wyatt Johnston", team=0, position="C", salary=894_167)  # Cheap but productive
analytics.add_player("53", "Dylan Holloway", team=1, position="LW", salary=863_333)

print(f"Loaded {len(analytics.players)} players")

import random

# Simulate 3 periods of hockey (~54,000 frames at 30fps)
FRAMES_PER_PERIOD = 18000  # 10 minutes simulated
TOTAL_FRAMES = FRAMES_PER_PERIOD * 3

# Define line combinations
team_0_line_1 = ["87", "59", "17"]  # Crosby line
team_0_line_2 = ["71", "91", "58"]  # Malkin line (with undervalued Johnston)
team_1_line_1 = ["97", "29", "93"]  # McDavid line
team_1_line_2 = ["18", "53", "25"]  # Secondary (with undervalued Holloway)

current_lines = {0: [], 1: []}

for frame in range(TOTAL_FRAMES):
    analytics.update_frame(frame)
    
    # Shift changes every ~45 seconds (1350 frames)
    if frame % 1350 == 0:
        # End current shifts
        for team in [0, 1]:
            for pid in current_lines[team]:
                analytics.track_shift_end(pid, frame)
        
        # Start new shifts (alternate lines)
        shift_num = frame // 1350
        if shift_num % 2 == 0:
            current_lines[0] = team_0_line_1
            current_lines[1] = team_1_line_1
        else:
            current_lines[0] = team_0_line_2
            current_lines[1] = team_1_line_2
        
        for team in [0, 1]:
            for pid in current_lines[team]:
                analytics.track_shift_start(pid, frame)
    
    # Update player positions (simulated)
    for team in [0, 1]:
        for i, pid in enumerate(current_lines[team]):
            # Simulate realistic positioning
            base_x = 30 if team == 0 else -30
            x = base_x + random.uniform(-40, 40)
            y = (i - 1) * 15 + random.uniform(-10, 10)
            analytics.update_position(pid, (x, y))
    
    # Scoring chances (realistic frequency: ~60 per game)
    if random.random() < 0.0012:  # ~65 chances per simulated game
        team = random.choice([0, 1])
        shooter = random.choice(current_lines[team])
        
        # Star players generate better chances
        if shooter in ["87", "97", "29"]:
            xg = random.uniform(0.08, 0.35)
        elif shooter in ["91", "53"]:  # Undervalued players performing well
            xg = random.uniform(0.06, 0.25)
        else:
            xg = random.uniform(0.03, 0.18)
        
        # 10% goal rate
        is_goal = random.random() < xg
        
        # 70% of goals have assists
        assists = []
        if is_goal and random.random() < 0.7:
            other_players = [p for p in current_lines[team] if p != shooter]
            assists = random.sample(other_players, min(2, len(other_players)))
        
        analytics.track_scoring_chance(
            player_id=shooter,
            xg=xg,
            frame=frame,
            resulted_in_goal=is_goal,
            assisted_by=assists
        )
    
    # Zone entries (realistic: ~100 per game)
    if random.random() < 0.002:
        team = random.choice([0, 1])
        entry_player = random.choice(current_lines[team])
        
        # Entry type depends on player skill
        if entry_player in ["87", "97", "29", "91"]:
            entry_type = "carry" if random.random() < 0.7 else "dump"
            success = random.random() < 0.65  # Better players succeed more
        else:
            entry_type = "carry" if random.random() < 0.5 else "dump"
            success = random.random() < 0.50
        
        analytics.track_zone_entry(
            player_id=entry_player,
            entry_type=entry_type,
            success=success,
            frame=frame,
            resulted_in_shot=success and random.random() < 0.4,
            xg_generated=random.uniform(0.02, 0.12) if success else 0
        )
    
    # Forecheck events
    if random.random() < 0.001:
        team = random.choice([0, 1])
        analytics.track_forecheck(
            team=team,
            players_involved=current_lines[team],
            frame=frame,
            pressure_duration=random.uniform(2, 8),
            caused_turnover=random.random() < 0.25,
            led_to_chance=random.random() < 0.15
        )

# End final shifts
for team in [0, 1]:
    for pid in current_lines[team]:
        analytics.track_shift_end(pid, TOTAL_FRAMES)

print(f"Simulated {TOTAL_FRAMES} frames ({TOTAL_FRAMES/30/60:.1f} minutes of game time)")

print("=" * 70)
print("UNDERVALUED PLAYERS (Moneyball Finds)")
print("=" * 70)

undervalued = analytics.find_undervalued_players(min_ice_time=5.0)

for player in undervalued:
    print(f"\n{player['name']} ({player['position']})")
    print(f"  Current Salary:   ${player['salary']:>12,}")
    print(f"  Expected Salary:  ${player['expected_salary']:>12,}")
    print(f"  Salary Delta:     ${player['salary_delta']:>+12,} (UNDERVALUED!)")
    print(f"  xG/60:            {player['xg_per_60']:>12.3f}")
    print(f"  Value/Million:    {player['value_per_million']:>12.3f}")
    print(f"  Ice Time:         {player['ice_time_minutes']:>12.1f} min")

print("=" * 70)
print("OVERVALUED PLAYERS (Cap Concerns)")
print("=" * 70)

overvalued = analytics.find_overvalued_players(min_ice_time=5.0)

if not overvalued:
    print("\nNo significantly overvalued players found.")
else:
    for player in overvalued:
        print(f"\n{player['name']} ({player['position']})")
        print(f"  Current Salary:   ${player['salary']:>12,}")
        print(f"  Expected Salary:  ${player['expected_salary']:>12,}")
        print(f"  Salary Delta:     ${player['salary_delta']:>+12,} (OVERVALUED)")
        print(f"  xG/60:            {player['xg_per_60']:>12.3f}")

print("=" * 70)
print("BEST PERFORMING LINE COMBINATIONS (by xG Differential)")
print("=" * 70)

best_lines = analytics.get_best_line_combinations(min_ice_time=1.0, top_n=5)

for i, line in enumerate(best_lines, 1):
    print(f"\n#{i}: {' - '.join(line['players'])}")
    print(f"  Ice Time:        {line['ice_time_minutes']:>8.1f} min")
    print(f"  Combined Salary: ${line['combined_salary']:>12,}")
    print(f"  xG For/60:       {line['xg_for_60']:>8.3f}")
    print(f"  xG Against/60:   {line['xg_against_60']:>8.3f}")
    print(f"  xG Diff/60:      {line['xg_differential_60']:>+8.3f}")

print("=" * 70)
print("BEST VALUE LINE COMBINATIONS (Performance per $Million)")
print("=" * 70)

value_lines = analytics.get_best_value_lines(min_ice_time=1.0, top_n=5)

for i, line in enumerate(value_lines, 1):
    print(f"\n#{i}: {' - '.join(line['players'])}")
    print(f"  Combined Salary:   ${line['combined_salary']:>12,}")
    print(f"  xG Differential:   {line['xg_differential_60']:>+8.3f}/60")
    print(f"  Value/Million:     {line['value_per_million']:>8.3f} (Higher = Better Value)")
    print(f"  Zone Entry Rate:   {line['entry_success_rate']:>8.1%}")

print("=" * 70)
print("SHIFT EFFICIENCY ANALYSIS")
print("=" * 70)

for player_id in ["87", "97", "91", "53"]:  # Mix of stars and undervalued
    report = analytics.get_shift_efficiency_report(player_id)
    
    if report.get('no_data'):
        continue
    
    print(f"\n{report['name']}:")
    print(f"  Total Shifts:           {report['total_shifts']}")
    print(f"  Avg Shift Length:       {report['avg_shift_length']:.1f} sec")
    print(f"  Recommended Length:     {report['recommended_shift_length']:.0f} sec")
    print(f"  Tired Shift %:          {report['tired_shift_percentage']:.1%}")
    print(f"  Avg Speed Dropoff:      {report['avg_speed_dropoff']:.1%}")
    
    if report.get('optimal_shifts', {}).get('count', 0) > 0:
        opt = report['optimal_shifts']
        print(f"  Optimal Shift xG:       {opt['xg_per_shift']:.3f}/shift")

print("=" * 70)
print("ZONE ENTRY ANALYSIS")
print("=" * 70)

for team, name in [(0, "Penguins"), (1, "Oilers")]:
    analysis = analytics.get_zone_entry_analysis(team)
    
    if analysis.get('no_data'):
        print(f"\n{name}: No data")
        continue
    
    print(f"\n{name}:")
    print(f"  Total Entries:      {analysis['total_entries']}")
    print(f"  Carry-In %:         {analysis['carry_percentage']:.1%}")
    
    carry = analysis['carry_in']
    dump = analysis['dump_in']
    
    print(f"\n  Carry-In Success:   {carry['success_rate']:.1%} (league avg: 55%)")
    print(f"  Carry-In Shot Rate: {carry['shot_rate']:.1%}")
    print(f"  Carry-In Avg xG:    {carry['avg_xg']:.3f}")
    
    print(f"\n  Dump-In Success:    {dump['success_rate']:.1%} (league avg: 45%)")
    print(f"  Dump-In Shot Rate:  {dump['shot_rate']:.1%}")
    
    print(f"\n  Recommendation:     {analysis['recommendation'].upper()}")

print("=" * 70)
print("FORECHECKING PATTERN ANALYSIS")
print("=" * 70)

for team, name in [(0, "Penguins"), (1, "Oilers")]:
    analysis = analytics.get_forecheck_analysis(team)
    
    if analysis.get('no_data'):
        print(f"\n{name}: No data")
        continue
    
    print(f"\n{name}:")
    print(f"  Total Forecheck Events:  {analysis['total_forecheck_events']}")
    print(f"  Dominant Style:          {analysis['dominant_style']}")
    print(f"  Aggression Score:        {analysis['aggression_score']:.2f} (0=passive, 1=aggressive)")
    
    if analysis['is_aggressive']:
        print(f"  Classification:          AGGRESSIVE (high pressure)")
    elif analysis['is_passive']:
        print(f"  Classification:          PASSIVE (trap/conservative)")
    else:
        print(f"  Classification:          BALANCED")
    
    print(f"\n  Turnover Rate:           {analysis['overall_turnover_rate']:.1%}")
    print(f"  Scoring Chance Rate:     {analysis['overall_chance_rate']:.1%}")
    print(f"  Avg Pressure Duration:   {analysis['avg_pressure_duration']:.1f} sec")

# Generate comprehensive report
report = analytics.get_moneyball_report()

print("=" * 70)
print("FULL MONEYBALL REPORT SUMMARY")
print("=" * 70)

print(f"\nTimestamp: {report['timestamp']}")
print(f"Frames Analyzed: {report['frame_count']:,}")

print(f"\nUndervalued Players Found: {len(report['undervalued_players'])}")
print(f"Overvalued Players Found: {len(report['overvalued_players'])}")
print(f"Line Combinations Tracked: {len(report['best_performing_lines'])}")

# Export to file
analytics.export_report('moneyball_report.json')
print("\nFull report exported to: moneyball_report.json")
