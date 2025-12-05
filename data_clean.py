import nfl_data_py as nfl
import pandas as pd


def load_and_prepare_nfl_data(seasons=range(2014, 2024)):
    seasons = list(seasons)

    inj = nfl.import_injuries(seasons)
    weekly = nfl.import_weekly_data(seasons)
    snaps = nfl.import_snap_counts(seasons)
    sched = nfl.import_schedules(seasons)
    players = nfl.import_players()


    inj_clean = inj[['season', 'week', 'gsis_id', 'report_status', 'practice_status']].copy()


    inj_clean['on_report'] = inj_clean[['report_status', 'practice_status']].notna().any(axis=1).astype(int)

    # Ensure season/week are int
    inj_clean['season'] = inj_clean['season'].astype(int)
    inj_clean['week'] = inj_clean['week'].astype(int)

    # Collapse to one row per season-week-player
    inj_week = inj_clean.groupby(
        ['season', 'week', 'gsis_id'], as_index=False
    ).agg({
        'on_report': 'max',
        'report_status': lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else None,
        'practice_status': lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else None
    })

    # Shift forward
    inj_next = inj_week.copy()
    inj_next['week'] = inj_next['week'] - 1
    inj_next = inj_next.rename(columns={
        'on_report': 'y',
        'report_status': 'next_report_status',
        'practice_status': 'next_practice_status'
    })

    df = weekly.copy()

    if 'player_id' in df.columns and 'gsis_id' not in df.columns:
        df = df.rename(columns={'player_id': 'gsis_id'})

    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    df = df.merge(
        inj_next[['season', 'week', 'gsis_id', 'y', 'next_report_status', 'next_practice_status']],
        on=['season', 'week', 'gsis_id'],
        how='left'
    )

    # Fill missing labels as 0 (not on next-week report)
    df['y'] = df['y'].fillna(0).astype(int)

    # Sort for time-dependent features
    df = df.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)


    # Lag-1 and rolling-3 features for numeric columns
    non_numeric_cols = [
        'gsis_id', 'player_name', 'player_display_name', 'position',
        'position_group', 'headshot_url', 'recent_team', 'season',
        'week', 'season_type', 'opponent_team',
        'next_report_status', 'next_practice_status', 'y'
    ]

    numeric_cols = [c for c in df.columns if c not in non_numeric_cols]

    for col in numeric_cols:
        df[f"{col}_lag1"] = df.groupby("gsis_id")[col].shift(1)
        df[f"{col}_roll3"] = (
            df.groupby("gsis_id")[col]
              .rolling(3, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    df = df.fillna(0)

    # 1 hote encode
    position_dummies = pd.get_dummies(df["position"], prefix="pos")
    df = pd.concat([df, position_dummies], axis=1)

    drop_cols = [
        'gsis_id', 'player_name', 'player_display_name', 'position',
        'position_group', 'headshot_url', 'recent_team', 'season',
        'week', 'season_type', 'opponent_team',
        'next_report_status', 'next_practice_status', 'y'
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df, feature_cols


if __name__ == "__main__":
    df, feature_cols = load_and_prepare_nfl_data()
    print(df.shape)
    print("Number of features:", len(feature_cols))
